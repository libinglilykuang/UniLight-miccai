import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random
import time
import subprocess
import sys
from monai.networks.nets import UNet
from thop import profile

# 从你的模块中导入
from models.student import LATUPStudent
from data.dataset import get_5fold_split, BraTS2023Dataset, MSDBTSDataset, precompute_sdfs_for_dataset
from utils.loss_metrics import (
    WeightedDiceBCELoss, DWKDLoss, BoundaryLoss,
    calculate_dice, calculate_hd95_per_channel
)

# ============================================================
# 1. 核心配置 (根据你的脚本整合)
# ============================================================
DATASET_TYPE = "brats2023"  # 👈 切换此处: "brats2023" 或 "msd"
CURRENT_FOLD = 4
NUM_FOLDS = 5
SEED = 42

# 路径配置 (自动适配)
if DATASET_TYPE == "brats2023":
    DATA_DIR = "/kaggle/input/bratsglioma/Training"
    TEACHER_WEIGHTS = "/kaggle/input/teacher4/pytorch/default/1/best_teacher_2023.pth"
    RESUME_LAST_PATH = "/kaggle/input/last-student4-2/pytorch/default/1/last_student_2023.pth"
    RESUME_BEST_PATH = "/kaggle/input/best-student4-2/pytorch/default/1/best_student_2023.pt"
else:
    DATA_DIR = "/kaggle/input/msd-brain-tumour/Task01_BrainTumour"
    TEACHER_WEIGHTS = "/kaggle/input/teacher5/pytorch/default/1/best_teacher_msd.pth"
    RESUME_LAST_PATH = "/kaggle/input/last-student-4-2/pytorch/default/1/last_student.pth"
    RESUME_BEST_PATH = "/kaggle/input/best-student-4-2/pytorch/default/1/best_student (1).pt"

SAVE_DIR = f"./kd_results_{DATASET_TYPE}_fold_{CURRENT_FOLD}"
SDF_DIR = os.path.join(SAVE_DIR, "sdfs")
os.makedirs(SDF_DIR, exist_ok=True)

BEST_PATH = os.path.join(SAVE_DIR, "best_student.pt")
LAST_PATH = os.path.join(SAVE_DIR, "last_student.pth")

TOTAL_EPOCHS = 100
BATCH_SIZE = 2
LR = 1e-4
ROI_SIZE = (128, 128, 128)
VAL_INTERVAL = 5
TEMPERATURE = 3.0

# 损失权重 (完全复刻)
ALPHA_TASK = 0.6
BETA_DWKD = 0.2
USE_RCE = True
GAMMA_RCE = 0.1
RCE_T = 0.9
USE_BOUNDARY = True
BOUNDARY_WEIGHT = 0.1


# ============================================================
# 2. 随机性与环境初始化
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)


# ============================================================
# 3. 效率评测函数 (复刻你的 measure_efficiency)
# ============================================================
def try_import_thop():
    try:
        from thop import profile
        return profile
    except Exception:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "thop"])
            from thop import profile
            return profile
        except Exception as e:
            return None

def measure_efficiency(model, name, roi_size=(128, 128, 128)):
    params = sum(p.numel() for p in model.parameters())
    flops_g = None
    profile = try_import_thop()
    if profile is not None:
        try:
            if name.lower().startswith("teach"):
                probe = UNet(
                    spatial_dims=3, in_channels=4, out_channels=3,
                    channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), num_res_units=2
                ).cpu().eval()
            else:
                # Reconstruct student for CPU probe (must disable deep supervision for flop count or handle tuple)
                probe = LATUPStudent(in_channels=4, out_channels=3, base_filters=16,
                                     deep_supervision=False).cpu().eval()

            dummy_cpu = torch.randn(1, 4, *roi_size)
            flops, _ = profile(probe, inputs=(dummy_cpu,), verbose=False)
            flops_g = flops / 1e9
        except Exception as e:
            print(f"[EFF] {name} FLOPs 计算失败：{e}")

    model_device = next(model.parameters()).device
    dummy = torch.randn(1, 4, *roi_size).to(model_device)
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)
            if model_device.type == "cuda":
                torch.cuda.synchronize()
    times = []
    runs = 10
    with torch.no_grad():
        if model_device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            for _ in range(runs):
                start_event.record()
                _ = model(dummy)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))
        else:
            for _ in range(runs):
                t0 = time.time()
                _ = model(dummy)
                t1 = time.time()
                times.append((t1 - t0) * 1000.0)
    times.sort()
    avg_ms = sum(times) / len(times)
    p50_ms = times[len(times) // 2]
    p95_ms = times[min(len(times) - 1, int(0.95 * (len(times) - 1)))]

    peak_mb = None
    if model_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy)
            torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"\n[EFF] {name} | Params: {params:,} (~{params / 1e6:.2f} M)")
    if flops_g is not None:
        print(f"[EFF] {name} | FLOPs (forward, ROI={roi_size}): {flops_g:.3f} G")
    else:
        print(f"[EFF] {name} | FLOPs: 未计算（thop 不可用）")
    print(
        f"[EFF] {name} | Inference Time (N=1, ROI={roi_size}): avg {avg_ms:.2f} ms | p50 {p50_ms:.2f} ms | p95 {p95_ms:.2f} ms")
    if peak_mb is not None:
        print(f"[EFF] {name} | Peak GPU Memory: {peak_mb:.1f} MB")
    else:
        print(f"[EFF] {name} | Peak GPU Memory: CPU 环境或未获取")

    return {
        "params_m": params / 1e6,
        "flops_g": flops_g,
        "lat_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "peak_mb": peak_mb,
    }

# ============================================================
# 4. 主训练流程
# ============================================================
def main():
    # --- 数据划分与加载 ---
    train_cases, val_cases = get_5fold_split(DATA_DIR, CURRENT_FOLD, dataset_type=DATASET_TYPE)
    precompute_sdfs_for_dataset(DATA_DIR, train_cases + val_cases, SDF_DIR, ROI_SIZE)

    if DATASET_TYPE == "brats2023":
        train_ds = BraTS2023Dataset(train_cases, SDF_DIR, ROI_SIZE, augment=True)
        val_ds = BraTS2023Dataset(val_cases, SDF_DIR, ROI_SIZE, augment=False)
    else:
        train_ds = MSDBTSDataset(DATA_DIR, train_cases, SDF_DIR, ROI_SIZE, augment=True)
        val_ds = MSDBTSDataset(DATA_DIR, val_cases, SDF_DIR, ROI_SIZE, augment=False)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

    # --- 模型初始化 ---
    teacher_model = UNet(spatial_dims=3, in_channels=4, out_channels=3,
                         channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), num_res_units=2).to(device)
    teacher_model.load_state_dict(torch.load(TEACHER_WEIGHTS, map_location='cpu'), strict=False)
    teacher_model.eval()

    student_model = LATUPStudent(in_channels=4, out_channels=3, base_filters=16, deep_supervision=True).to(device)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LR, weight_decay=1e-5)

    # --- 损失函数 ---
    task_loss_fn = WeightedDiceBCELoss(weights=[1.0, 1.0, 3.0])
    dwkd_loss_fn = DWKDLoss(temperature=TEMPERATURE, use_rce=USE_RCE, rce_t=RCE_T)
    boundary_loss_fn = BoundaryLoss()

    # --- 智能续训逻辑 (完全保留你的 Resume 逻辑) ---
    start_epoch = 1
    best_dice = 0.0

    if RESUME_LAST_PATH is not None and os.path.exists(RESUME_LAST_PATH):
        print(f"🔄 正在加载 Last 权重以恢复训练状态: {RESUME_LAST_PATH}")
        try:
            checkpoint = torch.load(RESUME_LAST_PATH, map_location=device)

            # 加载模型
            if 'model_state_dict' in checkpoint:
                student_model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                student_model.load_state_dict(checkpoint['state_dict'])
            else:
                student_model.load_state_dict(checkpoint)
            print("✅ 学生模型参数已恢复")

            # 加载优化器 (保持动量)
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("✅ 优化器状态已恢复")
                except Exception as e:
                    print(f"⚠️ 优化器加载失败 (可能参数不匹配)，使用新优化器: {e}")

            # 恢复 Epoch
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"✅ 起始 Epoch 更新为: {start_epoch}")

            # 暂时设置 best_dice 为 last 权重中记录的值
            if 'best_dice' in checkpoint:
                best_dice = checkpoint['best_dice']
                print(f"ℹ️ Last 权重记录的 Best Dice: {best_dice:.4f}")

        except Exception as e:
            print(f"❌ 加载 Last 权重失败，将从头开始训练: {e}")
    else:
        print(f"⚠️ 未找到 Last 权重文件或路径未配置，准备从头开始 (或仅基于 Best 权重)")

    # 2. 尝试加载 Best 权重（仅用于校准 best_dice 分数，防止覆盖）
    # 修复点：增加 (RESUME_BEST_PATH is not None) 的判断
    if RESUME_BEST_PATH is not None and os.path.exists(RESUME_BEST_PATH):
        print(f"🔍 检查 Best 权重文件以校准分数: {RESUME_BEST_PATH}")
        try:
            # map_location='cpu' 节省显存，只需要读取数字
            ckpt_best = torch.load(RESUME_BEST_PATH, map_location='cpu')

            # 如果我们没有成功加载 Last 权重，那么必须加载 Best 的模型参数作为起点
            # (同样检查 RESUME_LAST_PATH 是否存在或为None)
            last_exists = (RESUME_LAST_PATH is not None and os.path.exists(RESUME_LAST_PATH))

            if start_epoch == 1 and not last_exists:
                print("⚠️ 未找到 Last 权重，将使用 Best 权重作为训练起点...")
                if 'model_state_dict' in ckpt_best:
                    student_model.load_state_dict(ckpt_best['model_state_dict'])
                elif 'state_dict' in ckpt_best:
                    student_model.load_state_dict(ckpt_best['state_dict'])
                else:
                    student_model.load_state_dict(ckpt_best)

                if 'epoch' in ckpt_best:
                    start_epoch = ckpt_best['epoch'] + 1

            # 核心：校准 best_dice
            if 'best_dice' in ckpt_best:
                actual_best = ckpt_best['best_dice']
                if actual_best > best_dice:
                    print(f"✅ 校准 Best Dice: 从 {best_dice:.4f} 更新为历史真实最佳 {actual_best:.4f}")
                    best_dice = actual_best
                else:
                    print(f"ℹ️ 当前记录 ({best_dice:.4f}) 已优于或等于文件记录 ({actual_best:.4f})")

        except Exception as e:
            print(f"⚠️ 读取 Best 权重信息失败 (不影响训练): {e}")

    print(f"🚀 最终训练起点: Epoch {start_epoch}, 当前需超越的 Best Dice: {best_dice:.4f}")

    # --- 训练循环 ---
    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        student_model.train()
        epoch_loss = 0.0
        train_dices = []
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{TOTAL_EPOCHS}')

        for batch in pbar:
            # 修改：直接从 dataloader 解包 (image, seg, sdf, name)
            images, labels, sdfs, case_names = batch
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            sdfs = sdfs.to(device, non_blocking=True)  # SDF 已经和图像做了完全一样的增强

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            student_outputs = student_model(images)

            # [Modified] Deep Supervision Handling
            if isinstance(student_outputs, list):
                logits_main = student_outputs[0]
                logits_d2 = student_outputs[1]
                logits_d3 = student_outputs[2]

                # 1. Main Loss (Task + DWKD + Boundary)
                L_task = task_loss_fn(logits_main, labels)
                L_dwkdl, L_rce = dwkd_loss_fn(logits_main, teacher_outputs)

                L_boundary = torch.tensor(0.0, device=device)
                if USE_BOUNDARY:
                    L_boundary = boundary_loss_fn(logits_main, labels, precomputed_sdfs=sdfs)

                loss_main = ALPHA_TASK * L_task + \
                            BETA_DWKD * L_dwkdl + \
                            (GAMMA_RCE * L_rce if USE_RCE else 0.0) + \
                            (BOUNDARY_WEIGHT * L_boundary if USE_BOUNDARY else 0.0)

                # 2. Auxiliary Losses (Downsampled Task Loss)
                # D2 is 1/2 scale
                labels_d2 = F.interpolate(labels, scale_factor=0.5, mode='nearest')
                loss_d2 = ALPHA_TASK * task_loss_fn(logits_d2, labels_d2)

                # D3 is 1/4 scale
                labels_d3 = F.interpolate(labels, scale_factor=0.25, mode='nearest')
                loss_d3 = ALPHA_TASK * task_loss_fn(logits_d3, labels_d3)

                # Total Loss
                loss = loss_main + 0.5 * loss_d2 + 0.25 * loss_d3

                # For metrics use main output
                probs = torch.sigmoid(logits_main)

            else:
                # Fallback if DS disabled
                L_task = task_loss_fn(student_outputs, labels)
                L_dwkdl, L_rce = dwkd_loss_fn(student_outputs, teacher_outputs)

                L_boundary = torch.tensor(0.0, device=device)
                if USE_BOUNDARY:
                    L_boundary = boundary_loss_fn(student_outputs, labels, precomputed_sdfs=sdfs)

                loss = ALPHA_TASK * L_task + \
                       BETA_DWKD * L_dwkdl + \
                       (GAMMA_RCE * L_rce if USE_RCE else 0.0) + \
                       (BOUNDARY_WEIGHT * L_boundary if USE_BOUNDARY else 0.0)

                probs = torch.sigmoid(student_outputs)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = (probs > 0.5).float()
                train_dices.append(calculate_dice(preds, labels))

            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Dice': f'{np.mean(train_dices[-1]):.3f}'})

        epoch_loss /= max(len(train_loader), 1)
        mean_train_dice = float(np.mean([np.mean(d) for d in train_dices])) if train_dices else 0.0

        # 验证
        if epoch % VAL_INTERVAL == 0 or epoch == 1:
            student_model.eval()
            val_dices, val_hd95s = [], []
            with torch.no_grad():
                for batch in val_loader:
                    # 修改：解包 (image, seg, sdf, name)
                    images, labels, _, case_names = batch
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = student_model(images)
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    val_dices.append(calculate_dice(preds, labels))
                    val_hd95s.append(calculate_hd95_per_channel(preds, labels))

            val_dices_array = np.array(val_dices) if len(val_dices) > 0 else np.zeros((1, 3))
            mean_val_dice = float(np.mean(val_dices_array))
            wt_dice = float(np.mean(val_dices_array[:, 0]))
            tc_dice = float(np.mean(val_dices_array[:, 1]))
            et_dice = float(np.mean(val_dices_array[:, 2]))

            val_hd95_array = np.array(val_hd95s) if len(val_hd95s) > 0 else np.full((1, 3), np.inf, dtype=np.float32)
            finite_mask = np.isfinite(val_hd95_array)
            mean_val_hd95 = float(np.mean(val_hd95_array[finite_mask])) if finite_mask.any() else float('inf')
            wt_hd95 = float(np.nanmean(val_hd95_array[:, 0]))
            tc_hd95 = float(np.nanmean(val_hd95_array[:, 1]))
            et_hd95 = float(np.nanmean(val_hd95_array[:, 2]))

            print(f'Epoch {epoch}:')
            print(f'  训练损失 = {epoch_loss:.4f}, 训练Dice = {mean_train_dice:.4f}')
            print(f'  验证Dice = {mean_val_dice:.4f} | WT {wt_dice:.4f} / TC {tc_dice:.4f} / ET {et_dice:.4f}')
            print(f'  验证HD95(mm) = {mean_val_hd95:.2f} | WT {wt_hd95:.2f} / TC {tc_hd95:.2f} / ET {et_hd95:.2f}')

            if mean_val_dice > best_dice:
                best_dice = mean_val_dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': student_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'val_dices': [wt_dice, tc_dice, et_dice],
                    'val_hd95': [wt_hd95, tc_hd95, et_hd95],
                    'seed': 42,
                    'use_rce': USE_RCE,
                    'tau': TEMPERATURE,
                    'alpha_task': ALPHA_TASK,
                    'beta_dwkd': BETA_DWKD,
                    'gamma_rce': GAMMA_RCE,
                    'rce_t': RCE_T,
                    'use_boundary': USE_BOUNDARY,
                    'boundary_weight': BOUNDARY_WEIGHT
                }, BEST_PATH)
                print(f'  ✅ 保存最佳学生模型到 {BEST_PATH}, Dice = {best_dice:.4f}')
            else:
                print(f'Epoch {epoch}: 训练损失 = {epoch_loss:.4f}, 训练Dice = {mean_train_dice:.4f}')

        torch.save({
            'epoch': epoch,
            'model_state_dict': student_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice,
            'seed': 42,
            'use_rce': USE_RCE,
            'tau': TEMPERATURE,
            'alpha_task': ALPHA_TASK,
            'beta_dwkd': BETA_DWKD,
            'gamma_rce': GAMMA_RCE,
            'rce_t': RCE_T,
            'use_boundary': USE_BOUNDARY,
            'boundary_weight': BOUNDARY_WEIGHT
        }, LAST_PATH)

    print(f"\n训练完成! (best 验证 Dice: {best_dice:.4f})")

    # --- 最终评测 ---
    print("\n===== 最终效率评估 =====")
    measure_efficiency(teacher_model, "Teacher")
    measure_efficiency(student_model, "Student")


if __name__ == "__main__":
    main()