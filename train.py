
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

# 从你的模块中导入
from models.student import LATUPStudent
# ✅ 修改点1: 导入工厂函数 get_datasets
from data.dataset import get_datasets
from utils.loss_metrics import (
    WeightedDiceBCELoss, DWKDLoss, BoundaryLoss,
    calculate_dice, calculate_hd95_per_channel
)

# ============================================================
# 1. 核心配置
# ============================================================
DATASET_TYPE = "brats2023"  # 👈 切换 "brats2023" 或 "msd"
CURRENT_FOLD = 4
NUM_FOLDS = 5
SEED = 42

# 路径配置
if DATASET_TYPE == "brats2023":
    DATA_DIR = "/kaggle/input/bratsglioma/Training"
    TEACHER_WEIGHTS = "/kaggle/input/models/xiaoyouyou198/teacher2/pytorch/default/1/best_teacher_2023.pth"
    RESUME_LAST_PATH = None
    RESUME_BEST_PATH = None
else:
    DATA_DIR = "/kaggle/input/msd-brain-tumour/Task01_BrainTumour"
    TEACHER_WEIGHTS = "/kaggle/input/models/xiaoyouyou198/teacher5/pytorch/default/1/best_teacher_msd.pth"
    RESUME_LAST_PATH = None
    RESUME_BEST_PATH = None

SAVE_DIR = f"./kd_results_{DATASET_TYPE}_fold_{CURRENT_FOLD}"
SDF_DIR = os.path.join(SAVE_DIR, "sdfs")
os.makedirs(SDF_DIR, exist_ok=True)

BEST_PATH = os.path.join(SAVE_DIR, "best_student.pt")
LAST_PATH = os.path.join(SAVE_DIR, "last_student.pth")

TOTAL_EPOCHS = 100  # 建议设回 100，之前测试用的 1 太短
BATCH_SIZE = 2
LR = 1e-4
ROI_SIZE = (128, 128, 128)
VAL_INTERVAL = 5
TEMPERATURE = 3.0

# 损失权重
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
# 3. 效率评测函数
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

    # 简单的探针模型构建，用于计算 FLOPs
    if profile is not None:
        try:
            # 临时构建对应的 CPU 模型以避免显存问题
            if name.lower().startswith("teach"):
                probe = UNet(spatial_dims=3, in_channels=4, out_channels=3,
                             channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), num_res_units=2).cpu().eval()
            else:
                probe = LATUPStudent(in_channels=4, out_channels=3, base_filters=16,
                                     deep_supervision=False).cpu().eval()

            dummy_cpu = torch.randn(1, 4, *roi_size)
            flops, _ = profile(probe, inputs=(dummy_cpu,), verbose=False)
            flops_g = flops / 1e9
        except Exception as e:
            print(f"[EFF] {name} FLOPs error: {e}")

    model_device = next(model.parameters()).device
    dummy = torch.randn(1, 4, *roi_size).to(model_device)
    model.eval()

    # 预热
    with torch.no_grad():
        for _ in range(5): _ = model(dummy)

    # 测速
    times = []
    runs = 20
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

    print(f"\n[EFF] {name} | Params: {params / 1e6:.2f} M | FLOPs: {flops_g if flops_g else 'N/A'} G")
    print(f"[EFF] {name} | Latency: {avg_ms:.2f} ms")


# ============================================================
# 4. 主训练流程
# ============================================================
def main():
    # --- ✅ 修改点2: 使用 get_datasets 一键加载数据 ---
    print(f"🔄 正在加载数据集 (Mode: {DATASET_TYPE})...")

    train_ds, val_ds = get_datasets(
        data_dir=DATA_DIR,
        fold=CURRENT_FOLD,
        num_folds=NUM_FOLDS,
        seed=SEED,
        sdf_dir=SDF_DIR,
        roi_size=ROI_SIZE
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # --- 模型初始化 ---
    print("🛠️ 初始化模型...")
    teacher_model = UNet(
        spatial_dims=3, in_channels=4, out_channels=3,
        channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), num_res_units=2
    ).to(device)

    try:
        teacher_ckpt = torch.load(TEACHER_WEIGHTS, map_location='cpu')
        sd = teacher_ckpt['model_state_dict'] if 'model_state_dict' in teacher_ckpt else teacher_ckpt
        teacher_model.load_state_dict(sd, strict=False)
        print("✅ 老师模型加载成功")
    except Exception as e:
        print(f"❌ 老师模型加载失败: {e}")

    teacher_model.eval()

    student_model = LATUPStudent(in_channels=4, out_channels=3, base_filters=16, deep_supervision=True).to(device)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LR, weight_decay=1e-5)

    # --- 损失函数 ---
    task_loss_fn = WeightedDiceBCELoss(weights=[1.0, 1.0, 3.0])
    dwkd_loss_fn = DWKDLoss(temperature=TEMPERATURE, use_rce=USE_RCE, rce_t=RCE_T)
    boundary_loss_fn = BoundaryLoss()

    # --- 智能续训逻辑 ---
    start_epoch = 1
    best_dice = 0.0

    # 1. 加载 Last (恢复状态)
    if RESUME_LAST_PATH and os.path.exists(RESUME_LAST_PATH):
        print(f"🔄 Resume Last: {RESUME_LAST_PATH}")
        ckpt = torch.load(RESUME_LAST_PATH, map_location=device)
        student_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_dice = ckpt.get('best_dice', 0.0)

    # 2. 检查 Best (校准分数)
    if RESUME_BEST_PATH and os.path.exists(RESUME_BEST_PATH):
        ckpt_best = torch.load(RESUME_BEST_PATH, map_location='cpu')
        actual_best = ckpt_best.get('best_dice', 0.0)
        if actual_best > best_dice:
            best_dice = actual_best
            print(f"✅ 校准 Best Dice 为: {best_dice:.4f}")

            # 如果没加载 Last，就从 Best 开始
            if not (RESUME_LAST_PATH and os.path.exists(RESUME_LAST_PATH)):
                print("⚠️ 无 Last 权重，从 Best 权重开始微调...")
                student_model.load_state_dict(ckpt_best['model_state_dict'])

    print(f"🚀 开始训练: Epoch {start_epoch} - {TOTAL_EPOCHS}, Best Dice: {best_dice:.4f}")

    # --- 训练循环 ---
    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        student_model.train()
        epoch_loss = 0.0
        train_dices = []
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{TOTAL_EPOCHS}')

        for batch in pbar:
            images, labels, sdfs, _ = batch
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            sdfs = sdfs.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_outputs = teacher_model(images)

            student_outputs = student_model(images)

            # Deep Supervision 处理
            if isinstance(student_outputs, (list, tuple)):
                logits_main = student_outputs[0]
                logits_d2 = student_outputs[1]
                logits_d3 = student_outputs[2]

                # Main Loss
                L_task = task_loss_fn(logits_main, labels)
                L_dwkdl, L_rce = dwkd_loss_fn(logits_main, teacher_outputs)

                L_boundary = torch.tensor(0.0, device=device)
                if USE_BOUNDARY:
                    # ✅ 修改点3: 修复 BoundaryLoss 调用，补充 labels 参数
                    L_boundary = boundary_loss_fn(logits_main, labels, precomputed_sdfs=sdfs)

                loss_main = ALPHA_TASK * L_task + \
                            BETA_DWKD * L_dwkdl + \
                            (GAMMA_RCE * L_rce if USE_RCE else 0.0) + \
                            (BOUNDARY_WEIGHT * L_boundary if USE_BOUNDARY else 0.0)

                # Aux Loss
                labels_d2 = F.interpolate(labels, scale_factor=0.5, mode='nearest')
                loss_d2 = ALPHA_TASK * task_loss_fn(logits_d2, labels_d2)

                labels_d3 = F.interpolate(labels, scale_factor=0.25, mode='nearest')
                loss_d3 = ALPHA_TASK * task_loss_fn(logits_d3, labels_d3)

                loss = loss_main + 0.5 * loss_d2 + 0.25 * loss_d3

                probs = torch.sigmoid(logits_main)

            else:
                # No DS
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

        # --- 验证 ---
        if epoch % VAL_INTERVAL == 0 or epoch == 1:
            student_model.eval()
            val_dices, val_hd95s = [], []
            with torch.no_grad():
                for batch in val_loader:
                    images, labels, _, _ = batch
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                    outputs = student_model(images)

                    # ✅ 修改点4: 验证时处理 Deep Supervision 输出 (Tuple -> Tensor)
                    if isinstance(outputs, (list, tuple)):
                        outputs = outputs[0]

                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()

                    val_dices.append(calculate_dice(preds, labels))
                    val_hd95s.append(calculate_hd95_per_channel(preds, labels))

            # 指标计算
            val_dices_array = np.array(val_dices) if len(val_dices) > 0 else np.zeros((1, 3))
            mean_val_dice = float(np.mean(val_dices_array))

            val_hd95_array = np.array(val_hd95s) if len(val_hd95s) > 0 else np.full((1, 3), np.inf)
            mean_val_hd95 = float(np.nanmean(val_hd95_array))

            print(f'Epoch {epoch}:')
            print(f'  Train Loss={epoch_loss:.4f}, Dice={mean_train_dice:.4f}')
            print(
                f'  Val Dice={mean_val_dice:.4f} (WT={val_dices_array[:, 0].mean():.4f}, TC={val_dices_array[:, 1].mean():.4f}, ET={val_dices_array[:, 2].mean():.4f})')
            print(f'  Val HD95={mean_val_hd95:.2f}')

            # 保存 Best
            if mean_val_dice > best_dice:
                best_dice = mean_val_dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': student_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                }, BEST_PATH)
                print(f'  ✅ Saving Best Model: {best_dice:.4f}')
            else:
                print(f'  (Best was {best_dice:.4f})')

        # 保存 Last
        torch.save({
            'epoch': epoch,
            'model_state_dict': student_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice,
        }, LAST_PATH)

    print(f"\n训练完成! Best Dice: {best_dice:.4f}")

    print("\n===== 最终效率评估 =====")
    measure_efficiency(teacher_model, "Teacher")
    measure_efficiency(student_model, "Student")


if __name__ == "__main__":
    main()