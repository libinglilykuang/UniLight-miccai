import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random
import time
from models.teacher import get_teacher_model, load_teacher_weights

from models.student import LATUPStudent
from data.dataset import get_datasets
from utils.loss_metrics import (
    WeightedDiceBCELoss, DWKDLoss, BoundaryLoss,
    calculate_dice, calculate_hd95_per_channel
)

# ============================================================
# 1. Core Configuration
# ============================================================
DATASET_TYPE = "msd"  # Choose "brats2023" or "msd"
CURRENT_FOLD = 4  # from [0,4]
NUM_FOLDS = 5
SEED = 42

# Path Configurations
if DATASET_TYPE == "brats2023":
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "brats2023", "Training")
    TEACHER_WEIGHTS = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "checkpoints",
        "BraTS2023",
        "best_teacher_2023.pth"
    )
    RESUME_LAST_PATH = None
    RESUME_BEST_PATH = None
else:
    DATA_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "msd",
        "Task01_BrainTumour"
    )
    TEACHER_WEIGHTS = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "checkpoints",
        "MSD-BTS",
        "best_teacher_msd.pth"
    )
    RESUME_LAST_PATH = None
    RESUME_BEST_PATH = None

SAVE_DIR = f"./kd_results_{DATASET_TYPE}_fold_{CURRENT_FOLD}"
SDF_DIR = os.path.join(SAVE_DIR, "sdfs")
os.makedirs(SDF_DIR, exist_ok=True)

BEST_PATH = os.path.join(SAVE_DIR, "best_student.pt")
LAST_PATH = os.path.join(SAVE_DIR, "last_student.pth")

TOTAL_EPOCHS = 1
BATCH_SIZE = 2
LR = 1e-4
ROI_SIZE = (128, 128, 128)
VAL_INTERVAL = 5
TEMPERATURE = 3.0

# Loss Weights
ALPHA_TASK = 0.6
BETA_DWKD = 0.2
USE_RCE = True
GAMMA_RCE = 0.1
RCE_T = 0.9
USE_BOUNDARY = True
BOUNDARY_WEIGHT = 0.1


# ============================================================
# 2. Randomness and Environment Initialization
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
# 3. Efficiency Benchmarking Functions
# ============================================================
def benchmark_model(model, model_name, input_size=(1, 4, 128, 128, 128), device='cuda'):
    try:
        from thop import profile
    except ImportError:
        profile = None

    device = torch.device(device)
    model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_size).to(device)

    # --- 1. Params & MACs ---
    params_m, macs_g, tflops = 0, 0, 0
    if profile is not None:
        try:
            # Wrapper to handle models that return tuples
            class Wrapper(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m

                def forward(self, x):
                    out = self.m(x)
                    return out[0] if isinstance(out, (list, tuple)) else out

            macs, params = profile(Wrapper(model), inputs=(dummy_input,), verbose=False)
            params_m = params / 1e6
            macs_g = macs / 1e9
            # TFLOPS = 2 * MACs / 10^12 (Assuming multiply-accumulate is 2 operations)
            tflops = (2 * macs) / 1e12
        except Exception as e:
            print(f"thop calculation failed: {e}")

    # --- 2. Latency ---
    # Warm-up
    for _ in range(10):
        with torch.no_grad(): _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(100):  # Average over 100 runs
            with torch.no_grad():
                _ = model(dummy_input)
        end_event.record()
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event) / 100
    else:
        # CPU Fallback timing
        t0 = time.time()
        for _ in range(50):
            with torch.no_grad(): _ = model(dummy_input)
        latency_ms = (time.time() - t0) * 1000 / 50

    # --- 3. Inference Memory ---
    im_mb = 0
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input)
        im_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # --- 4. Training Memory (Simulating backward pass) ---
    tm_mb = 0
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.train()  # Switch back to training mode

        # Check if the model has trainable parameters
        if any(p.requires_grad for p in model.parameters()):
            opt = torch.optim.Adam(model.parameters())
            opt.zero_grad()
            out = model(dummy_input)
            loss = (out[0].mean() if isinstance(out, (list, tuple)) else out.mean())
            loss.backward()
            tm_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            tm_mb = 0  # Frozen models have no training overhead

    model.eval()

    return {
        "Model": model_name,
        "Params (M)": f"{params_m:.2f}",
        "MACs (G)": f"{macs_g:.2f}",
        "TFLOPS": f"{tflops:.4f}",
        "Latency (ms)": f"{latency_ms:.2f}",
        "Inference Mem (MB)": f"{im_mb:.0f}",
        "Training Mem (MB)": f"{tm_mb:.0f}"
    }


def print_table(results):
    """Prints a clean ASCII table for benchmark comparison"""
    if not results: return
    headers = list(results[0].keys())
    # Calculate column widths
    widths = [max(len(h), max(len(str(r[h])) for r in results)) + 2 for h in headers]

    border = "+" + "+".join(["-" * w for w in widths]) + "+"

    print("\n" + border)
    header_row = "|" + "|".join([f" {h:<{w - 2}} " for h, w in zip(headers, widths)]) + "|"
    print(header_row)
    print(border)

    for res in results:
        row = "|" + "|".join([f" {str(res[h]):<{w - 2}} " for h, w in zip(headers, widths)]) + "|"
        print(row)
    print(border + "\n")


# ============================================================
# 4. Main Training Pipeline
# ============================================================
def main():
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

    # --- Model Initialization ---
    """teacher_model = UNet(
        spatial_dims=3, in_channels=4, out_channels=3,
        channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), num_res_units=2
    ).to(device)

    try:
        teacher_ckpt = torch.load(TEACHER_WEIGHTS, map_location='cpu')
        sd = teacher_ckpt['model_state_dict'] if 'model_state_dict' in teacher_ckpt else teacher_ckpt
        teacher_model.load_state_dict(sd, strict=False)
        print("Teacher model loaded successfully")
    except Exception as e:
        print(f"Teacher model failed to load: {e}")

    teacher_model.eval()"""
    teacher_model = get_teacher_model(device)
    teacher_model = load_teacher_weights(teacher_model, TEACHER_WEIGHTS, device)

    student_model = LATUPStudent(in_channels=4, out_channels=3, base_filters=16, deep_supervision=True).to(device)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LR, weight_decay=1e-5)

    # --- Loss Functions ---
    task_loss_fn = WeightedDiceBCELoss(weights=[1.0, 1.0, 3.0])
    dwkd_loss_fn = DWKDLoss(temperature=TEMPERATURE, use_rce=USE_RCE, rce_t=RCE_T)
    boundary_loss_fn = BoundaryLoss()

    # --- Resume Training Logic ---
    start_epoch = 1
    best_dice = 0.0

    # 1. Resume from Last Checkpoint
    if RESUME_LAST_PATH and os.path.exists(RESUME_LAST_PATH):
        ckpt = torch.load(RESUME_LAST_PATH, map_location=device)
        student_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_dice = ckpt.get('best_dice', 0.0)

    # 2. Check Best Checkpoint
    if RESUME_BEST_PATH and os.path.exists(RESUME_BEST_PATH):
        ckpt_best = torch.load(RESUME_BEST_PATH, map_location='cpu')
        actual_best = ckpt_best.get('best_dice', 0.0)
        if actual_best > best_dice:
            best_dice = actual_best
            print(f"Calibrated Best Dice to: {best_dice:.4f}")

            # If Last was not loaded, start fine-tuning from Best
            if not (RESUME_LAST_PATH and os.path.exists(RESUME_LAST_PATH)):
                print("No Last weight found, starting from Best weight...")
                student_model.load_state_dict(ckpt_best['model_state_dict'])

    print(f"Starting training: Epoch {start_epoch} - {TOTAL_EPOCHS}, Best Dice: {best_dice:.4f}")

    # --- Training Loop ---
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

            # Deep Supervision
            if isinstance(student_outputs, (list, tuple)):
                logits_main = student_outputs[0]
                logits_d2 = student_outputs[1]
                logits_d3 = student_outputs[2]

                # Main Task Loss
                L_task = task_loss_fn(logits_main, labels)
                L_dwkdl, L_rce = dwkd_loss_fn(logits_main, teacher_outputs)

                L_boundary = torch.tensor(0.0, device=device)
                if USE_BOUNDARY:
                    L_boundary = boundary_loss_fn(logits_main, labels, precomputed_sdfs=sdfs)

                loss_main = ALPHA_TASK * L_task + \
                            BETA_DWKD * L_dwkdl + \
                            (GAMMA_RCE * L_rce if USE_RCE else 0.0) + \
                            (BOUNDARY_WEIGHT * L_boundary if USE_BOUNDARY else 0.0)

                # Auxiliary Losses for DS
                labels_d2 = F.interpolate(labels, scale_factor=0.5, mode='nearest')
                loss_d2 = ALPHA_TASK * task_loss_fn(logits_d2, labels_d2)

                labels_d3 = F.interpolate(labels, scale_factor=0.25, mode='nearest')
                loss_d3 = ALPHA_TASK * task_loss_fn(logits_d3, labels_d3)

                loss = loss_main + 0.5 * loss_d2 + 0.25 * loss_d3

                probs = torch.sigmoid(logits_main)

            else:
                # No Deep Supervision
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

        # --- Validation ---
        if epoch % VAL_INTERVAL == 0 or epoch == 1:
            student_model.eval()
            val_dices, val_hd95s = [], []
            with torch.no_grad():
                for batch in val_loader:
                    images, labels, _, _ = batch
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                    outputs = student_model(images)

                    # Extract main branch for validation if using Deep Supervision
                    if isinstance(outputs, (list, tuple)):
                        outputs = outputs[0]

                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()

                    val_dices.append(calculate_dice(preds, labels))
                    val_hd95s.append(calculate_hd95_per_channel(preds, labels))

            # Metric Calculations
            val_dices_array = np.array(val_dices) if len(val_dices) > 0 else np.zeros((1, 3))
            mean_val_dice = float(np.mean(val_dices_array))

            val_hd95_array = np.array(val_hd95s) if len(val_hd95s) > 0 else np.full((1, 3), np.inf)
            mean_val_hd95 = float(np.nanmean(val_hd95_array))

            print(f'Epoch {epoch}:')
            print(f'  Train Loss={epoch_loss:.4f}, Dice={mean_train_dice:.4f}')
            print(
                f'  Val Dice={mean_val_dice:.4f} (WT={val_dices_array[:, 0].mean():.4f}, TC={val_dices_array[:, 1].mean():.4f}, ET={val_dices_array[:, 2].mean():.4f})')
            print(f'  Val HD95={mean_val_hd95:.2f}')

            # Save Best Model
            if mean_val_dice > best_dice:
                best_dice = mean_val_dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': student_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                }, BEST_PATH)
                print(f'Saving Best Model: {best_dice:.4f}')
            else:
                print(f'(Best was {best_dice:.4f})')

        # Save Last Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': student_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice,
        }, LAST_PATH)

    print(f"\nTraining Complete! Best Dice: {best_dice:.4f}")

    dummy_shape = (1, 4, *ROI_SIZE)
    print("\n===== Comprehensive Efficiency Benchmark =====")

    # Benchmark Teacher
    res_t = benchmark_model(teacher_model, "Teacher (UNet)", dummy_shape, device)

    # Benchmark Student
    # Note: DS is disabled here to measure actual inference latency in production
    student_model_eval = LATUPStudent(in_channels=4, out_channels=3, base_filters=16, deep_supervision=False).to(device)
    student_model_eval.load_state_dict(student_model.state_dict(), strict=False)
    res_s = benchmark_model(student_model_eval, "Student (LATUP)", dummy_shape, device)

    # Print final comparison table
    print_table([res_t, res_s])


if __name__ == "__main__":
    main()
