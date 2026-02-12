import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
import time

# Import existing modules
from models.teacher import get_teacher_model
from data.dataset import get_datasets
from utils.loss_metrics import WeightedDiceBCELoss, calculate_dice

# ============================================================
# 1. Configuration
# ============================================================
DATASET_TYPE = "brats2023"  # Choose "brats2023" or "msd"
CURRENT_FOLD = 4
NUM_FOLDS = 5
SEED = 42

# Path Configuration
if DATASET_TYPE == "brats2023":
    DATA_DIR = "/kaggle/input/datasets/ahmedelzayat77/bratsglioma/Training"
else:
    DATA_DIR = "/kaggle/input/msd-brain-tumour/Task01_BrainTumour"

# Output directory for Teacher model
SAVE_DIR = f"./teacher_results_{DATASET_TYPE}_fold_{CURRENT_FOLD}"
os.makedirs(SAVE_DIR, exist_ok=True)

BEST_PATH = os.path.join(SAVE_DIR, "best_teacher.pth")
LAST_PATH = os.path.join(SAVE_DIR, "last_teacher.pth")

# Hyperparameters
TOTAL_EPOCHS = 100
BATCH_SIZE = 2
LR = 1e-4
ROI_SIZE = (128, 128, 128)
VAL_INTERVAL = 1


# ============================================================
# 2. Setup
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


def main():
    # --- 1. Data Loading ---
    # sdf_dir is set to None because the Teacher model does not require Boundary Loss/SDFs
    train_ds, val_ds = get_datasets(
        data_dir=DATA_DIR,
        fold=CURRENT_FOLD,
        num_folds=NUM_FOLDS,
        seed=SEED,
        sdf_dir=None,
        roi_size=ROI_SIZE
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    # --- 2. Model Initialization ---
    # Initializes the standard MONAI U-Net defined in teacher.py
    model = get_teacher_model(device)

    # Calculate and display parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"Teacher Initialized. Total Params: {params / 1e6:.2f} M")

    # --- 3. Optimizer & Loss ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    # Standard cosine annealing scheduler for segmentation tasks
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)

    # Teacher uses standard Supervised Loss (Dice + BCE)
    loss_function = WeightedDiceBCELoss(weights=[1.0, 1.0, 3.0])

    # --- 4. Training Loop ---
    best_dice = 0.0
    start_epoch = 1

    # Resume training logic
    if os.path.exists(LAST_PATH):
        print(f"Resuming training from checkpoint: {LAST_PATH}")
        ckpt = torch.load(LAST_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_dice = ckpt.get('best_dice', 0.0)

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        train_dices = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}")

        for batch in pbar:
            # Dataset returns: [image, label, sdf, case_name]
            # Note: Teacher model ignores SDF and case_name
            images, labels, _, _ = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate Loss (Supervised only)
            loss = loss_function(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate Training Dice for real-time monitoring
            with torch.no_grad():
                preds = (torch.sigmoid(outputs) > 0.5).float()
                train_dices.append(calculate_dice(preds, labels))

            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Dice': f"{np.mean(train_dices[-1]):.3f}"})

        # Update learning rate
        scheduler.step()
        epoch_loss /= len(train_loader)

        # --- Validation Loop ---
        if epoch % VAL_INTERVAL == 0:
            model.eval()
            val_dices = []

            with torch.no_grad():
                for batch in val_loader:
                    images, labels, _, _ = batch
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    preds = (torch.sigmoid(outputs) > 0.5).float()

                    val_dices.append(calculate_dice(preds, labels))

            mean_val_dice = np.mean(val_dices)
            val_dices_arr = np.array(val_dices)

            # Performance Logging by Channel
            wt_dice = val_dices_arr[:, 0].mean()
            tc_dice = val_dices_arr[:, 1].mean()
            et_dice = val_dices_arr[:, 2].mean()

            print(f"\nEvaluation for Epoch {epoch}:")
            print(f"  Average Train Loss: {epoch_loss:.4f}")
            print(f"  Average Val Dice : {mean_val_dice:.4f} (WT={wt_dice:.3f}, TC={tc_dice:.3f}, ET={et_dice:.3f})")

            # Update Best Model
            if mean_val_dice > best_dice:
                best_dice = mean_val_dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_dice': best_dice,
                }, BEST_PATH)
                print(f"New Best Teacher Saved! Score: ({best_dice:.4f})")

        # Save Last Checkpoint (for potential resumption)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice
        }, LAST_PATH)

    print(f"\nTeacher Training Completed! Highest Val Dice achieved: {best_dice:.4f}")


if __name__ == "__main__":
    main()