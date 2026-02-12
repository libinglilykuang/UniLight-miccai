import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


# ============================================================
# 1. TASK LOSS (Dice + BCE)
# ============================================================
class WeightedDiceBCELoss(nn.Module):
    """
    Combined Dice and Binary Cross Entropy loss with channel-wise weighting.
    Typically used for BraTS (WT, TC, ET) where classes are imbalanced.
    """

    def __init__(self, weights=[1.0, 1.0, 3.0]):
        super().__init__()
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        device = logits.device
        w = torch.tensor(self.weights).to(device).view(1, 3, 1, 1, 1)

        # Weighted BCE
        bce_loss = self.bce(logits, targets)
        bce_loss = (bce_loss * w).mean()

        # Weighted Dice
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3, 4))
        union = probs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4))
        dice_per_channel = 1 - (2. * intersection + 1e-6) / (union + 1e-6)
        weighted_dice = (dice_per_channel * torch.tensor(self.weights).to(device)).mean()

        return bce_loss + weighted_dice


# ============================================================
# 2. BOUNDARY LOSS (SDF-Based)
# ============================================================
class BoundaryLoss(nn.Module):
    """
    Boundary loss using Signed Distance Function (SDF).
    Penalizes pixels based on their distance from the true boundary.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_sdf_np(gt_binary):
        """Compute SDF map from binary ground truth"""
        if gt_binary.sum() == 0:
            return np.ones_like(gt_binary, dtype=np.float32)
        posmask = gt_binary.astype(bool)
        negmask = ~posmask
        dt_out = distance_transform_edt(negmask)
        dt_in = distance_transform_edt(posmask)
        sdf = dt_out.astype(np.float32)
        sdf[posmask] = -dt_in[posmask].astype(np.float32)
        # Normalize SDF
        maxv = np.max(np.abs(sdf)) if np.max(np.abs(sdf)) > 0 else 1.0
        return sdf / (maxv + 1e-8)

    def forward(self, logits, targets, precomputed_sdfs=None):
        probs = torch.sigmoid(logits)

        if precomputed_sdfs is not None:
            # Boundary Loss = Mean( Prob * SDF )
            val = (probs * precomputed_sdfs).mean()
            return val

        return torch.tensor(0.0, device=logits.device)


# ============================================================
# 3. KNOWLEDGE DISTILLATION LOSS (DWKD + RCE)
# ============================================================
class DWKDLoss(nn.Module):
    """
    Dynamic Weighted Knowledge Distillation (DWKD) and Reverse Cross Entropy (RCE).
    """

    def __init__(self, temperature=3.0, use_rce=True, rce_t=0.9):
        super().__init__()
        self.tau = temperature
        self.use_rce = use_rce
        self.rce_t = rce_t

    @staticmethod
    def _kl_bernoulli(p, q, eps=1e-6):
        p = p.clamp(eps, 1 - eps)
        q = q.clamp(eps, 1 - eps)
        return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))

    def forward(self, student_logits, teacher_logits):
        p_t = torch.sigmoid(teacher_logits / self.tau)
        p_s = torch.sigmoid(student_logits / self.tau)

        # Dynamic weight calculation (Alpha) based on prediction entropy
        eps = 1e-6
        h = -(p_t * torch.log(p_s.clamp(eps, 1 - eps)) +
              (1 - p_t) * torch.log((1 - p_s).clamp(eps, 1 - eps)))
        h_max = h.amax(dim=(1, 2, 3, 4), keepdim=True).clamp(min=eps)
        alpha = (h / h_max).detach()

        # DWKD using KL Divergence
        kl_pix = self._kl_bernoulli(p_t, p_s)
        l_dwkd = (alpha * kl_pix).mean()

        # Reverse Cross Entropy (RCE) for background/noise suppression
        l_rce = torch.tensor(0.0, device=student_logits.device)
        if self.use_rce:
            q_s = 1.0 - torch.sigmoid(student_logits)
            with torch.no_grad():
                q_t_raw = torch.sigmoid(teacher_logits)
                mask_confident = (q_t_raw >= self.rce_t)
            q_s = q_s.clamp(eps, 1 - eps)
            # Penalize student where teacher is not confident
            l_rce = -((1 - mask_confident.float()) * torch.log(q_s)).mean()

        return l_dwkd, l_rce


# ============================================================
# 4. EVALUATION METRICS (Dice & HD95)
# ============================================================
def calculate_dice(preds, targets):
    scores = []
    for i in range(3):
        pred_ch = preds[:, i]
        target_ch = targets[:, i]
        intersection = float((pred_ch * target_ch).sum().detach().cpu())
        pred_sum    = float(pred_ch.sum().detach().cpu())
        target_sum  = float(target_ch.sum().detach().cpu())
        union = pred_sum + target_sum
        if union == 0.0:
            dice = 1.0 if intersection == 0.0 else 0.0
        else:
            dice = (2.0 * intersection) / (union + 1e-6)
        scores.append(float(dice))
    return scores

def _surface(mask_bool):
    if mask_bool.sum() == 0:
        return np.zeros_like(mask_bool, dtype=bool)
    eroded = binary_erosion(mask_bool, iterations=1, border_value=0)
    return mask_bool ^ eroded

def _surface_distances(surface_a, surface_b, spacing=(1.0, 1.0, 1.0)):
    if surface_b.sum() == 0:
        return np.array([], dtype=np.float32)
    dt = distance_transform_edt(~surface_b, sampling=spacing)
    coords = np.argwhere(surface_a)
    if coords.size == 0:
        return np.array([], dtype=np.float32)
    return dt[tuple(coords.T)].astype(np.float32)

def hd95_binary(pred_bin, gt_bin, spacing=(1.0, 1.0, 1.0)):
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 0.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float('inf')
    s_p = _surface(pred_bin);
    s_g = _surface(gt_bin)
    d_pg = _surface_distances(s_p, s_g, spacing)
    d_gp = _surface_distances(s_g, s_p, spacing)
    if d_pg.size == 0 and d_gp.size == 0:
        return float('inf')
    merged = np.concatenate([d_pg, d_gp]) if d_pg.size and d_gp.size else (d_pg if d_pg.size else d_gp)
    return float(np.percentile(merged, 95))


def calculate_hd95_per_channel(preds, targets, roi_size=(128, 128, 128), spacing=(1.0, 1.0, 1.0)):
    # Convert tensors to numpy and threshold at 0.5
    preds_np = (preds.detach().cpu().numpy() > 0.5).astype(np.uint8)
    targets_np = (targets.detach().cpu().numpy() > 0.5).astype(np.uint8)
    hd_list = []

    # Calculate the diagonal of the ROI as the maximum possible penalty
    # This replaces the unresolved ROI_SIZE global variable
    max_penalty = np.sqrt(roi_size[0] ** 2 + roi_size[1] ** 2 + roi_size[2] ** 2)

    # Iterate through channels (e.g., WT, TC, ET)
    for c in range(preds_np.shape[1]):
        vals = []
        # Iterate through batch
        for b in range(preds_np.shape[0]):
            # Call binary HD95 function
            hd = hd95_binary(preds_np[b, c], targets_np[b, c], spacing)

            # Handle infinity (occurs when one or both masks are empty)
            if np.isinf(hd):
                pred_sum = preds_np[b, c].sum()
                target_sum = targets_np[b, c].sum()
                if pred_sum == 0 and target_sum == 0:
                    hd = 0.0  # Perfect match for two empty masks
                else:
                    # One mask is empty, apply maximum penalty
                    hd = max_penalty
            vals.append(hd)

        vals = np.array(vals, dtype=np.float32)
        # Append mean HD95 for current channel, ignoring NaNs if any
        hd_list.append(float(np.nanmean(vals)) if len(vals) > 0 else 0.0)

    return hd_list