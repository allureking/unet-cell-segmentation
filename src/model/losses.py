"""
Loss functions for binary segmentation.

Includes standard BCE loss and Dice loss, plus their combination.
Dice loss is particularly useful for imbalanced segmentation tasks
where the foreground (cells) occupies a small portion of the image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Operates on logits (applies sigmoid internally).
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice Loss.

    L = α * BCE + (1 - α) * Dice

    BCE provides pixel-wise gradients while Dice optimizes
    the global overlap metric directly.

    Args:
        alpha: weight for BCE component (default 0.5)
        smooth: smoothing factor for Dice (default 1.0)
    """

    def __init__(self, alpha=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Down-weights easy examples and focuses on hard negatives.

    Args:
        alpha: balancing factor (default 0.25)
        gamma: focusing parameter (default 2.0)
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss — generalization of Dice that allows asymmetric
    penalization of false positives vs false negatives.

    T = TP / (TP + α*FP + β*FN)

    Setting α=β=0.5 gives standard Dice loss.
    Higher β penalizes false negatives more (miss fewer cells).

    Args:
        alpha: FP penalty weight (default 0.3)
        beta: FN penalty weight (default 0.7)
        smooth: smoothing factor
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1 - targets_flat)).sum()
        fn = ((1 - probs_flat) * targets_flat).sum()

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky
