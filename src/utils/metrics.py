"""
Evaluation metrics for binary segmentation.
"""

import numpy as np
import torch


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    Intersection over Union (Jaccard Index).

    Args:
        pred: predicted logits or probabilities
        target: ground truth binary mask
        threshold: binarization threshold
        smooth: smoothing to avoid division by zero

    Returns:
        IoU score (float)
    """
    if isinstance(pred, torch.Tensor):
        pred = torch.sigmoid(pred)
        pred = (pred > threshold).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return ((intersection + smooth) / (union + smooth)).item()
    else:
        pred = (pred > threshold).astype(float)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return (intersection + smooth) / (union + smooth)


def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    Dice Coefficient (F1 Score for segmentation).

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        pred: predicted logits or probabilities
        target: ground truth binary mask
        threshold: binarization threshold
        smooth: smoothing to avoid division by zero

    Returns:
        Dice score (float)
    """
    if isinstance(pred, torch.Tensor):
        pred = torch.sigmoid(pred)
        pred = (pred > threshold).float()
        intersection = (pred * target).sum()
        total = pred.sum() + target.sum()
        return ((2.0 * intersection + smooth) / (total + smooth)).item()
    else:
        pred = (pred > threshold).astype(float)
        intersection = (pred * target).sum()
        total = pred.sum() + target.sum()
        return (2.0 * intersection + smooth) / (total + smooth)


def pixel_accuracy(pred, target, threshold=0.5):
    """
    Per-pixel classification accuracy.

    Args:
        pred: predicted logits or probabilities
        target: ground truth binary mask
        threshold: binarization threshold

    Returns:
        Accuracy (float)
    """
    if isinstance(pred, torch.Tensor):
        pred = torch.sigmoid(pred)
        pred = (pred > threshold).float()
        correct = (pred == target).sum()
        total = target.numel()
        return (correct / total).item()
    else:
        pred = (pred > threshold).astype(float)
        return (pred == target).mean()


def precision_recall(pred, target, threshold=0.5, smooth=1e-6):
    """
    Compute precision and recall for binary segmentation.

    Returns:
        (precision, recall) tuple
    """
    if isinstance(pred, torch.Tensor):
        pred = torch.sigmoid(pred)
        pred = (pred > threshold).float()
        tp = (pred * target).sum()
        precision = (tp + smooth) / (pred.sum() + smooth)
        recall = (tp + smooth) / (target.sum() + smooth)
        return precision.item(), recall.item()
    else:
        pred = (pred > threshold).astype(float)
        tp = (pred * target).sum()
        precision = (tp + smooth) / (pred.sum() + smooth)
        recall = (tp + smooth) / (target.sum() + smooth)
        return float(precision), float(recall)


def count_cells(mask, threshold=0.5, min_area=10):
    """
    Count cells in a binary segmentation mask using connected component labeling.

    Args:
        mask: 2D numpy array (predicted mask)
        threshold: binarization threshold
        min_area: minimum pixel area for a valid cell

    Returns:
        Number of detected cells
    """
    from scipy.ndimage import label

    binary = (mask > threshold).astype(np.uint8)
    labeled, num_features = label(binary)

    if min_area <= 1:
        return num_features

    # Filter out tiny components (noise)
    count = 0
    for i in range(1, num_features + 1):
        if (labeled == i).sum() >= min_area:
            count += 1

    return count


class MetricTracker:
    """Track and aggregate metrics over batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}

    def update(self, name, value, n=1):
        if name not in self.metrics:
            self.metrics[name] = {"sum": 0, "count": 0}
        self.metrics[name]["sum"] += value * n
        self.metrics[name]["count"] += n

    def avg(self, name):
        m = self.metrics[name]
        return m["sum"] / m["count"] if m["count"] > 0 else 0

    def summary(self):
        return {name: self.avg(name) for name in self.metrics}
