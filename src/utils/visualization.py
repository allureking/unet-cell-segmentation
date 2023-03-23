"""
Visualization utilities for microscopy cell segmentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import label


def plot_sample_grid(images, masks=None, predictions=None, n_samples=4, figsize=(16, 4)):
    """
    Plot a grid of images with optional masks and predictions.

    Args:
        images: array of shape (N, H, W) or (N, 1, H, W)
        masks: ground truth masks, same shape
        predictions: predicted masks, same shape
        n_samples: number of samples to display
        figsize: figure size
    """
    n_cols = 1
    if masks is not None:
        n_cols += 1
    if predictions is not None:
        n_cols += 1

    fig, axes = plt.subplots(n_samples, n_cols, figsize=figsize)
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(min(n_samples, len(images))):
        img = _squeeze_channel(images[i])
        col = 0

        axes[i, col].imshow(img, cmap="gray")
        axes[i, col].set_title("Input" if i == 0 else "")
        axes[i, col].axis("off")
        col += 1

        if masks is not None:
            msk = _squeeze_channel(masks[i])
            axes[i, col].imshow(msk, cmap="viridis")
            axes[i, col].set_title("Ground Truth" if i == 0 else "")
            axes[i, col].axis("off")
            col += 1

        if predictions is not None:
            pred = _squeeze_channel(predictions[i])
            axes[i, col].imshow(pred, cmap="viridis")
            axes[i, col].set_title("Prediction" if i == 0 else "")
            axes[i, col].axis("off")

    plt.tight_layout()
    return fig


def plot_prediction_overlay(image, mask, prediction, threshold=0.5, figsize=(15, 5)):
    """
    Plot original image, ground truth overlay, and prediction overlay side by side.

    Args:
        image: input image (H, W) or (1, H, W)
        mask: ground truth mask
        prediction: predicted mask (probabilities or logits)
        threshold: binarization threshold
        figsize: figure size
    """
    image = _squeeze_channel(image)
    mask = _squeeze_channel(mask)
    prediction = _squeeze_channel(prediction)

    # Binarize prediction
    pred_binary = (prediction > threshold).astype(float)

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Original image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Ground truth overlay
    axes[1].imshow(image, cmap="gray", alpha=0.7)
    axes[1].imshow(mask, cmap="Reds", alpha=0.4)
    axes[1].set_title(f"Ground Truth ({int(count_components(mask))} cells)")
    axes[1].axis("off")

    # Prediction overlay
    axes[2].imshow(image, cmap="gray", alpha=0.7)
    axes[2].imshow(pred_binary, cmap="Greens", alpha=0.4)
    axes[2].set_title(f"Prediction ({int(count_components(pred_binary))} cells)")
    axes[2].axis("off")

    # Difference map
    diff = np.abs(mask - pred_binary)
    axes[3].imshow(image, cmap="gray", alpha=0.5)
    axes[3].imshow(diff, cmap="hot", alpha=0.5)
    axes[3].set_title("Error Map")
    axes[3].axis("off")

    plt.tight_layout()
    return fig


def plot_training_history(train_losses, val_losses, train_metrics=None, val_metrics=None, figsize=(12, 4)):
    """
    Plot training and validation loss curves, optionally with metric curves.
    """
    n_plots = 1
    if train_metrics is not None:
        n_plots += 1

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Loss curves
    axes[0].plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Metric curves
    if train_metrics is not None and val_metrics is not None:
        axes[1].plot(epochs, train_metrics, "b-", label="Train Dice", linewidth=2)
        axes[1].plot(epochs, val_metrics, "r-", label="Val Dice", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Dice Score")
        axes[1].set_title("Dice Coefficient")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_cell_count_distribution(true_counts, pred_counts, figsize=(10, 4)):
    """
    Compare distribution of true vs predicted cell counts.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram comparison
    bins = np.arange(0, max(max(true_counts), max(pred_counts)) + 2) - 0.5
    axes[0].hist(true_counts, bins=bins, alpha=0.6, label="True", color="steelblue")
    axes[0].hist(pred_counts, bins=bins, alpha=0.6, label="Predicted", color="coral")
    axes[0].set_xlabel("Cell Count")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Cell Count Distribution")
    axes[0].legend()

    # Scatter plot: true vs predicted
    axes[1].scatter(true_counts, pred_counts, alpha=0.3, s=10, color="steelblue")
    max_count = max(max(true_counts), max(pred_counts))
    axes[1].plot([0, max_count], [0, max_count], "r--", linewidth=1, label="Perfect")
    axes[1].set_xlabel("True Count")
    axes[1].set_ylabel("Predicted Count")
    axes[1].set_title("True vs Predicted Cell Counts")
    axes[1].legend()
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def count_components(mask, min_area=1):
    """Count connected components in a binary mask."""
    labeled, n = label(mask > 0.5)
    if min_area <= 1:
        return n
    return sum(1 for i in range(1, n + 1) if (labeled == i).sum() >= min_area)


def _squeeze_channel(arr):
    """Remove channel dimension if present."""
    if isinstance(arr, np.ndarray):
        if arr.ndim == 3 and arr.shape[0] == 1:
            return arr[0]
    return arr
