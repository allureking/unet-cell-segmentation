"""
Evaluation script — load a trained model and compute metrics on validation set.

Usage:
    python -m src.evaluate --checkpoint checkpoints/best_model.pth
"""

import argparse
import os

import numpy as np
import torch
from scipy.ndimage import label

from src.model.unet import UNet
from src.data.dataset import load_train_data, CellDataset
from src.utils.metrics import dice_score, iou_score, pixel_accuracy, count_cells
from src.utils.visualization import (
    plot_prediction_overlay,
    plot_cell_count_distribution,
    plot_sample_grid,
)


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_features=config.get("base_features", 64),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"  Val dice: {checkpoint.get('val_dice', 'N/A')}")

    return model, config


@torch.no_grad()
def predict_batch(model, images, device):
    """Run inference on a batch of images."""
    images = images.to(device)
    logits = model(images)
    probs = torch.sigmoid(logits)
    return probs.cpu().numpy()


def evaluate(checkpoint_path, data_path=None, output_dir="results"):
    """Full evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(checkpoint_path, device)

    if data_path is None:
        data_path = config.get("data_path", "data/train_data.npz")

    # Load validation data
    _, _, X_val, y_val = load_train_data(data_path, val_split=config.get("val_split", 0.2))

    dataset = CellDataset(X_val, y_val)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    # Collect predictions
    all_preds = []
    all_targets = []
    dice_scores = []
    iou_scores = []

    for images, masks in loader:
        preds = predict_batch(model, images, device)
        all_preds.append(preds)
        all_targets.append(masks.numpy())

        for p, t in zip(preds, masks.numpy()):
            dice_scores.append(dice_score(p, t))
            iou_scores.append(iou_score(p, t))

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Print summary metrics
    print(f"\nEvaluation Results ({len(X_val)} samples)")
    print("-" * 40)
    print(f"  Mean Dice:     {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"  Mean IoU:      {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")

    # Cell counting accuracy
    true_counts = []
    pred_counts = []
    for i in range(len(all_preds)):
        tc = count_cells(all_targets[i][0])
        pc = count_cells(all_preds[i][0])
        true_counts.append(tc)
        pred_counts.append(pc)

    true_counts = np.array(true_counts)
    pred_counts = np.array(pred_counts)
    mae = np.mean(np.abs(true_counts - pred_counts))
    print(f"  Cell Count MAE: {mae:.2f}")
    print(f"  True count range:  [{true_counts.min()}, {true_counts.max()}]")
    print(f"  Pred count range:  [{pred_counts.min()}, {pred_counts.max()}]")

    # Save visualizations
    os.makedirs(output_dir, exist_ok=True)

    # Sample predictions
    fig = plot_sample_grid(
        all_preds[:4, 0], all_targets[:4, 0], all_preds[:4, 0], n_samples=4
    )
    fig.savefig(os.path.join(output_dir, "sample_predictions.png"), dpi=150, bbox_inches="tight")

    # Overlay for best and worst predictions
    best_idx = np.argmax(dice_scores)
    worst_idx = np.argmin(dice_scores)

    fig = plot_prediction_overlay(
        X_val[best_idx], y_val[best_idx], all_preds[best_idx][0]
    )
    fig.savefig(os.path.join(output_dir, "best_prediction.png"), dpi=150, bbox_inches="tight")

    fig = plot_prediction_overlay(
        X_val[worst_idx], y_val[worst_idx], all_preds[worst_idx][0]
    )
    fig.savefig(os.path.join(output_dir, "worst_prediction.png"), dpi=150, bbox_inches="tight")

    # Cell count distribution
    fig = plot_cell_count_distribution(true_counts, pred_counts)
    fig.savefig(os.path.join(output_dir, "count_distribution.png"), dpi=150, bbox_inches="tight")

    print(f"\nVisualizations saved to {output_dir}/")

    return {
        "dice_mean": np.mean(dice_scores),
        "dice_std": np.std(dice_scores),
        "iou_mean": np.mean(iou_scores),
        "iou_std": np.std(iou_scores),
        "count_mae": mae,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate U-Net model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.data, args.output)
