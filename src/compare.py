"""
Compare U-Net vs Attention U-Net performance.

Trains both models with identical hyperparameters and reports
segmentation quality and cell counting accuracy side by side.

Usage:
    python -m src.compare [--config configs/default.yaml]
"""

import argparse
import os
import time

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.model.unet import UNet
from src.model.attention_unet import AttentionUNet
from src.model.losses import BCEDiceLoss
from src.data.dataset import load_train_data, create_dataloaders
from src.data.augmentation import get_train_transforms
from src.utils.metrics import dice_score, iou_score, count_cells, MetricTracker
from src.train import train_one_epoch, validate, get_device, load_config


def compare_models(config):
    """Train and compare U-Net vs Attention U-Net."""
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = get_device()
    print(f"Device: {device}\n")

    # Load data
    X_train, y_train, X_val, y_val = load_train_data(
        config["data_path"], val_split=config["val_split"]
    )

    augmentation = get_train_transforms() if config["augmentation"] else None
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=config["batch_size"],
        transform=augmentation,
    )

    models = {
        "U-Net": UNet(in_channels=1, out_channels=1, base_features=config["base_features"]),
        "Attention U-Net": AttentionUNet(in_channels=1, out_channels=1, base_features=config["base_features"]),
    }

    results = {}

    for name, model in models.items():
        print(f"{'='*60}")
        print(f"Training {name} ({model.count_parameters():,} parameters)")
        print(f"{'='*60}")

        model = model.to(device)
        criterion = BCEDiceLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        history = {"train_loss": [], "val_loss": [], "val_dice": []}
        best_val_loss = float("inf")

        for epoch in range(1, config["epochs"] + 1):
            train_tracker = MetricTracker()
            train_one_epoch(model, train_loader, criterion, optimizer, device, train_tracker)

            val_tracker = MetricTracker()
            validate(model, val_loader, criterion, device, val_tracker)

            train_metrics = train_tracker.summary()
            val_metrics = val_tracker.summary()
            scheduler.step(val_metrics["loss"])

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_dice"].append(val_metrics["dice"])

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d} | "
                    f"Train: {train_metrics['loss']:.4f} | "
                    f"Val: {val_metrics['loss']:.4f} | "
                    f"Dice: {val_metrics['dice']:.4f}"
                )

        results[name] = {
            "history": history,
            "best_val_loss": best_val_loss,
            "final_dice": history["val_dice"][-1],
            "params": model.count_parameters(),
        }

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, res in results.items():
        epochs = range(1, len(res["history"]["val_loss"]) + 1)
        axes[0].plot(epochs, res["history"]["val_loss"], label=f"{name}", linewidth=2)
        axes[1].plot(epochs, res["history"]["val_dice"], label=f"{name}", linewidth=2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("Validation Loss Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].set_title("Dice Score Comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    fig.savefig("results/model_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to results/model_comparison.png")

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Model':<20} {'Params':>10} {'Best Loss':>12} {'Final Dice':>12}")
    print(f"{'-'*60}")
    for name, res in results.items():
        print(
            f"{name:<20} {res['params']:>10,} "
            f"{res['best_val_loss']:>12.4f} {res['final_dice']:>12.4f}"
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare U-Net architectures")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    compare_models(config)
