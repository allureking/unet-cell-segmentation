"""
Training pipeline for U-Net cell segmentation.

Usage:
    python -m src.train [--config configs/default.yaml]
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import numpy as np
import yaml

from src.model.unet import UNet
from src.model.losses import BCEDiceLoss, DiceLoss, FocalLoss, TverskyLoss
from src.data.dataset import load_train_data, create_dataloaders
from src.data.augmentation import get_train_transforms
from src.utils.metrics import dice_score, iou_score, MetricTracker
from src.utils.visualization import plot_training_history, plot_sample_grid


def get_device():
    """Select best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_loss_fn(name="bce_dice"):
    """Get loss function by name."""
    losses = {
        "bce": nn.BCEWithLogitsLoss(),
        "dice": DiceLoss(),
        "bce_dice": BCEDiceLoss(alpha=0.5),
        "focal": FocalLoss(),
        "tversky": TverskyLoss(alpha=0.3, beta=0.7),
    }
    if name not in losses:
        raise ValueError(f"Unknown loss: {name}. Choose from {list(losses.keys())}")
    return losses[name]


def train_one_epoch(model, loader, criterion, optimizer, device, tracker):
    """Train for one epoch."""
    model.train()
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        tracker.update("loss", loss.item(), images.size(0))
        tracker.update("dice", dice_score(outputs, masks), images.size(0))
        tracker.update("iou", iou_score(outputs, masks), images.size(0))


@torch.no_grad()
def validate(model, loader, criterion, device, tracker):
    """Validate the model."""
    model.eval()
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        tracker.update("loss", loss.item(), images.size(0))
        tracker.update("dice", dice_score(outputs, masks), images.size(0))
        tracker.update("iou", iou_score(outputs, masks), images.size(0))


def train(config):
    """Main training loop."""
    # Set seeds for reproducibility
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    device = get_device()
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_train_data(
        config["data_path"], val_split=config["val_split"]
    )
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")

    # Create dataloaders
    augmentation = get_train_transforms() if config["augmentation"] else None
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 0),
        transform=augmentation,
    )

    # Model
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_features=config["base_features"],
    ).to(device)

    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,}")

    # Loss and optimizer
    criterion = get_loss_fn(config["loss"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0),
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Training history
    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
    }

    # Checkpoint directory
    ckpt_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0

    print(f"\nTraining for {config['epochs']} epochs...")
    print("-" * 60)

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()

        # Train
        train_tracker = MetricTracker()
        train_one_epoch(model, train_loader, criterion, optimizer, device, train_tracker)

        # Validate
        val_tracker = MetricTracker()
        validate(model, val_loader, criterion, device, val_tracker)

        # Get metrics
        train_metrics = train_tracker.summary()
        val_metrics = val_tracker.summary()
        elapsed = time.time() - t0

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_dice"].append(train_metrics["dice"])
        history["val_dice"].append(val_metrics["dice"])

        # Step scheduler
        scheduler.step(val_metrics["loss"])

        # Print progress
        print(
            f"Epoch {epoch:3d}/{config['epochs']} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.1f}s"
        )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_dice": val_metrics["dice"],
                "config": config,
            }, os.path.join(ckpt_dir, "best_model.pth"))

    print("-" * 60)
    print(f"Best model at epoch {best_epoch} with val loss: {best_val_loss:.4f}")

    # Save final model
    torch.save({
        "epoch": config["epochs"],
        "model_state_dict": model.state_dict(),
        "config": config,
    }, os.path.join(ckpt_dir, "final_model.pth"))

    # Save training plots
    os.makedirs("results", exist_ok=True)
    fig = plot_training_history(
        history["train_loss"], history["val_loss"],
        history["train_dice"], history["val_dice"],
    )
    fig.savefig("results/training_history.png", dpi=150, bbox_inches="tight")
    print("Saved training history plot to results/training_history.png")

    return model, history


def load_config(path=None):
    """Load config from YAML file with defaults."""
    defaults = {
        "data_path": "data/train_data.npz",
        "val_split": 0.2,
        "batch_size": 16,
        "epochs": 50,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "base_features": 64,
        "loss": "bce_dice",
        "augmentation": True,
        "seed": 42,
        "checkpoint_dir": "checkpoints",
        "num_workers": 0,
    }

    if path and os.path.exists(path):
        with open(path) as f:
            user_config = yaml.safe_load(f)
        if user_config:
            defaults.update(user_config)

    return defaults


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for cell segmentation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)
