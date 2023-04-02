"""
Generate Kaggle submission by predicting cell counts on test images.

Usage:
    python -m src.predict --checkpoint checkpoints/best_model.pth --test data/test_images.npz
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from src.model.unet import UNet
from src.data.dataset import CellDataset
from src.utils.metrics import count_cells


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_features=config.get("base_features", 64),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_counts(model, test_images, device, batch_size=32, threshold=0.5, min_area=10):
    """
    Predict cell counts for test images.

    Pipeline:
        1. Run U-Net to get segmentation masks
        2. Binarize masks at threshold
        3. Count connected components per image

    Args:
        model: trained U-Net model
        test_images: numpy array (N, H, W)
        device: torch device
        batch_size: inference batch size
        threshold: mask binarization threshold
        min_area: minimum pixel area for a valid cell

    Returns:
        List of cell counts
    """
    dataset = CellDataset(test_images)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    all_counts = []
    for images, _ in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()

        for prob in probs:
            count = count_cells(prob[0], threshold=threshold, min_area=min_area)
            all_counts.append(count)

    return all_counts


def generate_submission(checkpoint_path, test_path, output_path="submission.csv"):
    """Generate Kaggle submission CSV."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    print(f"Loading test images from {test_path}")
    data = np.load(test_path)
    test_images = data["images"]
    print(f"  {test_images.shape[0]} test images, shape {test_images.shape[1:]}")

    print("Predicting cell counts...")
    counts = predict_counts(model, test_images, device)

    print(f"  Count range: [{min(counts)}, {max(counts)}]")
    print(f"  Mean count: {np.mean(counts):.1f}")

    # Create submission
    submission = pd.DataFrame({
        "id": range(len(counts)),
        "count": counts,
    })
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Kaggle submission")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--test", type=str, default="data/test_images.npz")
    parser.add_argument("--output", type=str, default="submission.csv")
    args = parser.parse_args()

    generate_submission(args.checkpoint, args.test, args.output)
