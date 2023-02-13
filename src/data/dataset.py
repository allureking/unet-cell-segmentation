"""
Dataset and DataLoader utilities for the cell microscopy dataset.

Handles loading .npz files from the Kaggle competition:
    "Counting Cells in Microscopy Images 2023"

Images are single-channel grayscale (128x128) with float32 values in [0, 255].
Masks are binary (0/1) indicating cell regions.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class CellDataset(Dataset):
    """
    PyTorch Dataset for microscopy cell images.

    Args:
        images: numpy array of shape (N, H, W), float32
        masks: numpy array of shape (N, H, W), float32 binary (optional)
        transform: optional callable for data augmentation
    """

    def __init__(self, images, masks=None, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Normalize to [0, 1] and add channel dimension
        image = self.images[idx].astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)

        if self.masks is not None:
            mask = self.masks[idx].astype(np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0)
        else:
            mask = torch.zeros(1, image.shape[1], image.shape[2])

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


def load_train_data(data_path, val_split=0.2, random_state=42):
    """
    Load training data and split into train/validation sets.

    Args:
        data_path: path to train_data.npz
        val_split: fraction of data for validation
        random_state: random seed for reproducibility

    Returns:
        (X_train, y_train, X_val, y_val) numpy arrays
    """
    data = np.load(data_path)
    images = data["images"]
    masks = data["masks"]

    X_train, X_val, y_train, y_val = train_test_split(
        images, masks, test_size=val_split, random_state=random_state
    )

    return X_train, y_train, X_val, y_val


def load_test_data(data_path):
    """Load test images (no masks)."""
    data = np.load(data_path)
    return data["images"]


def create_dataloaders(
    X_train, y_train, X_val, y_val, batch_size=16, num_workers=0, transform=None
):
    """Create train and validation DataLoaders."""
    train_dataset = CellDataset(X_train, y_train, transform=transform)
    val_dataset = CellDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
