"""
Data augmentation transforms for microscopy images.

All transforms operate on (image, mask) pairs to ensure
consistent spatial transformations are applied to both.
"""

import torch
import torch.nn.functional as F
import numpy as np
import random


class Compose:
    """Chain multiple transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class RandomHorizontalFlip:
    """Randomly flip horizontally with probability p."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        return image, mask


class RandomVerticalFlip:
    """Randomly flip vertically with probability p."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        return image, mask


class RandomRotation90:
    """Randomly rotate by 0, 90, 180, or 270 degrees."""

    def __call__(self, image, mask):
        k = random.randint(0, 3)
        if k > 0:
            image = torch.rot90(image, k, dims=[-2, -1])
            mask = torch.rot90(mask, k, dims=[-2, -1])
        return image, mask


class RandomGaussianNoise:
    """Add random Gaussian noise to the image only (not mask)."""

    def __init__(self, mean=0.0, std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        noise = torch.randn_like(image) * self.std + self.mean
        image = torch.clamp(image + noise, 0.0, 1.0)
        return image, mask


class RandomBrightnessContrast:
    """Randomly adjust brightness and contrast."""

    def __init__(self, brightness_range=0.1, contrast_range=0.1):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, image, mask):
        # Brightness
        brightness = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
        image = torch.clamp(image * brightness, 0.0, 1.0)

        # Contrast
        mean = image.mean()
        contrast = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)
        image = torch.clamp((image - mean) * contrast + mean, 0.0, 1.0)

        return image, mask


class RandomElasticDeform:
    """
    Apply random elastic deformation (simplified).
    Critical augmentation for biomedical segmentation per the original U-Net paper.
    Uses random displacement fields smoothed with Gaussian blur.
    """

    def __init__(self, alpha=20.0, sigma=3.0, p=0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, image, mask):
        if random.random() > self.p:
            return image, mask

        _, h, w = image.shape
        # Generate random displacement field
        dx = torch.randn(1, 1, h, w) * self.alpha
        dy = torch.randn(1, 1, h, w) * self.alpha

        # Smooth with Gaussian-like averaging
        k = int(self.sigma * 3) | 1  # ensure odd
        dx = F.avg_pool2d(dx, kernel_size=k, stride=1, padding=k // 2) * k * k
        dy = F.avg_pool2d(dy, kernel_size=k, stride=1, padding=k // 2) * k * k

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        grid[..., 0] += dx.squeeze() * 2.0 / w
        grid[..., 1] += dy.squeeze() * 2.0 / h

        # Apply to both image and mask
        image = F.grid_sample(
            image.unsqueeze(0), grid, mode="bilinear", padding_mode="reflection", align_corners=True
        ).squeeze(0)
        mask = F.grid_sample(
            mask.unsqueeze(0), grid, mode="nearest", padding_mode="zeros", align_corners=True
        ).squeeze(0)

        return image, mask


def get_train_transforms():
    """Standard training augmentation pipeline."""
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation90(),
        RandomBrightnessContrast(brightness_range=0.15, contrast_range=0.15),
        RandomGaussianNoise(std=0.02),
        RandomElasticDeform(alpha=20.0, sigma=3.0, p=0.3),
    ])
