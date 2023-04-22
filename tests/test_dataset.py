"""Unit tests for dataset and augmentation."""

import pytest
import numpy as np
import torch

from src.data.dataset import CellDataset
from src.data.augmentation import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation90,
    RandomGaussianNoise,
    RandomBrightnessContrast,
    Compose,
    get_train_transforms,
)
from src.utils.metrics import dice_score, iou_score, count_cells


class TestCellDataset:
    def test_basic_loading(self):
        images = np.random.rand(10, 128, 128).astype(np.float32) * 255
        masks = np.random.randint(0, 2, (10, 128, 128)).astype(np.float32)
        dataset = CellDataset(images, masks)
        assert len(dataset) == 10

    def test_normalization(self):
        images = np.ones((5, 64, 64), dtype=np.float32) * 255
        masks = np.ones((5, 64, 64), dtype=np.float32)
        dataset = CellDataset(images, masks)
        img, msk = dataset[0]
        # Image should be normalized to [0, 1]
        assert img.max().item() == pytest.approx(1.0, abs=1e-5)
        assert img.min().item() >= 0.0

    def test_channel_dimension(self):
        images = np.random.rand(3, 128, 128).astype(np.float32) * 255
        masks = np.random.rand(3, 128, 128).astype(np.float32)
        dataset = CellDataset(images, masks)
        img, msk = dataset[0]
        assert img.shape == (1, 128, 128)
        assert msk.shape == (1, 128, 128)

    def test_no_masks(self):
        images = np.random.rand(5, 64, 64).astype(np.float32) * 255
        dataset = CellDataset(images, masks=None)
        img, msk = dataset[0]
        assert msk.shape == (1, 64, 64)
        assert msk.sum() == 0


class TestAugmentations:
    def setup_method(self):
        self.image = torch.rand(1, 64, 64)
        self.mask = torch.randint(0, 2, (1, 64, 64)).float()

    def test_horizontal_flip(self):
        transform = RandomHorizontalFlip(p=1.0)
        img_aug, mask_aug = transform(self.image, self.mask)
        # Flipped back should equal original
        img_back = torch.flip(img_aug, dims=[-1])
        assert torch.allclose(img_back, self.image)

    def test_vertical_flip(self):
        transform = RandomVerticalFlip(p=1.0)
        img_aug, mask_aug = transform(self.image, self.mask)
        img_back = torch.flip(img_aug, dims=[-2])
        assert torch.allclose(img_back, self.image)

    def test_rotation_preserves_shape(self):
        transform = RandomRotation90()
        img_aug, mask_aug = transform(self.image, self.mask)
        assert img_aug.shape == self.image.shape

    def test_noise_changes_image(self):
        torch.manual_seed(42)
        transform = RandomGaussianNoise(std=0.1)
        img_aug, mask_aug = transform(self.image.clone(), self.mask.clone())
        # Noise should change the image
        assert not torch.allclose(img_aug, self.image, atol=1e-3)
        # But mask should be unchanged
        assert torch.equal(mask_aug, self.mask)

    def test_brightness_preserves_range(self):
        transform = RandomBrightnessContrast(brightness_range=0.5, contrast_range=0.5)
        img_aug, _ = transform(self.image, self.mask)
        assert img_aug.min() >= 0.0
        assert img_aug.max() <= 1.0

    def test_compose_pipeline(self):
        pipeline = get_train_transforms()
        img_aug, mask_aug = pipeline(self.image, self.mask)
        assert img_aug.shape == self.image.shape
        assert mask_aug.shape == self.mask.shape


class TestMetrics:
    def test_dice_perfect_match(self):
        pred = np.ones((32, 32))
        target = np.ones((32, 32))
        score = dice_score(pred, target)
        assert score > 0.99

    def test_dice_no_overlap(self):
        pred = np.ones((32, 32))
        target = np.zeros((32, 32))
        score = dice_score(pred, target)
        assert score < 0.01

    def test_iou_perfect_match(self):
        pred = np.ones((32, 32))
        target = np.ones((32, 32))
        score = iou_score(pred, target)
        assert score > 0.99

    def test_count_cells_single(self):
        mask = np.zeros((64, 64))
        mask[10:20, 10:20] = 1  # One 10x10 cell
        assert count_cells(mask) == 1

    def test_count_cells_multiple(self):
        mask = np.zeros((64, 64))
        mask[5:15, 5:15] = 1    # Cell 1
        mask[30:40, 30:40] = 1  # Cell 2
        mask[50:60, 50:60] = 1  # Cell 3
        assert count_cells(mask) == 3

    def test_count_cells_min_area(self):
        mask = np.zeros((64, 64))
        mask[10:20, 10:20] = 1   # 100 pixels
        mask[40, 40] = 1          # 1 pixel (noise)
        assert count_cells(mask, min_area=10) == 1
        assert count_cells(mask, min_area=1) == 2
