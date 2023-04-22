"""Unit tests for model architectures."""

import pytest
import torch

from src.model.unet import UNet, DoubleConv, EncoderBlock, DecoderBlock
from src.model.attention_unet import AttentionUNet, AttentionGate
from src.model.losses import DiceLoss, BCEDiceLoss, FocalLoss, TverskyLoss


class TestDoubleConv:
    def test_output_shape(self):
        block = DoubleConv(1, 64)
        x = torch.randn(2, 1, 128, 128)
        out = block(x)
        assert out.shape == (2, 64, 128, 128)

    def test_channel_change(self):
        block = DoubleConv(64, 128)
        x = torch.randn(1, 64, 32, 32)
        assert block(x).shape == (1, 128, 32, 32)


class TestUNet:
    def test_output_shape_default(self):
        model = UNet(in_channels=1, out_channels=1, base_features=64)
        x = torch.randn(2, 1, 128, 128)
        out = model(x)
        assert out.shape == (2, 1, 128, 128)

    def test_output_shape_small(self):
        model = UNet(in_channels=1, out_channels=1, base_features=32)
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        assert out.shape == (1, 1, 64, 64)

    def test_rgb_input(self):
        model = UNet(in_channels=3, out_channels=1, base_features=32)
        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert out.shape == (1, 1, 128, 128)

    def test_multi_class_output(self):
        model = UNet(in_channels=1, out_channels=3, base_features=32)
        x = torch.randn(1, 1, 128, 128)
        out = model(x)
        assert out.shape == (1, 3, 128, 128)

    def test_odd_input_size(self):
        """U-Net should handle non-power-of-2 dimensions gracefully."""
        model = UNet(in_channels=1, out_channels=1, base_features=32)
        x = torch.randn(1, 1, 100, 100)
        out = model(x)
        assert out.shape[2:] == x.shape[2:]

    def test_parameter_count(self):
        model = UNet(base_features=64)
        params = model.count_parameters()
        assert params > 0
        # 64-base U-Net should have ~31M params
        assert params > 1_000_000

    def test_gradient_flow(self):
        model = UNet(base_features=32)
        x = torch.randn(1, 1, 64, 64, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestAttentionUNet:
    def test_output_shape(self):
        model = AttentionUNet(in_channels=1, out_channels=1, base_features=32)
        x = torch.randn(2, 1, 128, 128)
        out = model(x)
        assert out.shape == (2, 1, 128, 128)

    def test_attention_maps(self):
        model = AttentionUNet(in_channels=1, out_channels=1, base_features=32)
        x = torch.randn(1, 1, 128, 128)
        out, attn_maps = model.get_attention_maps(x)
        assert out.shape == (1, 1, 128, 128)
        assert len(attn_maps) == 4
        # Attention maps should be between 0 and 1 (sigmoid output)
        for am in attn_maps:
            assert am.min() >= 0
            assert am.max() <= 1

    def test_more_params_than_unet(self):
        unet = UNet(base_features=32)
        attn_unet = AttentionUNet(base_features=32)
        assert attn_unet.count_parameters() > unet.count_parameters()


class TestAttentionGate:
    def test_output_preserves_shape(self):
        gate = AttentionGate(gate_channels=128, skip_channels=128, inter_channels=64)
        g = torch.randn(1, 128, 16, 16)
        s = torch.randn(1, 128, 16, 16)
        out = gate(g, s)
        assert out.shape == s.shape


class TestLossFunctions:
    def setup_method(self):
        self.logits = torch.randn(4, 1, 32, 32)
        self.targets = torch.randint(0, 2, (4, 1, 32, 32)).float()

    def test_dice_loss_range(self):
        loss_fn = DiceLoss()
        loss = loss_fn(self.logits, self.targets)
        assert 0 <= loss.item() <= 1

    def test_dice_loss_perfect(self):
        loss_fn = DiceLoss()
        targets = torch.ones(1, 1, 8, 8)
        logits = torch.ones(1, 1, 8, 8) * 10  # strong positive
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.1

    def test_bce_dice_loss(self):
        loss_fn = BCEDiceLoss(alpha=0.5)
        loss = loss_fn(self.logits, self.targets)
        assert loss.item() > 0

    def test_focal_loss(self):
        loss_fn = FocalLoss()
        loss = loss_fn(self.logits, self.targets)
        assert loss.item() > 0

    def test_tversky_loss(self):
        loss_fn = TverskyLoss(alpha=0.3, beta=0.7)
        loss = loss_fn(self.logits, self.targets)
        assert 0 <= loss.item() <= 1

    def test_all_losses_differentiable(self):
        for LossClass in [DiceLoss, BCEDiceLoss, FocalLoss, TverskyLoss]:
            logits = torch.randn(2, 1, 16, 16, requires_grad=True)
            targets = torch.randint(0, 2, (2, 1, 16, 16)).float()
            loss = LossClass()(logits, targets)
            loss.backward()
            assert logits.grad is not None
