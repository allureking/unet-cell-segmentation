"""
Attention U-Net: Learning Where to Look for the Pancreas

Implementation based on:
    Oktay, O. et al. (2018).
    "Attention U-Net: Learning Where to Look for the Pancreas"
    https://arxiv.org/abs/1804.03999

Adds attention gates to the standard U-Net decoder path, allowing
the model to focus on relevant spatial regions while suppressing
irrelevant background features.
"""

import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """
    Attention Gate module.

    Learns a spatial attention map from the gating signal (decoder features)
    and the skip connection (encoder features). The attention weights
    highlight salient regions and suppress irrelevant ones.

    Args:
        gate_channels: channels from decoder (gating signal)
        skip_channels: channels from encoder (skip connection)
        inter_channels: intermediate channel dimension
    """

    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.W_gate(gate)
        x = self.W_skip(skip)

        # Upsample gate to match skip resolution if needed
        if g.shape[2:] != x.shape[2:]:
            g = nn.functional.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=True)

        attention = self.psi(self.relu(g + x))
        return skip * attention


class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) x 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionUNet(nn.Module):
    """
    U-Net with Attention Gates.

    The attention mechanism allows the decoder to focus on cell regions
    and ignore background noise in microscopy images.

    Architecture:
        Encoder: Standard 4-level encoder (same as U-Net)
        Decoder: Attention gates applied before skip concatenation
        Output: 1x1 conv to single channel

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        base_features: Base feature count (doubled at each level)
    """

    def __init__(self, in_channels=1, out_channels=1, base_features=64):
        super().__init__()
        f = base_features

        # Encoder
        self.enc1 = DoubleConv(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(f * 8, f * 16)

        # Decoder with attention gates
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.attn4 = AttentionGate(f * 8, f * 8, f * 4)
        self.dec4 = DoubleConv(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.attn3 = AttentionGate(f * 4, f * 4, f * 2)
        self.dec3 = DoubleConv(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.attn2 = AttentionGate(f * 2, f * 2, f)
        self.dec2 = DoubleConv(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.attn1 = AttentionGate(f, f, f // 2)
        self.dec1 = DoubleConv(f * 2, f)

        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with attention
        d4 = self.up4(b)
        e4_att = self.attn4(gate=d4, skip=e4)
        d4 = self._pad_and_cat(d4, e4_att)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        e3_att = self.attn3(gate=d3, skip=e3)
        d3 = self._pad_and_cat(d3, e3_att)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2_att = self.attn2(gate=d2, skip=e2)
        d2 = self._pad_and_cat(d2, e2_att)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_att = self.attn1(gate=d1, skip=e1)
        d1 = self._pad_and_cat(d1, e1_att)
        d1 = self.dec1(d1)

        return self.final_conv(d1)

    def _pad_and_cat(self, x, skip):
        """Handle potential size mismatch and concatenate."""
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = nn.functional.pad(
            x, (diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2)
        )
        return torch.cat([x, skip], dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_attention_maps(self, x):
        """
        Forward pass that also returns attention maps for visualization.

        Returns:
            (output, attention_maps) where attention_maps is a list of 4 tensors
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        attention_maps = []

        d4 = self.up4(b)
        e4_att = self.attn4(gate=d4, skip=e4)
        # Extract attention weights (before multiplication)
        g4 = self.attn4.W_gate(d4)
        if g4.shape[2:] != e4.shape[2:]:
            g4 = nn.functional.interpolate(g4, size=e4.shape[2:], mode="bilinear", align_corners=True)
        x4 = self.attn4.W_skip(e4)
        attn4_map = self.attn4.psi(self.attn4.relu(g4 + x4))
        attention_maps.append(attn4_map)

        d4 = self._pad_and_cat(d4, e4_att)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        e3_att = self.attn3(gate=d3, skip=e3)
        g3 = self.attn3.W_gate(d3)
        if g3.shape[2:] != e3.shape[2:]:
            g3 = nn.functional.interpolate(g3, size=e3.shape[2:], mode="bilinear", align_corners=True)
        x3 = self.attn3.W_skip(e3)
        attn3_map = self.attn3.psi(self.attn3.relu(g3 + x3))
        attention_maps.append(attn3_map)

        d3 = self._pad_and_cat(d3, e3_att)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2_att = self.attn2(gate=d2, skip=e2)
        g2 = self.attn2.W_gate(d2)
        if g2.shape[2:] != e2.shape[2:]:
            g2 = nn.functional.interpolate(g2, size=e2.shape[2:], mode="bilinear", align_corners=True)
        x2 = self.attn2.W_skip(e2)
        attn2_map = self.attn2.psi(self.attn2.relu(g2 + x2))
        attention_maps.append(attn2_map)

        d2 = self._pad_and_cat(d2, e2_att)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_att = self.attn1(gate=d1, skip=e1)
        g1 = self.attn1.W_gate(d1)
        if g1.shape[2:] != e1.shape[2:]:
            g1 = nn.functional.interpolate(g1, size=e1.shape[2:], mode="bilinear", align_corners=True)
        x1 = self.attn1.W_skip(e1)
        attn1_map = self.attn1.psi(self.attn1.relu(g1 + x1))
        attention_maps.append(attn1_map)

        d1 = self._pad_and_cat(d1, e1_att)
        d1 = self.dec1(d1)

        return self.final_conv(d1), attention_maps
