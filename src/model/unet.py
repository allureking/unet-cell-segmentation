"""
U-Net: Convolutional Networks for Biomedical Image Segmentation

Implementation based on:
    Ronneberger, O., Fischer, P., & Brox, T. (2015).
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/abs/1505.04597

Adapted for cell counting in microscopy images via binary segmentation
followed by connected-component labeling.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive conv layers: (Conv2d -> BN -> ReLU) x 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """Downsampling: MaxPool -> DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.pool_conv(x)


class DecoderBlock(nn.Module):
    """Upsampling: ConvTranspose2d -> Concat skip -> DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle potential size mismatch from odd dimensions
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = nn.functional.pad(
            x, (diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2)
        )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net architecture for binary segmentation.

    Architecture:
        Encoder: 4 downsampling blocks (64 -> 128 -> 256 -> 512)
        Bottleneck: 1024 channels
        Decoder: 4 upsampling blocks (512 -> 256 -> 128 -> 64)
        Output: 1x1 conv to single channel (sigmoid applied externally)

    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        out_channels: Number of output channels (1 for binary segmentation)
        base_features: Number of features in the first encoder block
    """

    def __init__(self, in_channels=1, out_channels=1, base_features=64):
        super().__init__()
        f = base_features

        # Encoder path
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = EncoderBlock(f, f * 2)
        self.enc3 = EncoderBlock(f * 2, f * 4)
        self.enc4 = EncoderBlock(f * 4, f * 8)

        # Bottleneck
        self.bottleneck = EncoderBlock(f * 8, f * 16)

        # Decoder path
        self.dec4 = DecoderBlock(f * 16, f * 8)
        self.dec3 = DecoderBlock(f * 8, f * 4)
        self.dec2 = DecoderBlock(f * 4, f * 2)
        self.dec1 = DecoderBlock(f * 2, f)

        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.final_conv(d1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
