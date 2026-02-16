"""
Residual U-Net: U-Net with residual connections in each conv block.

Combines skip connections from U-Net with residual learning from ResNet.
The residual connections help with gradient flow in deeper networks and
can improve convergence speed.
"""

import torch
import torch.nn as nn


class ResidualConv(nn.Module):
    """
    Residual convolutional block.

    Two conv layers with a skip connection:
        x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU

    If input/output channels differ, a 1x1 conv adjusts the residual path.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv_block(x) + self.shortcut(x))


class ResUNet(nn.Module):
    """
    U-Net with residual connections.

    Same encoder-decoder structure as standard U-Net, but each DoubleConv
    block is replaced with a ResidualConv block for better gradient flow.
    """

    def __init__(self, in_channels=1, out_channels=1, base_features=64):
        super().__init__()
        f = base_features

        # Encoder
        self.enc1 = ResidualConv(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualConv(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualConv(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualConv(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualConv(f * 8, f * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = ResidualConv(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualConv(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualConv(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ResidualConv(f * 2, f)

        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(self._match_and_cat(self.up4(b), e4))
        d3 = self.dec3(self._match_and_cat(self.up3(d4), e3))
        d2 = self.dec2(self._match_and_cat(self.up2(d3), e2))
        d1 = self.dec1(self._match_and_cat(self.up1(d2), e1))

        return self.final_conv(d1)

    def _match_and_cat(self, x, skip):
        """Pad x to match skip dimensions, then concatenate."""
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = nn.functional.pad(
            x, (diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2)
        )
        return torch.cat([x, skip], dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
