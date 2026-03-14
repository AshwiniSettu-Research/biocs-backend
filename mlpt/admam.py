"""
Adaptive Depthwise Multi-Kernel Atrous Module (ADMAM).

Modified ASPP (Atrous Spatial Pyramid Pooling) adapted for 1D peptide sequences.
Uses multi-kernel convolutions, depthwise convolutions, and depthwise separable
convolutions to capture multi-scale hierarchical features.
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=False):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation, groups=in_channels, bias=bias,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ADMAM(nn.Module):
    """
    Adaptive Depthwise Multi-Kernel Atrous Module.

    Applies 5 parallel convolution branches with different kernel sizes and
    dilation rates to capture multi-scale features, then fuses them.

    Input:  (batch, in_channels, seq_len) -- e.g. (B, 39, 64)
    Output: (batch, out_channels, seq_len) -- e.g. (B, 64, 64)
    """

    def __init__(self, in_channels=39, out_channels=64):
        super().__init__()

        # Branch 1: 1x1 convolution (pointwise, captures local features)
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Branch 2: Depthwise separable conv, kernel=3, dilation=6
        self.branch2 = nn.Sequential(
            DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size=3, dilation=6),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Branch 3: Depthwise separable conv, kernel=5, dilation=12
        self.branch3 = nn.Sequential(
            DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size=5, dilation=12),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Branch 4: Depthwise separable conv, kernel=7, dilation=18
        self.branch4 = nn.Sequential(
            DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size=7, dilation=18),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Branch 5: Global average pooling + 1x1 conv
        self.branch5_pool = nn.AdaptiveAvgPool1d(1)
        self.branch5_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion: concatenate 5 branches -> fuse to out_channels
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        seq_len = x.shape[2]

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # Global pooling branch: pool to 1, conv, then broadcast back
        b5 = self.branch5_pool(x)
        b5 = self.branch5_conv(b5)
        b5 = b5.expand(-1, -1, seq_len)

        # Concatenate and fuse
        concat = torch.cat([b1, b2, b3, b4, b5], dim=1)
        out = self.fusion(concat)
        return out
