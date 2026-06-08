"""Lightweight residual attention blocks for detector feature maps."""

from __future__ import annotations

import torch
from torch import nn


class SEResidualAttention(nn.Module):
    """SE-style channel attention with an explicit residual path.

    Input and output tensors use detector feature layout ``[B, C, H, W]``.
    The block is intentionally small and built from common export-friendly
    operations for later ONNX/RKNN integration work.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        min_hidden_channels: int = 4,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if reduction <= 0:
            raise ValueError("reduction must be positive")
        if min_hidden_channels <= 0:
            raise ValueError("min_hidden_channels must be positive")

        hidden_channels = min(
            channels,
            max(min_hidden_channels, channels // reduction),
        )
        self.hidden_channels = hidden_channels
        self.residual_scale = float(residual_scale)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep this alias read-only: in-place edits before the final add would
        # corrupt the residual path.
        identity = x
        weights = self.attention(self.pool(x))
        attended = x * weights
        return identity + self.residual_scale * attended


__all__ = ["SEResidualAttention"]
