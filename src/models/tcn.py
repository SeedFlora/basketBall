"""Lightweight temporal convolutional network for pose features."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class TemporalResidualBlock(nn.Module):
    """Two dilated temporal convolutions with a residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so sequence length is preserved")
        padding = dilation * (kernel_size - 1) // 2
        self.temporal = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
        )
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.activation(self.temporal(x) + self.residual(x)))


def _frame_mask(valid_mask: Tensor | None, x: Tensor) -> Tensor:
    if valid_mask is None:
        return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    mask = valid_mask.to(device=x.device, dtype=torch.bool)
    if mask.ndim > 2:
        mask = mask.flatten(2).any(dim=-1)
    if mask.shape != x.shape[:2]:
        raise ValueError(f"valid_mask has shape {tuple(valid_mask.shape)}, expected [B,T,...]")
    return mask


class TCNClassifier(nn.Module):
    """Compact dilated TCN accepting features in ``[B, T, F]`` order."""

    def __init__(
        self,
        in_features: int,
        num_classes: int = 3,
        channels: Sequence[int] = (64, 96, 128),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if in_features <= 0 or num_classes <= 0 or not channels:
            raise ValueError("in_features, num_classes and channels must be positive")
        if any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain positive widths")
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        blocks: list[nn.Module] = []
        width = self.in_features
        for index, channel in enumerate(channels):
            blocks.append(
                TemporalResidualBlock(
                    width,
                    int(channel),
                    dilation=2**index,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )
            width = int(channel)
        self.network = nn.Sequential(*blocks)
        self.embedding_dim = 2 * width
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.embedding_dim = width
        self.classifier = nn.Linear(width, self.num_classes)

    def extract_features(self, x: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"TCNClassifier expects [B,T,F], got {tuple(x.shape)}")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected F={self.in_features}, got F={x.shape[-1]}")
        mask = _frame_mask(valid_mask, x)
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        encoded = self.network(x.transpose(1, 2)).transpose(1, 2)
        weights = mask.unsqueeze(-1).to(dtype=encoded.dtype)
        mean = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        neg_inf = torch.tensor(float("-inf"), dtype=encoded.dtype, device=encoded.device)
        maximum = encoded.masked_fill(~mask.unsqueeze(-1), neg_inf).amax(dim=1)
        maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        return self.projection(torch.cat((mean, maximum), dim=-1))

    def forward(
        self,
        x: Tensor,
        valid_mask: Tensor | None = None,
        return_embedding: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        embedding = self.extract_features(x, valid_mask)
        logits = self.classifier(embedding)
        return (logits, embedding) if return_embedding else logits


TemporalConvNet = TCNClassifier
LightweightTCN = TCNClassifier
TCN = TCNClassifier
