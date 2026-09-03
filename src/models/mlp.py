"""Aggregated-statistics MLP baseline for pose sequences."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _frame_mask(valid_mask: Tensor | None, x: Tensor) -> Tensor:
    """Return a boolean ``[B, T]`` mask for sequence tensors."""

    if valid_mask is None:
        return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    mask = valid_mask.to(device=x.device, dtype=torch.bool)
    if mask.ndim > 2:
        mask = mask.flatten(2).any(dim=-1)
    if mask.shape != x.shape[:2]:
        raise ValueError(
            f"valid_mask must start with {tuple(x.shape[:2])}, got {tuple(valid_mask.shape)}"
        )
    return mask


def _masked_moments(x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    weights = mask.unsqueeze(-1).to(dtype=x.dtype)
    count = weights.sum(dim=1).clamp_min(1.0)
    mean = (x * weights).sum(dim=1) / count
    variance = ((x - mean.unsqueeze(1)).square() * weights).sum(dim=1) / count

    pos_inf = torch.tensor(float("inf"), dtype=x.dtype, device=x.device)
    neg_inf = torch.tensor(float("-inf"), dtype=x.dtype, device=x.device)
    minimum = x.masked_fill(~mask.unsqueeze(-1), pos_inf).amin(dim=1)
    maximum = x.masked_fill(~mask.unsqueeze(-1), neg_inf).amax(dim=1)
    empty = ~mask.any(dim=1)
    minimum = torch.where(empty.unsqueeze(-1), torch.zeros_like(minimum), minimum)
    maximum = torch.where(empty.unsqueeze(-1), torch.zeros_like(maximum), maximum)
    return mean, variance.clamp_min(0).sqrt(), minimum, maximum


class AggregatedMLP(nn.Module):
    """MLP over temporal mean/std/range and first-difference statistics.

    Parameters
    ----------
    in_features:
        Feature dimension per frame. Input tensors have shape ``[B, T, F]``.
    num_classes:
        Number of action classes.
    hidden_dims:
        Compact hidden-layer widths. The default is far below five million
        parameters for the pose representations used by this project.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 3,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if in_features <= 0 or num_classes <= 0:
            raise ValueError("in_features and num_classes must be positive")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")

        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.embedding_dim = int(hidden_dims[-1])

        layers: list[nn.Module] = []
        width = 6 * self.in_features
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(width, int(hidden)),
                    nn.LayerNorm(int(hidden)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            width = int(hidden)
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes)

    def aggregate(self, x: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"AggregatedMLP expects [B,T,F], got {tuple(x.shape)}")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected F={self.in_features}, got F={x.shape[-1]}")
        mask = _frame_mask(valid_mask, x)
        mean, std, minimum, maximum = _masked_moments(x, mask)
        if x.shape[1] > 1:
            velocity = x[:, 1:] - x[:, :-1]
            velocity_mask = mask[:, 1:] & mask[:, :-1]
            velocity_mean, velocity_std, _, _ = _masked_moments(velocity, velocity_mask)
        else:
            velocity_mean = torch.zeros_like(mean)
            velocity_std = torch.zeros_like(std)
        return torch.cat((mean, std, minimum, maximum, velocity_mean, velocity_std), dim=-1)

    def extract_features(self, x: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        """Return the penultimate embedding."""

        return self.encoder(self.aggregate(x, valid_mask))

    def forward(
        self,
        x: Tensor,
        valid_mask: Tensor | None = None,
        return_embedding: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        embedding = self.extract_features(x, valid_mask)
        logits = self.classifier(embedding)
        return (logits, embedding) if return_embedding else logits


# Friendly aliases used by experiment scripts and external notebooks.
MLPBaseline = AggregatedMLP
PoseMLP = AggregatedMLP
MLPClassifier = AggregatedMLP
