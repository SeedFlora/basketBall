"""Loss construction using training-split class frequencies only."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn


def class_counts(labels: Iterable[int] | np.ndarray | Tensor, num_classes: int) -> Tensor:
    """Count zero-based class labels with explicit validation."""

    values = torch.as_tensor(labels, dtype=torch.long).flatten().cpu()
    if values.numel() == 0:
        raise ValueError("Cannot calculate class counts from an empty training split")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if values.min().item() < 0 or values.max().item() >= num_classes:
        raise ValueError(
            f"Training labels must be in [0, {num_classes - 1}], got "
            f"[{values.min().item()}, {values.max().item()}]"
        )
    return torch.bincount(values, minlength=num_classes)


def compute_class_weights(
    labels: Iterable[int] | np.ndarray | Tensor,
    num_classes: int,
) -> Tensor:
    """Return balanced inverse-frequency weights from training labels.

    Missing classes receive weight zero rather than infinity. The caller should
    report that class absence because no classifier can learn an unseen class.
    """

    counts = class_counts(labels, num_classes).to(dtype=torch.float32)
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return weights


def resolve_class_weights(
    labels: Iterable[int] | np.ndarray | Tensor,
    num_classes: int,
    weighting: str | bool | Sequence[float] | Tensor | None = "auto",
) -> Tensor | None:
    """Resolve ``auto``, disabled, or explicit CrossEntropy weights."""

    if weighting is None or weighting is False:
        return None
    if isinstance(weighting, str):
        normalized = weighting.strip().lower()
        if normalized in {"none", "off", "false", "unweighted"}:
            return None
        if normalized != "auto":
            raise ValueError("class_weighting must be 'auto', disabled, or a numeric sequence")
        return compute_class_weights(labels, num_classes)
    if weighting is True:
        return compute_class_weights(labels, num_classes)
    weights = torch.as_tensor(weighting, dtype=torch.float32).flatten()
    if weights.numel() != num_classes:
        raise ValueError(f"Expected {num_classes} class weights, got {weights.numel()}")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("class weights must be finite and non-negative")
    return weights


def build_cross_entropy(
    labels: Iterable[int] | np.ndarray | Tensor,
    num_classes: int,
    weighting: str | bool | Sequence[float] | Tensor | None = "auto",
    device: torch.device | str | None = None,
) -> tuple[nn.CrossEntropyLoss, Tensor | None]:
    """Build CrossEntropyLoss and return the resolved weights for reporting."""

    weights = resolve_class_weights(labels, num_classes, weighting)
    device_weights = weights.to(device=device) if weights is not None else None
    return nn.CrossEntropyLoss(weight=device_weights), weights


make_classification_loss = build_cross_entropy


__all__ = [
    "build_cross_entropy",
    "class_counts",
    "compute_class_weights",
    "make_classification_loss",
    "resolve_class_weights",
]


def reconstruction_score_mask(
    original_valid_mask: Tensor,
    synthetic_mask: Tensor,
) -> Tensor:
    """Return the only scientifically valid reconstruction scoring support."""

    original = torch.as_tensor(original_valid_mask, dtype=torch.bool)
    synthetic = torch.as_tensor(synthetic_mask, dtype=torch.bool, device=original.device)
    if original.shape != synthetic.shape:
        raise ValueError("original_valid_mask and synthetic_mask must have identical shape")
    if bool((synthetic & ~original).any()):
        raise ValueError("synthetic_mask includes originally missing keypoints")
    return original & synthetic


def masked_reconstruction_loss(
    predictions: Tensor,
    targets: Tensor,
    original_valid_mask: Tensor,
    synthetic_mask: Tensor,
    *,
    loss: str = "smooth_l1",
    beta: float = 0.05,
) -> Tensor:
    """Coordinate loss evaluated only on deliberately hidden valid keypoints."""

    if predictions.shape != targets.shape or predictions.ndim < 2:
        raise ValueError("predictions and targets must have matching [...,C] shape")
    support = reconstruction_score_mask(original_valid_mask, synthetic_mask).to(
        device=predictions.device
    )
    if support.shape != predictions.shape[:-1]:
        raise ValueError("reconstruction masks must match predictions without coordinate axis")
    if not bool(support.any()):
        raise ValueError("No synthetically masked valid keypoints are available for loss")
    residual = predictions[support] - targets.to(predictions.device)[support]
    token = loss.strip().lower().replace("-", "_")
    if token in {"smooth_l1", "huber"}:
        return torch.nn.functional.smooth_l1_loss(
            residual, torch.zeros_like(residual), beta=float(beta), reduction="mean"
        )
    if token in {"l1", "mae"}:
        return residual.abs().mean()
    if token in {"l2", "mse"}:
        return residual.square().mean()
    raise ValueError("loss must be smooth_l1, l1, or mse")


class MaskedReconstructionLoss(nn.Module):
    """Module wrapper preserving original/synthetic missingness distinction."""

    def __init__(self, loss: str = "smooth_l1", beta: float = 0.05) -> None:
        super().__init__()
        self.loss = loss
        self.beta = float(beta)

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        original_valid_mask: Tensor,
        synthetic_mask: Tensor,
    ) -> Tensor:
        return masked_reconstruction_loss(
            predictions,
            targets,
            original_valid_mask,
            synthetic_mask,
            loss=self.loss,
            beta=self.beta,
        )


__all__ += [
    "MaskedReconstructionLoss",
    "masked_reconstruction_loss",
    "reconstruction_score_mask",
]
