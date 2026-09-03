"""Serializable classification metrics used during training."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import Tensor


def _numpy_labels(values: Iterable[int] | np.ndarray | Tensor) -> np.ndarray:
    if isinstance(values, Tensor):
        values = values.detach().cpu().numpy()
    result = np.asarray(values, dtype=np.int64).reshape(-1)
    return result


def classification_metrics(
    targets: Iterable[int] | np.ndarray | Tensor,
    predictions: Iterable[int] | np.ndarray | Tensor,
    num_classes: int = 3,
    class_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Compute accuracy, aggregate F-scores, per-class scores, and confusion."""

    y_true = _numpy_labels(targets)
    y_pred = _numpy_labels(predictions)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("targets and predictions must be non-empty and equally sized")
    labels = np.arange(num_classes, dtype=np.int64)
    names = list(class_names or [str(index) for index in labels])
    if len(names) != num_classes:
        raise ValueError(f"Expected {num_classes} class names, got {len(names)}")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    per_class = {
        names[index]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in range(num_classes)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
        "n_samples": int(y_true.size),
    }


compute_metrics = classification_metrics


__all__ = ["classification_metrics", "compute_metrics"]


def _expand_joint_scale(scale: Tensor, target_shape: torch.Size) -> Tensor:
    """Broadcast bbox scale to joint mask shape [B,T,V] or [T,V]."""

    value = torch.as_tensor(scale, dtype=torch.float64)
    while value.ndim < len(target_shape):
        value = value.unsqueeze(-1)
    try:
        return value.expand(target_shape)
    except RuntimeError as exc:
        raise ValueError(
            f"bbox scale shape {tuple(scale.shape)} cannot broadcast to {tuple(target_shape)}"
        ) from exc


def reconstruction_metrics(
    predictions: Tensor | np.ndarray,
    targets: Tensor | np.ndarray,
    original_valid_mask: Tensor | np.ndarray,
    synthetic_mask: Tensor | np.ndarray,
    *,
    bbox_diagonal: Tensor | np.ndarray | None = None,
    joint_names: Sequence[str] | None = None,
    pck_thresholds: Sequence[float] = (0.05, 0.10, 0.20),
) -> dict[str, object]:
    """Compute masked reconstruction metrics in source-pixel coordinates.

    The authoritative support is ``original_valid_mask & synthetic_mask``.
    Source-missing joints therefore cannot enter any aggregate, PCK denominator,
    or per-joint result.
    """

    predicted = torch.as_tensor(predictions, dtype=torch.float64).cpu()
    target = torch.as_tensor(targets, dtype=torch.float64).cpu()
    original = torch.as_tensor(original_valid_mask, dtype=torch.bool).cpu()
    synthetic = torch.as_tensor(synthetic_mask, dtype=torch.bool).cpu()
    if predicted.shape != target.shape or original.shape != predicted.shape[:-1]:
        raise ValueError("reconstruction coordinate and validity shapes do not align")
    if synthetic.shape != original.shape:
        raise ValueError("synthetic_mask and original_valid_mask shapes differ")
    if bool((synthetic & ~original).any()):
        raise ValueError("synthetic_mask contains originally missing keypoints")
    support = original & synthetic
    count = int(support.sum())
    if count == 0:
        raise ValueError("No synthetically masked valid keypoints are available for metrics")
    if predicted.ndim < 3 or predicted.shape[-1] != 2:
        raise ValueError("reconstruction metrics expect [...,V,2] coordinates")
    num_joints = int(predicted.shape[-2])
    names = list(joint_names or [str(index) for index in range(num_joints)])
    if len(names) != num_joints:
        raise ValueError(f"Expected {num_joints} joint names, got {len(names)}")
    thresholds = tuple(float(value) for value in pck_thresholds)
    if not thresholds or any(value <= 0 for value in thresholds):
        raise ValueError("pck_thresholds must contain positive values")

    residual_full = predicted - target
    distance_full = torch.linalg.vector_norm(residual_full, dim=-1)
    residual = residual_full[support]
    distances = torch.linalg.vector_norm(residual, dim=-1)
    axis_mae = residual.abs().mean(dim=0)
    result: dict[str, object] = {
        "mae": float(residual.abs().mean()),
        "rmse": float(residual.square().mean().sqrt()),
        "mpjpe": float(distances.mean()),
        "median_joint_error": float(distances.median()),
        "per_axis_mae": [float(value) for value in axis_mae],
        "scored_joint_count": count,
        "coordinate_count": int(residual.numel()),
        "original_valid_joint_count": int(original.sum()),
        "original_missing_joint_count": int((~original).sum()),
        "synthetically_masked_joint_count": int(synthetic.sum()),
        "per_joint_pixel_error": {},
        "per_joint_scored_count": {},
    }
    per_joint_pixel: dict[str, float | None] = {}
    per_joint_count: dict[str, int] = {}
    for index, name in enumerate(names):
        joint_support = support[..., index]
        joint_count = int(joint_support.sum())
        per_joint_count[name] = joint_count
        per_joint_pixel[name] = (
            float(distance_full[..., index][joint_support].mean()) if joint_count else None
        )
    result["per_joint_pixel_error"] = per_joint_pixel
    result["per_joint_scored_count"] = per_joint_count
    if bbox_diagonal is not None:
        scale_full = _expand_joint_scale(
            torch.as_tensor(bbox_diagonal, dtype=torch.float64), support.shape
        )
        usable_support = support & torch.isfinite(scale_full) & (
            scale_full > torch.finfo(scale_full.dtype).eps
        )
        normalized_full = distance_full / scale_full.clamp_min(
            torch.finfo(scale_full.dtype).eps
        )
        normalized = normalized_full[usable_support]
        result["nmje"] = float(normalized.mean()) if normalized.numel() else None
        result["nmje_joint_count"] = int(normalized.numel())
        pck = {
            f"{threshold:.2f}": (
                float((normalized <= threshold).to(torch.float64).mean())
                if normalized.numel()
                else None
            )
            for threshold in thresholds
        }
        result["pck"] = pck
        for threshold, value in zip(thresholds, pck.values(), strict=True):
            result[f"pck@{threshold:.2f}"] = value
        per_joint_normalized: dict[str, float | None] = {}
        for index, name in enumerate(names):
            joint_support = usable_support[..., index]
            per_joint_normalized[name] = (
                float(normalized_full[..., index][joint_support].mean())
                if bool(joint_support.any())
                else None
            )
        result["per_joint_normalized_error"] = per_joint_normalized
    else:
        result["nmje"] = None
        result["nmje_joint_count"] = 0
        result["pck"] = {f"{threshold:.2f}": None for threshold in thresholds}
        for threshold in thresholds:
            result[f"pck@{threshold:.2f}"] = None
        result["per_joint_normalized_error"] = {name: None for name in names}
    return result


compute_reconstruction_metrics = reconstruction_metrics


__all__ += ["compute_reconstruction_metrics", "reconstruction_metrics"]
