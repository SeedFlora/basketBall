"""TrackID3x3 reconstruction metrics plus archived classification helpers.

NMJE is primary for the public masked-keypoint workflow. The classification
functions and their historical default labels remain only for backward-
compatible tests and pre-pivot artifacts; TrackID3x3 supplies no action labels.
All public functions return JSON-serialisable Python values.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

DEFAULT_CLASS_NAMES: dict[int, str] = {0: "dribble", 1: "lay-up", 2: "shoot"}
SUMMARY_METRICS: tuple[str, ...] = (
    "macro_f1",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "weighted_f1",
)

RECONSTRUCTION_SUMMARY_METRICS: tuple[str, ...] = (
    "nmje",
    "pck_at_0_05",
    "pck_at_0_10",
    "pck_at_0_20",
)
DEFAULT_PCK_THRESHOLDS: tuple[float, ...] = (0.05, 0.10, 0.20)


def pck_metric_key(threshold: float) -> str:
    """Return the stable snake-case key for a PCK threshold."""

    return f"pck_at_{float(threshold):.2f}".replace(".", "_")


@dataclass(frozen=True, slots=True)
class ReconstructionErrorArrays:
    """Dense reconstruction errors; unscored locations are represented by NaN."""

    normalized: np.ndarray
    coordinate_units: np.ndarray
    scoring_mask: np.ndarray
    bbox_scale: np.ndarray


def bbox_normalization_scale(
    bboxes: Any,
    *,
    bbox_format: str = "xywh",
    normalization: str = "diagonal",
    target_leading_shape: Sequence[int] | None = None,
    valid_mask: Any | None = None,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Return a positive per-frame scale for bbox-normalized pose errors.

    TrackID3x3 bounding boxes use ``xywh``.  The primary convention is the
    Euclidean bbox diagonal, ``sqrt(width**2 + height**2)``.  Alternative
    conventions are explicit to prevent numerically incomparable runs from
    being silently pooled.
    """

    boxes = np.asarray(bboxes, dtype=float)
    if boxes.ndim < 1 or boxes.shape[-1] != 4:
        raise ValueError("bboxes must end in four values")
    token = bbox_format.lower().replace("-", "").replace("_", "")
    if token == "xywh":
        width, height = boxes[..., 2], boxes[..., 3]
    elif token in {"xyxy", "ltrb"}:
        width = boxes[..., 2] - boxes[..., 0]
        height = boxes[..., 3] - boxes[..., 1]
    else:
        raise ValueError("bbox_format must be 'xywh' or 'xyxy'")
    if target_leading_shape is not None:
        target_shape = tuple(int(value) for value in target_leading_shape)
        while width.ndim < len(target_shape):
            width = np.expand_dims(width, axis=-1)
            height = np.expand_dims(height, axis=-1)
        try:
            width = np.broadcast_to(width, target_shape)
            height = np.broadcast_to(height, target_shape)
        except ValueError as error:
            raise ValueError(
                f"bbox leading shape {boxes.shape[:-1]} cannot broadcast to {target_shape}"
            ) from error
    checked = np.ones_like(width, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if checked.shape != width.shape:
        raise ValueError(f"bbox valid_mask must have shape {width.shape}")
    if np.any((~np.isfinite(width) | ~np.isfinite(height)) & checked):
        raise ValueError("bbox dimensions contain non-finite values at scored frames")
    if np.any(((width <= epsilon) | (height <= epsilon)) & checked):
        raise ValueError("bbox width and height must be positive at scored frames")
    safe_width = np.where(checked, width, 1.0)
    safe_height = np.where(checked, height, 1.0)

    normalization_token = normalization.lower().replace("-", "_")
    if normalization_token in {"diagonal", "diag"}:
        scale = np.hypot(safe_width, safe_height)
    elif normalization_token in {"max_side", "max"}:
        scale = np.maximum(safe_width, safe_height)
    elif normalization_token in {"sqrt_area", "area_sqrt"}:
        scale = np.sqrt(safe_width * safe_height)
    else:
        raise ValueError("normalization must be diagonal, max_side, or sqrt_area")
    return np.asarray(scale, dtype=float)


def reconstruction_error_arrays(
    prediction: Any,
    target: Any,
    *,
    bboxes: Any,
    evaluation_mask: Any,
    original_valid_mask: Any | None = None,
    sequence_mask: Any | None = None,
    bbox_format: str = "xywh",
    normalization: str = "diagonal",
) -> ReconstructionErrorArrays:
    """Compute dense Euclidean and bbox-normalized joint errors.

    Scoring is restricted to ``evaluation_mask & original_valid_mask`` and, if
    supplied, valid sequence frames.  In the masking experiment,
    ``evaluation_mask`` is the synthetic held-out mask: naturally absent joints
    are never treated as reconstruction targets.
    """

    predicted = np.asarray(prediction, dtype=float)
    expected = np.asarray(target, dtype=float)
    if predicted.shape != expected.shape or predicted.ndim < 3 or predicted.shape[-1] != 2:
        raise ValueError("prediction and target must share shape [..., T, V, 2]")
    joint_shape = expected.shape[:-1]
    held_out = np.asarray(evaluation_mask, dtype=bool)
    if held_out.shape == expected.shape:
        held_out = np.all(held_out, axis=-1)
    if held_out.shape != joint_shape:
        raise ValueError(f"evaluation_mask must have shape {joint_shape}")
    scoring = held_out.copy()
    if original_valid_mask is not None:
        valid = np.asarray(original_valid_mask, dtype=bool)
        if valid.shape == expected.shape:
            valid = np.all(valid, axis=-1)
        if valid.shape != joint_shape:
            raise ValueError(f"original_valid_mask must have shape {joint_shape}")
        scoring &= valid
    if sequence_mask is not None:
        frames = np.asarray(sequence_mask, dtype=bool)
        expected_frame_shape = expected.shape[:-2]
        if frames.shape != expected_frame_shape:
            raise ValueError(f"sequence_mask must have shape {expected_frame_shape}")
        scoring &= np.broadcast_to(frames[..., None], joint_shape)
    if not np.any(scoring):
        raise ValueError("no valid synthetically held-out joints are available for scoring")
    if np.any(~np.isfinite(expected[scoring])):
        raise ValueError("target contains non-finite values at scored joints")
    if np.any(~np.isfinite(predicted[scoring])):
        raise ValueError("prediction contains non-finite values at scored joints")

    scales = bbox_normalization_scale(
        bboxes,
        bbox_format=bbox_format,
        normalization=normalization,
        target_leading_shape=expected.shape[:-2],
        valid_mask=np.any(scoring, axis=-1),
    )
    distances = np.linalg.norm(predicted - expected, axis=-1)
    normalized = distances / scales[..., None]
    dense_distance = np.where(scoring, distances, np.nan)
    dense_normalized = np.where(scoring, normalized, np.nan)
    return ReconstructionErrorArrays(
        normalized=dense_normalized,
        coordinate_units=dense_distance,
        scoring_mask=scoring,
        bbox_scale=scales,
    )


def reconstruction_metrics(
    prediction: Any,
    target: Any,
    *,
    bboxes: Any,
    evaluation_mask: Any,
    original_valid_mask: Any | None = None,
    sequence_mask: Any | None = None,
    joint_names: Sequence[str] | None = None,
    bbox_format: str = "xywh",
    normalization: str = "diagonal",
    pck_thresholds: Sequence[float] = DEFAULT_PCK_THRESHOLDS,
) -> dict[str, Any]:
    """Compute TrackID3x3 masked-keypoint reconstruction metrics.

    NMJE (lower is better) is primary.  PCK is the fraction of scored joints
    whose normalized error is at or below each requested threshold.
    """

    arrays = reconstruction_error_arrays(
        prediction,
        target,
        bboxes=bboxes,
        evaluation_mask=evaluation_mask,
        original_valid_mask=original_valid_mask,
        sequence_mask=sequence_mask,
        bbox_format=bbox_format,
        normalization=normalization,
    )
    thresholds = [float(value) for value in pck_thresholds]
    if not thresholds or any(not np.isfinite(value) or value <= 0 for value in thresholds):
        raise ValueError("pck_thresholds must contain positive finite values")
    errors = arrays.normalized[arrays.scoring_mask]
    coordinate_errors = arrays.coordinate_units[arrays.scoring_mask]
    num_joints = arrays.normalized.shape[-1]
    if joint_names is None:
        names = [f"joint_{index}" for index in range(num_joints)]
    else:
        names = [str(value) for value in joint_names]
        if len(names) != num_joints:
            raise ValueError("joint_names must have one entry per joint")

    pck = {f"{threshold:.2f}": float(np.mean(errors <= threshold)) for threshold in thresholds}
    result: dict[str, Any] = {
        "task": "keypoint_reconstruction",
        "primary_metric": "nmje",
        "primary_metric_direction": "lower_is_better",
        "bbox_format": bbox_format,
        "normalization": f"bbox_{normalization}",
        "n_scored_joints": int(errors.size),
        "n_sequences_with_scored_joints": int(
            1
            if arrays.scoring_mask.ndim == 2 and arrays.scoring_mask.any()
            else np.count_nonzero(
                np.any(
                    arrays.scoring_mask.reshape(arrays.scoring_mask.shape[0], -1),
                    axis=1,
                )
            )
        ),
        "nmje": float(np.mean(errors)),
        "mean_joint_error_coordinate_units": float(np.mean(coordinate_errors)),
        "pck": pck,
    }
    for threshold, value in zip(thresholds, pck.values(), strict=True):
        key = pck_metric_key(threshold)
        result[key] = value
        result[f"PCK@{threshold:.2f}"] = value

    per_joint: dict[str, dict[str, float | int | None]] = {}
    for joint_index, name in enumerate(names):
        selected = arrays.scoring_mask[..., joint_index]
        joint_errors = arrays.normalized[..., joint_index][selected]
        joint_coordinate_errors = arrays.coordinate_units[..., joint_index][selected]
        if joint_errors.size == 0:
            per_joint[name] = {
                "index": joint_index,
                "count": 0,
                "nmje": None,
                "mean_joint_error_coordinate_units": None,
                **{pck_metric_key(threshold): None for threshold in thresholds},
            }
            continue
        joint_result: dict[str, float | int | None] = {
            "index": joint_index,
            "count": int(joint_errors.size),
            "nmje": float(np.mean(joint_errors)),
            "mean_joint_error_coordinate_units": float(np.mean(joint_coordinate_errors)),
        }
        joint_result.update(
            {
                pck_metric_key(threshold): float(
                    np.mean(joint_errors <= threshold)
                )
                for threshold in thresholds
            }
        )
        per_joint[name] = joint_result
    result["per_joint"] = per_joint
    return result


def compute_reconstruction_metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for :func:`reconstruction_metrics`."""

    return reconstruction_metrics(*args, **kwargs)


def _label_vector(values: Any, *, name: str) -> np.ndarray:
    """Convert labels, probabilities, or logits to a one-dimensional vector."""

    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim == 2:
        if array.shape[1] == 1:
            array = array[:, 0]
        else:
            array = np.argmax(array, axis=1)
    elif array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional labels or a 2-D score matrix")

    if array.size and not np.all(np.isfinite(array.astype(float, copy=False))):
        raise ValueError(f"{name} contains non-finite values")
    return array.astype(int, copy=False)


def _normalise_class_names(
    labels: Sequence[int], class_names: Mapping[int, str] | Sequence[str] | None
) -> list[str]:
    if class_names is None:
        return [DEFAULT_CLASS_NAMES.get(int(label), str(int(label))) for label in labels]
    if isinstance(class_names, Mapping):
        return [str(class_names.get(int(label), str(int(label)))) for label in labels]
    names = [str(value) for value in class_names]
    if len(names) != len(labels):
        raise ValueError("class_names must contain exactly one name per label")
    return names


def classification_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Sequence[int] | None = None,
    class_names: Mapping[int, str] | Sequence[str] | None = None,
    zero_division: int = 0,
) -> dict[str, Any]:
    """Compute the complete classification metric set.

    Parameters
    ----------
    y_true:
        Ground-truth integer labels (or a one-hot matrix).
    y_pred:
        Predicted integer labels or a ``[N, C]`` probability/logit matrix.
    labels:
        Ordered labels for reporting.  By default the union of observed labels
        is used, except that the canonical three basketball labels are retained
        whenever every observed value belongs to ``{0, 1, 2}``.
    class_names:
        Either a label-to-name mapping or names aligned with ``labels``.

    Returns
    -------
    dict
        JSON-serialisable summary, per-class metrics, and confusion matrix.
    """

    truth = _label_vector(y_true, name="y_true")
    prediction = _label_vector(y_pred, name="y_pred")
    if truth.shape[0] != prediction.shape[0]:
        raise ValueError("y_true and y_pred must contain the same number of samples")
    if truth.size == 0:
        raise ValueError("classification metrics require at least one sample")

    observed = sorted(set(truth.tolist()) | set(prediction.tolist()))
    if labels is None:
        label_values = [0, 1, 2] if set(observed).issubset(DEFAULT_CLASS_NAMES) else observed
    else:
        label_values = [int(label) for label in labels]
        if len(label_values) != len(set(label_values)):
            raise ValueError("labels must be unique")
        unknown = set(observed).difference(label_values)
        if unknown:
            raise ValueError(f"observed labels are absent from labels: {sorted(unknown)}")
    names = _normalise_class_names(label_values, class_names)

    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        prediction,
        labels=label_values,
        average=None,
        zero_division=zero_division,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        truth,
        prediction,
        labels=label_values,
        average="macro",
        zero_division=zero_division,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        truth,
        prediction,
        labels=label_values,
        average="weighted",
        zero_division=zero_division,
    )
    matrix = confusion_matrix(truth, prediction, labels=label_values)

    per_class: dict[str, dict[str, int | float]] = {}
    for index, (label, name) in enumerate(zip(label_values, names, strict=True)):
        per_class[name] = {
            "label": int(label),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }

    return {
        "primary_metric": "macro_f1",
        "n_samples": int(truth.size),
        "labels": [int(value) for value in label_values],
        "class_names": names,
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def compute_classification_metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible descriptive alias for :func:`classification_metrics`."""

    return classification_metrics(*args, **kwargs)


def compute_metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Concise alias for :func:`classification_metrics`."""

    return classification_metrics(*args, **kwargs)


def aggregate_metric_runs(
    runs: Iterable[Mapping[str, Any]],
    *,
    metric_names: Sequence[str] = SUMMARY_METRICS,
) -> dict[str, dict[str, float | int]]:
    """Aggregate only metrics that are actually present in executed runs.

    Missing values, non-numeric values, and non-finite values are ignored.  A
    metric absent from every run is omitted rather than represented by a zero.
    Population standard deviation (``ddof=0``) is used; one run therefore has a
    standard deviation of zero without implying multi-seed evidence.
    """

    materialised = list(runs)
    aggregated: dict[str, dict[str, float | int]] = {}
    for metric in metric_names:
        values: list[float] = []
        for run in materialised:
            value = run.get(metric)
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric = float(value)
                if np.isfinite(numeric):
                    values.append(numeric)
        if values:
            aggregated[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "n": len(values),
            }
    return aggregated


def summary_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Return the finite scalar summary metrics present in ``metrics``."""

    result: dict[str, float] = {}
    for key in SUMMARY_METRICS:
        value = metrics.get(key)
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
            result[key] = float(value)
    return result
