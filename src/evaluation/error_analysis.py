"""Archived classification error analysis, outside the TrackID3x3 workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_PCK_THRESHOLDS,
    reconstruction_error_arrays,
)


def _label_names(
    labels: np.ndarray,
    class_names: Mapping[int, str] | Sequence[str] | None,
) -> list[str]:
    if class_names is None:
        mapping = DEFAULT_CLASS_NAMES
    elif isinstance(class_names, Mapping):
        mapping = {int(key): str(value) for key, value in class_names.items()}
    else:
        mapping = {index: str(value) for index, value in enumerate(class_names)}
    return [mapping.get(int(label), str(int(label))) for label in labels]


def _optional_vector(value: Any, n_samples: int, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 0:
        array = np.repeat(array, n_samples)
    if len(array) != n_samples:
        raise ValueError(f"{name} must have one value per sample")
    return array


def _metadata_frame(metadata: Any, n_samples: int) -> pd.DataFrame:
    if metadata is None:
        return pd.DataFrame(index=np.arange(n_samples))
    if isinstance(metadata, pd.DataFrame):
        frame = metadata.reset_index(drop=True).copy()
    elif isinstance(metadata, Mapping):
        frame = pd.DataFrame(dict(metadata))
    else:
        frame = pd.DataFrame(list(metadata))
    if len(frame) != n_samples:
        raise ValueError("metadata must have one row per sample")
    return frame


def sequence_motion_magnitude(
    sequences: Any,
    *,
    valid_mask: Any | None = None,
) -> np.ndarray:
    """Compute mean frame-to-frame joint displacement for each sequence."""

    array = np.asarray(sequences, dtype=float)
    if array.ndim < 3:
        raise ValueError("sequences must have shape [N, T, ..., C]")
    if array.shape[1] < 2:
        return np.zeros(array.shape[0], dtype=float)
    displacement = np.linalg.norm(np.diff(array, axis=1), axis=-1)
    finite = np.isfinite(displacement)
    if valid_mask is not None:
        validity = np.asarray(valid_mask, dtype=bool)
        if validity.shape == array.shape:
            validity = np.all(validity, axis=-1)
        expected = array.shape[:-1]
        if validity.shape != expected:
            raise ValueError(f"valid_mask must have shape {expected} or {array.shape}")
        pair_valid = validity[:, 1:] & validity[:, :-1]
        while pair_valid.ndim < displacement.ndim:
            pair_valid = pair_valid[..., None]
        finite &= pair_valid
    reduce_axes = tuple(range(1, displacement.ndim))
    totals = np.where(finite, displacement, 0.0).sum(axis=reduce_axes)
    counts = finite.sum(axis=reduce_axes)
    return np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)


def missing_joint_fraction(valid_mask: Any) -> np.ndarray:
    """Return the invalid-joint fraction per sample from an explicit mask."""

    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim < 2:
        raise ValueError("valid_mask must include sample and observation dimensions")
    if mask.ndim >= 4:
        mask = np.all(mask, axis=-1)
    axes = tuple(range(1, mask.ndim))
    return 1.0 - np.mean(mask, axis=axes)


def build_error_analysis_table(
    y_true: Any,
    y_pred: Any,
    *,
    sample_ids: Any | None = None,
    class_names: Mapping[int, str] | Sequence[str] | None = DEFAULT_CLASS_NAMES,
    metadata: Any | None = None,
    sequences: Any | None = None,
    sequence_lengths: Any | None = None,
    valid_mask: Any | None = None,
    confidence: Any | None = None,
    include_correct: bool = False,
) -> pd.DataFrame:
    """Build a per-sample table without inventing unavailable attributes.

    Duration, motion, missingness, and confidence columns are included only
    when their source arrays or metadata fields exist.  Consequently an absent
    value is distinguishable from a measured zero.
    """

    truth = np.asarray(y_true).reshape(-1).astype(int)
    prediction_array = np.asarray(y_pred)
    if prediction_array.ndim == 2:
        prediction_array = np.argmax(prediction_array, axis=1)
    prediction = prediction_array.reshape(-1).astype(int)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must contain the same number of samples")
    n_samples = len(truth)
    ids = _optional_vector(sample_ids, n_samples, "sample_ids")
    if ids is None:
        ids = np.asarray([f"sample_{index}" for index in range(n_samples)], dtype=object)

    frame = pd.DataFrame(
        {
            "sample_id": ids.astype(str),
            "true_label": truth,
            "true_class": _label_names(truth, class_names),
            "predicted_label": prediction,
            "predicted_class": _label_names(prediction, class_names),
            "correct": truth == prediction,
        }
    )
    source_metadata = _metadata_frame(metadata, n_samples)

    lengths = _optional_vector(sequence_lengths, n_samples, "sequence_lengths")
    if lengths is not None:
        frame["duration_frames"] = lengths
    else:
        for candidate in ("duration_frames", "num_frames", "sequence_length", "duration"):
            if candidate in source_metadata and source_metadata[candidate].notna().any():
                target = "duration" if candidate == "duration" else "duration_frames"
                frame[target] = source_metadata[candidate].to_numpy()
                break

    if sequences is not None:
        sequence_array = np.asarray(sequences)
        if sequence_array.shape[0] != n_samples:
            raise ValueError("sequences must contain one sequence per sample")
        frame["mean_motion_magnitude"] = sequence_motion_magnitude(
            sequence_array, valid_mask=valid_mask
        )
    else:
        for candidate in ("mean_motion_magnitude", "motion_magnitude", "motion"):
            if candidate in source_metadata and source_metadata[candidate].notna().any():
                frame["mean_motion_magnitude"] = source_metadata[candidate].to_numpy()
                break

    validity = _optional_vector(valid_mask, n_samples, "valid_mask")
    if validity is not None:
        frame["missing_joint_fraction"] = missing_joint_fraction(validity)
    else:
        for candidate in ("missing_joint_fraction", "missing_rate", "missing_joints"):
            if candidate in source_metadata and source_metadata[candidate].notna().any():
                frame["missing_joint_fraction"] = source_metadata[candidate].to_numpy()
                break

    confidence_array = _optional_vector(confidence, n_samples, "confidence")
    if confidence_array is not None:
        if confidence_array.ndim == 1:
            frame["mean_confidence"] = confidence_array.astype(float)
        else:
            axes = tuple(range(1, confidence_array.ndim))
            frame["mean_confidence"] = np.nanmean(
                confidence_array.astype(float), axis=axes
            )
    else:
        for candidate in ("mean_confidence", "confidence", "pose_confidence"):
            if candidate in source_metadata and source_metadata[candidate].notna().any():
                frame["mean_confidence"] = source_metadata[candidate].to_numpy()
                break

    if not include_correct:
        frame = frame.loc[~frame["correct"]]
    return frame.reset_index(drop=True)


def confusion_pair_summary(error_table: pd.DataFrame) -> pd.DataFrame:
    """Count observed error directions; return an empty table if there are none."""

    required = {"true_class", "predicted_class", "correct"}
    missing = required.difference(error_table.columns)
    if missing:
        raise ValueError(f"error table is missing columns: {sorted(missing)}")
    errors = error_table.loc[~error_table["correct"]]
    if errors.empty:
        return pd.DataFrame(columns=["true_class", "predicted_class", "count"])
    return (
        errors.groupby(["true_class", "predicted_class"], sort=True)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def save_error_analysis(table: pd.DataFrame, path: str | Path) -> Path:
    """Write an error table to CSV and return its resolved path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return destination.resolve()


def build_reconstruction_error_table(
    prediction: Any,
    target: Any,
    *,
    bboxes: Any,
    evaluation_mask: Any,
    original_valid_mask: Any | None = None,
    sequence_mask: Any | None = None,
    sample_ids: Any | None = None,
    domains: Any | None = None,
    video_ids: Any | None = None,
    track_ids: Any | None = None,
    mask_ratio: float | None = None,
    pck_thresholds: Sequence[float] = DEFAULT_PCK_THRESHOLDS,
    distal_joint_indices: Sequence[int] = (5, 6, 8, 9),
) -> pd.DataFrame:
    """Create one evidence-backed reconstruction-error row per sequence."""

    predicted = np.asarray(prediction)
    if predicted.ndim != 4:
        raise ValueError("reconstruction error tables require [N,T,V,2] arrays")
    arrays = reconstruction_error_arrays(
        predicted,
        target,
        bboxes=bboxes,
        evaluation_mask=evaluation_mask,
        original_valid_mask=original_valid_mask,
        sequence_mask=sequence_mask,
    )
    n_samples = predicted.shape[0]
    ids = _optional_vector(sample_ids, n_samples, "sample_ids")
    if ids is None:
        ids = np.asarray([f"sequence_{index}" for index in range(n_samples)])
    rows: list[dict[str, Any]] = []
    expected = np.asarray(target, dtype=float)
    source_valid = (
        np.ones(expected.shape[:-1], dtype=bool)
        if original_valid_mask is None
        else np.asarray(original_valid_mask, dtype=bool)
    )
    frame_valid = (
        np.ones(expected.shape[:2], dtype=bool)
        if sequence_mask is None
        else np.asarray(sequence_mask, dtype=bool)
    )
    distal = np.zeros(expected.shape[-2], dtype=bool)
    for joint_index in distal_joint_indices:
        if not 0 <= int(joint_index) < len(distal):
            raise ValueError("distal_joint_indices contains an out-of-range joint")
        distal[int(joint_index)] = True
    optional = {
        "domain": _optional_vector(domains, n_samples, "domains"),
        "video_id": _optional_vector(video_ids, n_samples, "video_ids"),
        "track_id": _optional_vector(track_ids, n_samples, "track_ids"),
    }
    for index in range(n_samples):
        selected = arrays.scoring_mask[index]
        errors = arrays.normalized[index][selected]
        row: dict[str, Any] = {
            "sample_id": str(ids[index]),
            "n_scored_joints": int(errors.size),
            "nmje": float(np.mean(errors)) if errors.size else np.nan,
        }
        scored_frames = np.any(selected, axis=-1) & frame_valid[index]
        scales = arrays.bbox_scale[index][scored_frames]
        row["mean_bbox_diagonal"] = float(np.mean(scales)) if scales.size else np.nan
        radial_values: list[float] = []
        for time_index in np.flatnonzero(frame_valid[index]):
            observed_joints = source_valid[index, time_index]
            points = expected[index, time_index, observed_joints]
            if len(points) < 2:
                continue
            center = points.mean(axis=0)
            spread = float(np.linalg.norm(points - center, axis=-1).mean())
            scale = float(arrays.bbox_scale[index, time_index])
            if np.isfinite(scale) and scale > 0:
                radial_values.append(spread / scale)
        row["pose_spread_bbox_fraction"] = (
            float(np.mean(radial_values)) if radial_values else np.nan
        )
        distal_selected = selected & distal[None, :]
        central_selected = selected & ~distal[None, :]
        distal_errors = arrays.normalized[index][distal_selected]
        central_errors = arrays.normalized[index][central_selected]
        row["masked_distal_joints"] = int(distal_selected.sum())
        row["masked_non_distal_joints"] = int(central_selected.sum())
        row["distal_nmje"] = (
            float(np.mean(distal_errors)) if distal_errors.size else np.nan
        )
        row["non_distal_nmje"] = (
            float(np.mean(central_errors)) if central_errors.size else np.nan
        )
        for threshold in pck_thresholds:
            key = f"pck_at_{float(threshold):.2f}".replace(".", "_")
            row[key] = float(np.mean(errors <= threshold)) if errors.size else np.nan
        if mask_ratio is not None:
            row["mask_ratio"] = float(mask_ratio)
        for column, values in optional.items():
            if values is not None:
                row[column] = values[index]
        rows.append(row)
    table = pd.DataFrame(rows)
    if "mean_bbox_diagonal" in table and table["mean_bbox_diagonal"].notna().any():
        percentiles = table["mean_bbox_diagonal"].rank(method="average", pct=True)
        table["bbox_diagonal_percentile"] = percentiles
        table["player_scale_bin"] = np.select(
            [percentiles <= 0.25, percentiles >= 0.75],
            ["small", "large"],
            default="medium",
        )
    return table


def build_per_joint_reconstruction_table(
    prediction: Any,
    target: Any,
    *,
    bboxes: Any,
    evaluation_mask: Any,
    original_valid_mask: Any | None = None,
    sequence_mask: Any | None = None,
    joint_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return measured bbox-normalized error for every joint with targets."""

    arrays = reconstruction_error_arrays(
        prediction,
        target,
        bboxes=bboxes,
        evaluation_mask=evaluation_mask,
        original_valid_mask=original_valid_mask,
        sequence_mask=sequence_mask,
    )
    count = arrays.normalized.shape[-1]
    names = (
        [f"joint_{index}" for index in range(count)]
        if joint_names is None
        else [str(value) for value in joint_names]
    )
    if len(names) != count:
        raise ValueError("joint_names must have one entry per joint")
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        selected = arrays.scoring_mask[..., index]
        errors = arrays.normalized[..., index][selected]
        rows.append(
            {
                "joint_index": index,
                "joint": str(name),
                "n_scored": int(errors.size),
                "nmje": float(np.mean(errors)) if errors.size else np.nan,
            }
        )
    return pd.DataFrame(rows)
