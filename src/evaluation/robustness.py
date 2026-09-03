"""Deterministic missing-keypoint robustness evaluation.

Occlusion is applied to the test representation only; this module never trains
or updates a model.  A missing observation always refers to an entire joint at
one time step (all coordinate channels), avoiding unrealistically independent
coordinate dropout.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .metrics import DEFAULT_CLASS_NAMES, classification_metrics, summary_metrics

SUPPORTED_STRATEGIES: tuple[str, ...] = (
    "zero_mask",
    "temporal_interpolation",
    "joint_masking",
)


def _validate_rate(missing_rate: float) -> float:
    rate = float(missing_rate)
    if not np.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError("missing_rate must be a finite value in [0, 1]")
    return rate


def _canonical(
    sequences: Any,
    *,
    temporal_axis: int,
    joint_axis: int,
    feature_axis: int,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    array = np.asarray(sequences)
    if array.ndim < 3:
        raise ValueError("sequences must have at least temporal, joint, and feature axes")
    requested_axes = (temporal_axis, joint_axis, feature_axis)
    if any(axis < -array.ndim or axis >= array.ndim for axis in requested_axes):
        raise ValueError(f"axis is outside an array with {array.ndim} dimensions")
    axes = tuple(axis % array.ndim for axis in requested_axes)
    if len(set(axes)) != 3:
        raise ValueError("temporal_axis, joint_axis, and feature_axis must be distinct")
    moved = np.moveaxis(array, axes, (-3, -2, -1))
    return moved, axes


def generate_missing_joint_mask(
    sequences: Any,
    missing_rate: float,
    *,
    seed: int = 42,
    whole_joint: bool = False,
    temporal_axis: int = -3,
    joint_axis: int = -2,
    feature_axis: int = -1,
    eligible_mask: Any | None = None,
) -> np.ndarray:
    """Return a deterministic boolean mask in canonical ``[..., T, J]`` order.

    With ``whole_joint=False``, exactly ``round(rate * T * J)`` observations are
    selected independently for every leading sample.  With ``whole_joint=True``,
    exactly ``round(rate * J)`` complete joint trajectories are selected.  This
    makes repeated evaluations and degradation curves reproducible.
    """

    rate = _validate_rate(missing_rate)
    canonical, _ = _canonical(
        sequences,
        temporal_axis=temporal_axis,
        joint_axis=joint_axis,
        feature_axis=feature_axis,
    )
    leading_shape = canonical.shape[:-3]
    time_steps, num_joints = canonical.shape[-3:-1]
    num_samples = int(np.prod(leading_shape)) if leading_shape else 1
    mask = np.zeros((num_samples, time_steps, num_joints), dtype=bool)
    if eligible_mask is None:
        eligible = np.ones_like(mask)
    else:
        eligible_array = np.asarray(eligible_mask, dtype=bool)
        expected = (*leading_shape, time_steps, num_joints)
        if eligible_array.shape == canonical.shape:
            eligible_array = np.all(eligible_array, axis=-1)
        if eligible_array.shape != expected:
            raise ValueError(f"eligible_mask must have shape {expected}, got {eligible_array.shape}")
        eligible = eligible_array.reshape((num_samples, time_steps, num_joints))
    if rate == 0.0:
        return mask.reshape((*leading_shape, time_steps, num_joints))

    rng = np.random.default_rng(int(seed))
    for sample_index in range(num_samples):
        if whole_joint:
            available_joints = np.flatnonzero(np.any(eligible[sample_index], axis=0))
            count = min(
                len(available_joints),
                int(np.floor(rate * len(available_joints) + 0.5)),
            )
            if count:
                indices = rng.choice(available_joints, size=count, replace=False)
                mask[sample_index, :, indices] = eligible[sample_index, :, indices]
        else:
            available = np.flatnonzero(eligible[sample_index].reshape(-1))
            count = min(len(available), int(np.floor(rate * len(available) + 0.5)))
            if count:
                flat_indices = rng.choice(available, size=count, replace=False)
                mask[sample_index].reshape(-1)[flat_indices] = True
    return mask.reshape((*leading_shape, time_steps, num_joints))


def deterministic_missing_mask(*args: Any, **kwargs: Any) -> np.ndarray:
    """Alias for :func:`generate_missing_joint_mask`."""

    return generate_missing_joint_mask(*args, **kwargs)


def temporal_interpolate(*args: Any, **kwargs: Any) -> np.ndarray:
    """Alias for :func:`temporal_interpolate_missing`."""

    return temporal_interpolate_missing(*args, **kwargs)


def temporal_interpolate_missing(
    sequences: Any,
    missing_mask: Any,
    *,
    temporal_axis: int = -3,
    joint_axis: int = -2,
    feature_axis: int = -1,
) -> np.ndarray:
    """Linearly interpolate masked joint coordinates along time.

    End gaps use the nearest valid observation.  If an entire joint trajectory
    is missing, its values are set to zero.  Non-finite source coordinates are
    treated as missing as well.  Inputs are not modified.
    """

    canonical, original_axes = _canonical(
        sequences,
        temporal_axis=temporal_axis,
        joint_axis=joint_axis,
        feature_axis=feature_axis,
    )
    output = canonical.astype(float, copy=True)
    mask = np.asarray(missing_mask, dtype=bool)
    expected_shape = output.shape[:-1]
    if mask.shape == output.shape:
        mask = np.any(mask, axis=-1)
    if mask.shape != expected_shape:
        raise ValueError(
            f"missing_mask must have shape {expected_shape} (or {output.shape}), got {mask.shape}"
        )

    leading_shape = output.shape[:-3]
    time_steps, num_joints, channels = output.shape[-3:]
    flat_output = output.reshape((-1, time_steps, num_joints, channels))
    flat_mask = mask.reshape((-1, time_steps, num_joints))
    time_index = np.arange(time_steps, dtype=float)
    for sample_index in range(flat_output.shape[0]):
        for joint_index in range(num_joints):
            requested_missing = flat_mask[sample_index, :, joint_index]
            for channel_index in range(channels):
                values = flat_output[sample_index, :, joint_index, channel_index]
                missing = requested_missing | ~np.isfinite(values)
                if not np.any(missing):
                    continue
                valid = ~missing
                if not np.any(valid):
                    values[:] = 0.0
                else:
                    values[missing] = np.interp(
                        time_index[missing], time_index[valid], values[valid]
                    )

    restored = flat_output.reshape((*leading_shape, time_steps, num_joints, channels))
    return np.moveaxis(restored, (-3, -2, -1), original_axes)


def apply_missing_joints(
    sequences: Any,
    missing_rate: float,
    *,
    strategy: str = "zero_mask",
    seed: int = 42,
    confidence_channel: int | None = None,
    temporal_axis: int = -3,
    joint_axis: int = -2,
    feature_axis: int = -1,
    eligible_mask: Any | None = None,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Apply a deterministic test-time missing-joint strategy.

    ``joint_masking`` masks whole joint trajectories.  It intentionally keeps
    the same input shape as the trained model; a model with a separate validity
    mask may consume the returned mask via ``return_mask=True``.
    """

    aliases = {
        "zero": "zero_mask",
        "interpolation": "temporal_interpolation",
        "temporal": "temporal_interpolation",
        "joint": "joint_masking",
    }
    strategy = aliases.get(strategy, strategy)
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"strategy must be one of {SUPPORTED_STRATEGIES}, got {strategy!r}")

    canonical, original_axes = _canonical(
        sequences,
        temporal_axis=temporal_axis,
        joint_axis=joint_axis,
        feature_axis=feature_axis,
    )
    mask = generate_missing_joint_mask(
        sequences,
        missing_rate,
        seed=seed,
        whole_joint=strategy == "joint_masking",
        temporal_axis=temporal_axis,
        joint_axis=joint_axis,
        feature_axis=feature_axis,
        eligible_mask=eligible_mask,
    )

    if strategy == "temporal_interpolation":
        result = temporal_interpolate_missing(
            sequences,
            mask,
            temporal_axis=temporal_axis,
            joint_axis=joint_axis,
            feature_axis=feature_axis,
        )
        result_canonical, _ = _canonical(
            result,
            temporal_axis=temporal_axis,
            joint_axis=joint_axis,
            feature_axis=feature_axis,
        )
    else:
        result_canonical = canonical.copy()
        result_canonical[mask] = 0

    if confidence_channel is not None:
        channel_count = result_canonical.shape[-1]
        if confidence_channel < -channel_count or confidence_channel >= channel_count:
            raise ValueError("confidence_channel is outside the feature axis")
        channel = confidence_channel % channel_count
        confidence = result_canonical[..., channel]
        confidence[mask] = 0

    if strategy == "temporal_interpolation":
        output = np.moveaxis(result_canonical, (-3, -2, -1), original_axes)
    else:
        output = np.moveaxis(result_canonical, (-3, -2, -1), original_axes)
    return (output, mask) if return_mask else output


def simulate_missing_joints(*args: Any, **kwargs: Any) -> Any:
    """Alias for :func:`apply_missing_joints`."""

    return apply_missing_joints(*args, **kwargs)


def _numpy_prediction(prediction: Any) -> np.ndarray:
    if isinstance(prediction, Mapping):
        for key in ("predictions", "prediction", "logits", "output"):
            if key in prediction:
                prediction = prediction[key]
                break
    if isinstance(prediction, (tuple, list)):
        prediction = prediction[0]
    if hasattr(prediction, "detach"):
        prediction = prediction.detach()
    if hasattr(prediction, "cpu"):
        prediction = prediction.cpu()
    return np.asarray(prediction)


def evaluate_robustness(
    predict_fn: Callable[[np.ndarray], Any],
    sequences: Any,
    y_true: Any,
    *,
    missing_rates: Sequence[float] = (0.0, 0.1, 0.2, 0.3),
    strategies: Sequence[str] = SUPPORTED_STRATEGIES,
    seed: int = 42,
    labels: Sequence[int] | None = None,
    class_names: Mapping[int, str] | Sequence[str] | None = DEFAULT_CLASS_NAMES,
    confidence_channel: int | None = None,
) -> pd.DataFrame:
    """Evaluate a frozen predictor under deterministic test-time occlusion."""

    array = np.asarray(sequences)
    truth = np.asarray(y_true)
    if array.shape[0] != truth.shape[0]:
        raise ValueError("sequences and y_true must have the same first dimension")

    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        for rate in missing_rates:
            perturbed, mask = apply_missing_joints(
                array,
                rate,
                strategy=strategy,
                seed=seed,
                confidence_channel=confidence_channel,
                return_mask=True,
            )
            prediction = _numpy_prediction(predict_fn(perturbed))
            metrics = classification_metrics(
                truth,
                prediction,
                labels=labels,
                class_names=class_names,
            )
            row: dict[str, Any] = {
                "strategy": strategy,
                "missing_rate": float(rate),
                "realized_missing_rate": float(np.mean(mask)) if mask.size else 0.0,
                "seed": int(seed),
                "n_samples": int(metrics["n_samples"]),
            }
            row.update(summary_metrics(metrics))
            rows.append(row)
    return pd.DataFrame(rows)
