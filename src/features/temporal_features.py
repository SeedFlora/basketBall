"""Temporal standardisation and motion features for variable-length poses."""

from __future__ import annotations

import numpy as np


def uniform_positions(source_length: int, target_length: int) -> np.ndarray:
    """Return uniformly spaced floating-point source positions, including ends."""

    if source_length <= 0 or target_length <= 0:
        raise ValueError("source_length and target_length must be positive.")
    if target_length == 1:
        return np.asarray([(source_length - 1) / 2.0], dtype=np.float64)
    return np.linspace(0.0, float(source_length - 1), target_length, dtype=np.float64)


def temporal_resample(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    target_length: int,
    confidence: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Uniformly interpolate a pose sequence to exactly ``target_length`` frames.

    Each joint is interpolated only from its own valid observations.  Two or more
    observations yield linear interpolation between the first/last observation;
    one observation is repeated.  Frames outside a joint's observed temporal range
    remain invalid.  When length already matches, missingness is preserved exactly.
    """

    points = np.asarray(keypoints, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    if points.ndim != 3 or mask.shape != points.shape[:2]:
        raise ValueError(f"Expected keypoints [T,V,C] and mask [T,V], got {points.shape}, {mask.shape}.")
    if target_length <= 0:
        raise ValueError("target_length must be positive.")
    conf: np.ndarray | None = None
    if confidence is not None:
        conf = np.asarray(confidence, dtype=np.float32)
        if conf.shape != mask.shape:
            raise ValueError(f"confidence {conf.shape} does not match mask {mask.shape}.")
    finite = np.isfinite(points).all(axis=-1)
    mask = mask & finite
    if points.shape[0] == target_length:
        output = np.where(mask[..., None], points, 0.0).astype(np.float32, copy=True)
        output_conf = None
        if conf is not None:
            output_conf = np.where(mask & np.isfinite(conf), conf, 0.0).astype(np.float32, copy=True)
        return output, mask.copy(), output_conf

    target_positions = uniform_positions(points.shape[0], target_length)
    output = np.zeros((target_length, points.shape[1], points.shape[2]), dtype=np.float32)
    output_mask = np.zeros((target_length, points.shape[1]), dtype=bool)
    output_conf = (
        np.zeros((target_length, points.shape[1]), dtype=np.float32) if conf is not None else None
    )
    source_positions = np.arange(points.shape[0], dtype=np.float64)
    for joint in range(points.shape[1]):
        valid_indices = np.flatnonzero(mask[:, joint])
        if valid_indices.size == 0:
            continue
        if valid_indices.size == 1:
            output[:, joint] = points[valid_indices[0], joint]
            output_mask[:, joint] = True
            if output_conf is not None and conf is not None:
                output_conf[:, joint] = conf[valid_indices[0], joint]
            continue
        within = (target_positions >= valid_indices[0]) & (target_positions <= valid_indices[-1])
        for coordinate in range(points.shape[2]):
            output[within, joint, coordinate] = np.interp(
                target_positions[within],
                source_positions[valid_indices],
                points[valid_indices, joint, coordinate],
            )
        output_mask[within, joint] = True
        if output_conf is not None and conf is not None:
            output_conf[within, joint] = np.interp(
                target_positions[within],
                source_positions[valid_indices],
                conf[valid_indices, joint],
            )
    output[~output_mask] = 0.0
    if output_conf is not None:
        output_conf[~output_mask] = 0.0
    return output, output_mask, output_conf


def uniform_temporal_resample(
    keypoints: np.ndarray, valid_mask: np.ndarray, target_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper returning coordinates and mask only."""

    points, mask, _ = temporal_resample(keypoints, valid_mask, target_length)
    return points, mask


def compute_motion(
    keypoints: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute first-order displacement, masked across missing observations."""

    points = np.asarray(keypoints, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    if points.ndim != 3 or mask.shape != points.shape[:2]:
        raise ValueError(f"Expected [T,V,C] and [T,V], got {points.shape}, {mask.shape}.")
    motion = np.zeros_like(points, dtype=np.float32)
    motion_mask = np.zeros_like(mask, dtype=bool)
    if points.shape[0] > 1:
        pair_valid = mask[1:] & mask[:-1]
        motion[1:] = np.where(
            pair_valid[..., None], points[1:] - points[:-1], 0.0
        ).astype(np.float32, copy=False)
        motion_mask[1:] = pair_valid
    return motion, motion_mask
