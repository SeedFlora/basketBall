"""Translation and scale normalisation for pose sequences."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _joint_index(names: Sequence[str], name: str) -> int | None:
    try:
        return [str(item).lower() for item in names].index(name)
    except ValueError:
        return None


def _pair_center(
    frame: np.ndarray, valid: np.ndarray, left: int | None, right: int | None
) -> np.ndarray | None:
    indices = [index for index in (left, right) if index is not None and valid[index]]
    if not indices:
        return None
    return frame[indices].mean(axis=0)


def normalize_pose(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    joint_names: Sequence[str] | np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Centre and scale a ``[T,V,C]`` skeleton with anatomical fallbacks.

    Centre priority is mid-hip, shoulder midpoint, then the mean of valid joints.
    Scale priority is shoulder width, hip-to-shoulder torso length, then RMS joint
    distance from the centre.  A sequence median fills frames whose preferred
    scale is unavailable.  Invalid joints remain exactly zero and are never used
    in statistics.
    """

    points = np.asarray(keypoints, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool).copy()
    names = [str(item).lower() for item in np.asarray(joint_names).tolist()]
    if points.ndim != 3 or mask.shape != points.shape[:2]:
        raise ValueError(f"Expected [T,V,C] and [T,V], got {points.shape}, {mask.shape}.")
    if len(names) != points.shape[1]:
        raise ValueError(f"joint_names length {len(names)} does not match V={points.shape[1]}.")
    finite = np.isfinite(points).all(axis=-1)
    mask &= finite
    clean = np.where(mask[..., None], points, 0.0).astype(np.float32, copy=False)

    left_hip = _joint_index(names, "left_hip")
    right_hip = _joint_index(names, "right_hip")
    left_shoulder = _joint_index(names, "left_shoulder")
    right_shoulder = _joint_index(names, "right_shoulder")

    centers = np.zeros((points.shape[0], points.shape[2]), dtype=np.float32)
    scales = np.full(points.shape[0], np.nan, dtype=np.float32)
    for time in range(points.shape[0]):
        frame = clean[time]
        valid = mask[time]
        hip_center = _pair_center(frame, valid, left_hip, right_hip)
        shoulder_center = _pair_center(frame, valid, left_shoulder, right_shoulder)
        center = hip_center if hip_center is not None else shoulder_center
        if center is None and np.any(valid):
            center = frame[valid].mean(axis=0)
        if center is None:
            center = np.zeros(points.shape[2], dtype=np.float32)
        centers[time] = center

        scale: float | None = None
        if (
            left_shoulder is not None
            and right_shoulder is not None
            and valid[left_shoulder]
            and valid[right_shoulder]
        ):
            candidate = float(np.linalg.norm(frame[left_shoulder] - frame[right_shoulder]))
            if candidate > epsilon:
                scale = candidate
        if scale is None and hip_center is not None and shoulder_center is not None:
            candidate = float(np.linalg.norm(shoulder_center - hip_center))
            if candidate > epsilon:
                scale = candidate
        if scale is None and np.any(valid):
            distances = np.linalg.norm(frame[valid] - center, axis=-1)
            candidate = float(np.sqrt(np.mean(np.square(distances))))
            if candidate > epsilon:
                scale = candidate
        if scale is not None:
            scales[time] = scale

    usable_scales = scales[np.isfinite(scales) & (scales > epsilon)]
    fallback = float(np.median(usable_scales)) if usable_scales.size else 1.0
    scales = np.where(np.isfinite(scales) & (scales > epsilon), scales, fallback)
    normalized = (clean - centers[:, None, :]) / scales[:, None, None]
    normalized = np.where(mask[..., None], normalized, 0.0).astype(np.float32, copy=False)
    return normalized, mask


def translation_normalize(
    keypoints: np.ndarray, valid_mask: np.ndarray, joint_names: Sequence[str] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Centre a skeleton without scale normalisation (ablation helper)."""

    points = np.asarray(keypoints, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    _, normalized_mask = normalize_pose(points, mask, joint_names)
    # Recover only translation centring by matching the original per-frame spread.
    # Implement directly to avoid relying on unavailable scale metadata.
    names = [str(item).lower() for item in np.asarray(joint_names).tolist()]
    left_hip = _joint_index(names, "left_hip")
    right_hip = _joint_index(names, "right_hip")
    left_shoulder = _joint_index(names, "left_shoulder")
    right_shoulder = _joint_index(names, "right_shoulder")
    output = np.zeros_like(points)
    for time in range(points.shape[0]):
        valid = normalized_mask[time]
        hip = _pair_center(points[time], valid, left_hip, right_hip)
        shoulder = _pair_center(points[time], valid, left_shoulder, right_shoulder)
        center = hip if hip is not None else shoulder
        if center is None and np.any(valid):
            center = points[time, valid].mean(axis=0)
        if center is not None:
            output[time, valid] = points[time, valid] - center
    return output.astype(np.float32, copy=False), normalized_mask
