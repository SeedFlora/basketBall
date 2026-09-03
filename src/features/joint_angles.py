"""Masked anatomical joint-angle features for known skeleton layouts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_ANGLE_TRIPLES = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
}

# Exact TrackID3x3 source indices: (name, first endpoint, vertex, third
# endpoint).  Only the ten documented points are used; no hip or knee is
# fabricated for this reduced layout.
TRACKID10_ANGLE_TRIPLETS: tuple[tuple[str, int, int, int], ...] = (
    ("left_shoulder", 3, 1, 7),
    ("right_shoulder", 4, 2, 7),
    ("left_elbow", 1, 3, 5),
    ("right_elbow", 2, 4, 6),
    ("shoulder_center", 1, 7, 2),
)
TRACKID10_ANGLE_NAMES = tuple(item[0] for item in TRACKID10_ANGLE_TRIPLETS)


def angle_at_joint(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute angle A-B-C in radians and return a non-degenerate mask."""

    a = np.asarray(point_a, dtype=np.float32)
    b = np.asarray(point_b, dtype=np.float32)
    c = np.asarray(point_c, dtype=np.float32)
    if a.shape != b.shape or b.shape != c.shape or a.shape[-1] < 2:
        raise ValueError(f"Angle points must share shape [...,C>=2], got {a.shape}, {b.shape}, {c.shape}.")
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba, axis=-1)
    norm_bc = np.linalg.norm(bc, axis=-1)
    denominator = norm_ba * norm_bc
    valid = np.isfinite(denominator) & (denominator > epsilon)
    dot = np.sum(ba * bc, axis=-1)
    cosine = np.zeros_like(dot, dtype=np.float32)
    np.divide(dot, denominator, out=cosine, where=valid)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.arccos(cosine).astype(np.float32)
    angle = np.where(valid, angle, 0.0).astype(np.float32, copy=False)
    return angle, valid


def compute_joint_angles(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    joint_names: Sequence[str] | np.ndarray,
    *,
    coordinate_dims: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute eight limb angles where the supplied joint names support them.

    Unknown layouts safely return zero-width ``[T,0]`` arrays and an empty name
    list.  This makes layout capability inspectable without indexing guessed
    joints.  Callers that require angles can then raise a contextual error.
    """

    points = np.asarray(keypoints, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    names = [str(item).lower() for item in np.asarray(joint_names).tolist()]
    if points.ndim != 3 or mask.shape != points.shape[:2]:
        raise ValueError(f"Expected [T,V,C] and [T,V], got {points.shape}, {mask.shape}.")
    if len(names) != points.shape[1]:
        raise ValueError(f"joint_names length {len(names)} does not match V={points.shape[1]}.")
    dims = points.shape[-1] if coordinate_dims is None else int(coordinate_dims)
    if dims < 2 or dims > points.shape[-1]:
        raise ValueError(f"coordinate_dims must be in [2,{points.shape[-1]}], got {dims}.")
    index = {name: position for position, name in enumerate(names)}
    available = [
        (angle_name, triple)
        for angle_name, triple in _ANGLE_TRIPLES.items()
        if all(joint in index for joint in triple)
    ]
    if not available:
        return (
            np.empty((points.shape[0], 0), dtype=np.float32),
            np.empty((points.shape[0], 0), dtype=bool),
            [],
        )
    angles = np.zeros((points.shape[0], len(available)), dtype=np.float32)
    angles_mask = np.zeros((points.shape[0], len(available)), dtype=bool)
    angle_names: list[str] = []
    for column, (angle_name, triple) in enumerate(available):
        a, b, c = (index[name] for name in triple)
        values, geometric_valid = angle_at_joint(
            points[:, a, :dims], points[:, b, :dims], points[:, c, :dims]
        )
        valid = mask[:, a] & mask[:, b] & mask[:, c] & geometric_valid
        angles[:, column] = np.where(valid, values, 0.0)
        angles_mask[:, column] = valid
        angle_names.append(angle_name)
    return angles, angles_mask, angle_names


def compute_trackid10_angles(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Compute the five stable angles supported by the TrackID3x3 layout.

    Inputs may have any leading dimensions and must end in ``[10,2]`` and
    ``[10]``. Degenerate/non-finite vectors are explicitly masked. The shared
    :func:`angle_at_joint` primitive clamps cosine before ``arccos``.
    """

    points = np.asarray(keypoints, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    if points.ndim < 2 or points.shape[-2:] != (10, 2):
        raise ValueError(f"TrackID3x3 keypoints must end in [10,2], got {points.shape}.")
    if mask.shape != points.shape[:-1]:
        raise ValueError(f"TrackID3x3 mask {mask.shape} does not align with {points.shape}.")
    output_shape = points.shape[:-2] + (len(TRACKID10_ANGLE_TRIPLETS),)
    angles = np.zeros(output_shape, dtype=np.float32)
    angle_mask = np.zeros(output_shape, dtype=bool)
    for output_index, (_, first, vertex, third) in enumerate(TRACKID10_ANGLE_TRIPLETS):
        values, geometric_valid = angle_at_joint(
            points[..., first, :],
            points[..., vertex, :],
            points[..., third, :],
            epsilon=epsilon,
        )
        valid = (
            mask[..., first]
            & mask[..., vertex]
            & mask[..., third]
            & geometric_valid
        )
        angles[..., output_index] = np.where(valid, values, 0.0)
        angle_mask[..., output_index] = valid
    return angles, angle_mask, TRACKID10_ANGLE_NAMES


def joint_angles(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    joint_names: Sequence[str] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Alias for :func:`compute_joint_angles`."""

    return compute_joint_angles(keypoints, valid_mask, joint_names)
