"""Leakage-safe TrackID3x3 coordinate and angle representations.

All data-dependent transforms accept an explicit *observed* mask.  This is
important for masked reconstruction: hidden targets must never determine the
centre or scale used to construct model inputs.  Returned offsets/scales make
every coordinate representation reversible to source pixels.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.data.trackid3x3 import TRACKID10_JOINT_NAMES

from .joint_angles import (
    TRACKID10_ANGLE_NAMES,
    TRACKID10_ANGLE_TRIPLETS,
    compute_trackid10_angles,
)

TRACKID_ANGLE_TRIPLETS = TRACKID10_ANGLE_TRIPLETS
TRACKID_ANGLE_NAMES = TRACKID10_ANGLE_NAMES


def _validate_pose(
    keypoints: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(keypoints, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    if points.ndim < 2 or points.shape[-2:] != (len(TRACKID10_JOINT_NAMES), 2):
        raise ValueError(f"keypoints must end in [10,2], got {points.shape}.")
    if mask.shape != points.shape[:-1]:
        raise ValueError(f"valid_mask {mask.shape} does not align with keypoints {points.shape}.")
    return points, mask


def _validate_bboxes(
    bboxes: np.ndarray | None,
    prefix_shape: tuple[int, ...],
    bbox_valid: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray]:
    if bboxes is None:
        return None, np.zeros(prefix_shape, dtype=bool)
    boxes = np.asarray(bboxes, dtype=np.float32)
    if boxes.shape != (*prefix_shape, 4):
        raise ValueError(f"bboxes must have shape {(*prefix_shape, 4)}, got {boxes.shape}.")
    inferred = np.isfinite(boxes).all(axis=-1) & (boxes[..., 2] > 0) & (boxes[..., 3] > 0)
    if bbox_valid is not None:
        explicit = np.asarray(bbox_valid, dtype=bool)
        if explicit.shape != prefix_shape:
            raise ValueError(f"bbox_valid must have shape {prefix_shape}, got {explicit.shape}.")
        inferred &= explicit
    return boxes, inferred


def _validate_bbox_sources(
    bbox_sources: np.ndarray | None, prefix_shape: tuple[int, ...]
) -> np.ndarray:
    if bbox_sources is None:
        return np.full(prefix_shape, "independent_unspecified", dtype="U32")
    sources = np.asarray(bbox_sources).astype(str)
    if sources.shape != prefix_shape:
        raise ValueError(f"bbox_sources must have shape {prefix_shape}, got {sources.shape}.")
    return sources


def _observed_tight_bbox(
    points: np.ndarray, observed: np.ndarray, *, epsilon: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    count = observed.sum(axis=-1)
    lower = np.min(np.where(observed[..., None], points, np.inf), axis=-2)
    upper = np.max(np.where(observed[..., None], points, -np.inf), axis=-2)
    extent = upper - lower
    valid = (
        (count >= 2)
        & np.isfinite(lower).all(axis=-1)
        & np.isfinite(upper).all(axis=-1)
        & (extent > epsilon).all(axis=-1)
    )
    boxes = np.concatenate([lower, extent], axis=-1)
    boxes = np.where(valid[..., None], boxes, np.nan).astype(np.float32)
    return boxes, valid


def _masked_centroid(points: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = mask[..., None].astype(np.float32)
    count = weights.sum(axis=-2)
    valid = count[..., 0] > 0
    center = np.divide(
        (points * weights).sum(axis=-2),
        np.maximum(count, 1.0),
        out=np.zeros(points.shape[:-2] + (2,), dtype=np.float32),
    )
    return center, valid


def observed_root(
    keypoints: np.ndarray,
    observed_mask: np.ndarray,
    *,
    bboxes: np.ndarray | None = None,
    bbox_valid: np.ndarray | None = None,
    bbox_sources: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a centre using observed inputs only.

    Priority: observed centre/hip-midpoint joint; both observed shoulders;
    observed-joint centroid; independent source bbox centre.  The final return
    value is a string array recording which rule was used for every frame.
    """

    points, observed = _validate_pose(keypoints, observed_mask)
    prefix = points.shape[:-2]
    boxes, box_ok = _validate_bboxes(bboxes, prefix, bbox_valid)
    sources = _validate_bbox_sources(bbox_sources, prefix)
    # A pose-tight source bbox contains target coordinates and is not an
    # independent fallback after masking.
    independent_box_ok = box_ok & (sources != "pose_tight_fallback")
    centroid, centroid_ok = _masked_centroid(points, observed)
    root = np.zeros((*prefix, 2), dtype=np.float32)
    valid = np.zeros(prefix, dtype=bool)
    source = np.full(prefix, "unavailable", dtype="U32")

    if boxes is not None:
        value = boxes[..., :2] + boxes[..., 2:] / 2.0
        root = np.where(independent_box_ok[..., None], value, root)
        valid |= independent_box_ok
        source = np.where(independent_box_ok, "bbox_center", source)

    use_centroid = centroid_ok
    root = np.where(use_centroid[..., None], centroid, root)
    valid |= use_centroid
    source = np.where(use_centroid, "observed_centroid", source)

    shoulders_ok = observed[..., 1] & observed[..., 2]
    shoulder_midpoint = (points[..., 1, :] + points[..., 2, :]) / 2.0
    root = np.where(shoulders_ok[..., None], shoulder_midpoint, root)
    valid |= shoulders_ok
    source = np.where(shoulders_ok, "observed_shoulders", source)

    center_ok = observed[..., 7]
    root = np.where(center_ok[..., None], points[..., 7, :], root)
    valid |= center_ok
    source = np.where(center_ok, "observed_center", source)
    return root.astype(np.float32), valid, source


def observed_scale(
    keypoints: np.ndarray,
    observed_mask: np.ndarray,
    root: np.ndarray,
    root_valid: np.ndarray,
    *,
    bboxes: np.ndarray | None = None,
    bbox_valid: np.ndarray | None = None,
    bbox_sources: np.ndarray | None = None,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute an observed-only scalar scale and record its provenance.

    The primary scale follows the paper's torso definition: distance from the
    midpoint of both shoulders to the centre (hip-midpoint) joint.  If any of
    those joints is hidden, the robust median non-zero radial distance of
    observed joints is used.  An independent source bbox diagonal is the final
    fallback.  No hidden target coordinate participates in any rule.
    """

    points, observed = _validate_pose(keypoints, observed_mask)
    prefix = points.shape[:-2]
    centre = np.asarray(root, dtype=np.float32)
    centre_ok = np.asarray(root_valid, dtype=bool)
    if centre.shape != (*prefix, 2) or centre_ok.shape != prefix:
        raise ValueError("root/root_valid do not align with keypoints.")
    boxes, box_ok = _validate_bboxes(bboxes, prefix, bbox_valid)
    sources = _validate_bbox_sources(bbox_sources, prefix)
    independent_box_ok = box_ok & (sources != "pose_tight_fallback")

    scale = np.ones(prefix, dtype=np.float32)
    valid = np.zeros(prefix, dtype=bool)
    source = np.full(prefix, "unavailable", dtype="U32")
    if boxes is not None:
        diagonal = np.linalg.norm(boxes[..., 2:4], axis=-1)
        use_bbox = independent_box_ok & np.isfinite(diagonal) & (diagonal > epsilon)
        scale = np.where(use_bbox, diagonal, scale)
        valid |= use_bbox
        source = np.where(use_bbox, "bbox_diagonal", source)

    distances = np.linalg.norm(points - centre[..., None, :], axis=-1)
    radial_values = np.where(observed & (distances > epsilon), distances, np.nan)
    with np.errstate(all="ignore"):
        robust = np.nanmedian(radial_values, axis=-1)
    robust_ok = centre_ok & np.isfinite(robust) & (robust > epsilon)
    scale = np.where(robust_ok, robust, scale)
    valid |= robust_ok
    source = np.where(robust_ok, "observed_median_radius", source)

    shoulder_midpoint = (points[..., 1, :] + points[..., 2, :]) / 2.0
    torso = np.linalg.norm(shoulder_midpoint - points[..., 7, :], axis=-1)
    torso_ok = (
        observed[..., 1]
        & observed[..., 2]
        & observed[..., 7]
        & np.isfinite(torso)
        & (torso > epsilon)
    )
    scale = np.where(torso_ok, torso, scale)
    valid |= torso_ok
    source = np.where(torso_ok, "observed_torso", source)
    return scale.astype(np.float32), valid, source


def compute_trackid_angles(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Return five explicitly-defined geometric angles in radians."""

    points, mask = _validate_pose(keypoints, valid_mask)
    return compute_trackid10_angles(points, mask, epsilon=1e-8)


def build_observed_representation(
    keypoints: np.ndarray,
    original_valid_mask: np.ndarray,
    observed_mask: np.ndarray,
    representation: str,
    *,
    bboxes: np.ndarray | None = None,
    bbox_valid: np.ndarray | None = None,
    bbox_sources: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build reversible model inputs/targets from an explicit observed mask.

    ``targets`` contains every originally valid joint in the observed-derived
    coordinate system. ``inputs`` zeros everything not observed.  Consumers
    must use the returned masks rather than interpret coordinate zeros.
    """

    points, original = _validate_pose(keypoints, original_valid_mask)
    observed = np.asarray(observed_mask, dtype=bool)
    if observed.shape != original.shape:
        raise ValueError("observed_mask does not align with original_valid_mask.")
    if np.any(observed & ~original):
        raise ValueError("observed_mask may contain only originally valid joints.")
    prefix = points.shape[:-2]
    boxes, box_ok = _validate_bboxes(bboxes, prefix, bbox_valid)
    sources = _validate_bbox_sources(bbox_sources, prefix)
    token = representation.strip().lower().replace("-", "_")
    aliases = {
        "bbox": "bbox_normalized",
        "root": "root_centered",
        "scale": "scale_normalized",
    }
    token = aliases.get(token, token)
    if token not in {"raw", "bbox_normalized", "root_centered", "scale_normalized"}:
        raise ValueError(
            "representation must be raw, bbox_normalized, root_centered, or scale_normalized."
        )

    if token == "raw":
        offset = np.zeros((*prefix, 2), dtype=np.float32)
        scale_xy = np.ones((*prefix, 2), dtype=np.float32)
        transform_valid = np.ones(prefix, dtype=bool)
        center_source = np.full(prefix, "origin", dtype="U32")
        scale_source = np.full(prefix, "unit", dtype="U32")
        transform_bbox = np.full((*prefix, 4), np.nan, dtype=np.float32)
    elif token == "bbox_normalized":
        if boxes is None:
            raise ValueError("bbox_normalized requires source bboxes.")
        fallback = sources == "pose_tight_fallback"
        observed_bbox, observed_bbox_ok = _observed_tight_bbox(points, observed)
        use_source = box_ok & ~fallback
        use_observed = observed_bbox_ok & fallback
        transform_bbox = np.where(
            use_observed[..., None],
            observed_bbox,
            np.where(use_source[..., None], boxes, np.nan),
        ).astype(np.float32)
        transform_valid = use_source | use_observed
        offset = np.where(transform_valid[..., None], transform_bbox[..., :2], 0.0).astype(
            np.float32
        )
        scale_xy = np.where(
            transform_valid[..., None], transform_bbox[..., 2:4], 1.0
        ).astype(np.float32)
        center_source = np.where(
            use_observed,
            "observed_pose_tight",
            np.where(use_source, "source_bbox_origin", "unavailable"),
        ).astype("U32")
        scale_source = np.where(
            use_observed,
            "observed_pose_tight",
            np.where(use_source, "source_bbox_xywh", "unavailable"),
        ).astype("U32")
    else:
        offset, root_ok, center_source = observed_root(
            points,
            observed,
            bboxes=boxes,
            bbox_valid=box_ok,
            bbox_sources=sources,
        )
        transform_bbox = np.where(
            (box_ok & (sources != "pose_tight_fallback"))[..., None], boxes, np.nan
        ).astype(np.float32) if boxes is not None else np.full((*prefix, 4), np.nan, dtype=np.float32)
        if token == "root_centered":
            scale_xy = np.ones((*prefix, 2), dtype=np.float32)
            transform_valid = root_ok
            scale_source = np.full(prefix, "unit", dtype="U32")
        else:
            scalar, scale_ok, scale_source = observed_scale(
                points,
                observed,
                offset,
                root_ok,
                bboxes=boxes,
                bbox_valid=box_ok,
                bbox_sources=sources,
            )
            scale_xy = np.repeat(scalar[..., None], 2, axis=-1)
            transform_valid = root_ok & scale_ok

    represented = (points - offset[..., None, :]) / scale_xy[..., None, :]
    target_mask = original & transform_valid[..., None]
    input_mask = observed & transform_valid[..., None]
    targets = np.where(target_mask[..., None], represented, 0.0).astype(np.float32)
    inputs = np.where(input_mask[..., None], represented, 0.0).astype(np.float32)
    return {
        "inputs": inputs,
        "targets": targets,
        "input_mask": input_mask,
        "target_valid_mask": target_mask,
        "inverse_offset": offset.astype(np.float32),
        "inverse_scale": scale_xy.astype(np.float32),
        "transform_valid": transform_valid,
        "center_source": center_source,
        "scale_source": scale_source,
        "transform_bbox_xywh": transform_bbox,
        "representation": token,
    }


def inverse_trackid_representation(
    coordinates: np.ndarray, inverse_offset: np.ndarray, inverse_scale: np.ndarray
) -> np.ndarray:
    """Map represented coordinates back to the source pixel coordinate system."""

    values = np.asarray(coordinates, dtype=np.float32)
    offset = np.asarray(inverse_offset, dtype=np.float32)
    scale = np.asarray(inverse_scale, dtype=np.float32)
    try:
        return (values * scale[..., None, :] + offset[..., None, :]).astype(np.float32)
    except ValueError as exc:
        raise ValueError("inverse_offset/inverse_scale do not broadcast to coordinates.") from exc


def build_all_representations(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    *,
    bboxes: np.ndarray,
    bbox_valid: np.ndarray | None = None,
    bbox_sources: np.ndarray | None = None,
    observed_mask: np.ndarray | None = None,
) -> Mapping[str, Mapping[str, Any]]:
    """Build the four coordinate variants plus explicit geometric angles."""

    points, valid = _validate_pose(keypoints, valid_mask)
    observed = valid if observed_mask is None else np.asarray(observed_mask, dtype=bool)
    result: dict[str, Mapping[str, Any]] = {}
    for name in ("raw", "bbox_normalized", "root_centered", "scale_normalized"):
        result[name] = build_observed_representation(
            points,
            valid,
            observed,
            name,
            bboxes=bboxes,
            bbox_valid=bbox_valid,
            bbox_sources=bbox_sources,
        )
    angles, angle_mask, names = compute_trackid_angles(
        result["scale_normalized"]["inputs"], result["scale_normalized"]["input_mask"]
    )
    result["angles"] = {"values": angles, "mask": angle_mask, "names": names}
    return result


__all__ = [
    "TRACKID_ANGLE_NAMES",
    "TRACKID_ANGLE_TRIPLETS",
    "build_all_representations",
    "build_observed_representation",
    "compute_trackid_angles",
    "inverse_trackid_representation",
    "observed_root",
    "observed_scale",
]
