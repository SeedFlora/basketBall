"""Leakage-safe runtime representations for masked reconstruction.

The canonical NumPy geometry lives in :mod:`src.features.trackid_features`.
This module provides its tensor-facing reconstruction adapter: hide targets in
pixels first, build transforms from the surviving observations, retain inverse
context, and expose the five supported angles as masked node auxiliaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.features.joint_angles import (
    TRACKID10_ANGLE_NAMES,
    TRACKID10_ANGLE_TRIPLETS,
    compute_trackid10_angles,
)
from src.features.trackid_features import build_observed_representation

from .masking import apply_reconstruction_mask

RECONSTRUCTION_REPRESENTATIONS: tuple[str, ...] = (
    "raw",
    "bbox_normalized",
    "root_centered",
    "scale_normalized",
    "normalized_angles",
)

# Public index-only view retained for callers that do not need semantic names.
TRACKID3X3_ANGLE_TRIPLETS: tuple[tuple[int, int, int], ...] = tuple(
    (first, vertex, third)
    for _, first, vertex, third in TRACKID10_ANGLE_TRIPLETS
)
TRACKID_ANGLE_NAMES = TRACKID10_ANGLE_NAMES


def canonical_representation_name(name: str) -> str:
    token = str(name).strip().lower().replace("-", "_").replace("+", "_")
    aliases = {
        "normalized": "bbox_normalized",
        "normalised": "bbox_normalized",
        "bbox": "bbox_normalized",
        "root": "root_centered",
        "scale": "scale_normalized",
        "scale_normalized_angles": "normalized_angles",
    }
    token = aliases.get(token, token)
    if token not in RECONSTRUCTION_REPRESENTATIONS:
        raise ValueError(
            f"Unknown representation {name!r}; choose {', '.join(RECONSTRUCTION_REPRESENTATIONS)}"
        )
    return token


def bbox_diagonal(bbox_xywh: Tensor) -> Tensor:
    boxes = torch.as_tensor(bbox_xywh, dtype=torch.float32)
    if boxes.shape[-1] != 4:
        raise ValueError(f"bbox_xywh must end in 4 values, got {tuple(boxes.shape)}")
    return torch.linalg.vector_norm(boxes[..., 2:4], dim=-1)


def _tensor(value: np.ndarray, reference: Tensor, *, boolean: bool = False) -> Tensor:
    dtype = torch.bool if boolean else reference.dtype
    return torch.as_tensor(value, dtype=dtype, device=reference.device)


def prepare_reconstruction_representation(
    pixel_coordinates: Tensor,
    original_valid_mask: Tensor,
    synthetic_mask: Tensor,
    *,
    representation: str = "bbox_normalized",
    bbox_xywh: Tensor | None = None,
    bbox_valid: Tensor | None = None,
    bbox_source: Any | None = None,
    sequence_mask: Tensor | None = None,
) -> dict[str, Tensor | str | list[str]]:
    """Mask pixels first, then build one reversible observed-only representation.

    ``normalized_angles`` uses scale-normalized 2-D coordinates as the predicted
    quantity and supplies the five documented angles as one explicitly-masked
    auxiliary value on their vertex nodes.  Output coordinates therefore remain
    ``[...,10,2]`` for every representation.
    """

    name = canonical_representation_name(representation)
    pixels = torch.as_tensor(pixel_coordinates, dtype=torch.float32)
    original = torch.as_tensor(original_valid_mask, dtype=torch.bool, device=pixels.device)
    synthetic = torch.as_tensor(synthetic_mask, dtype=torch.bool, device=pixels.device)
    single = pixels.ndim == 3
    if single:
        pixels = pixels.unsqueeze(0)
        original = original.unsqueeze(0)
        synthetic = synthetic.unsqueeze(0)
        if bbox_xywh is not None:
            bbox_xywh = torch.as_tensor(bbox_xywh).unsqueeze(0)
        if bbox_valid is not None:
            bbox_valid = torch.as_tensor(bbox_valid).unsqueeze(0)
        if bbox_source is not None:
            bbox_source = np.expand_dims(np.asarray(bbox_source), axis=0)
        if sequence_mask is not None:
            sequence_mask = torch.as_tensor(sequence_mask).unsqueeze(0)
    if pixels.ndim != 4 or pixels.shape[-2:] != (10, 2):
        raise ValueError(f"pixel_coordinates must be [B,T,10,2], got {tuple(pixels.shape)}")
    if original.shape != pixels.shape[:-1] or synthetic.shape != original.shape:
        raise ValueError("validity masks do not align with pixel coordinates")
    if sequence_mask is None:
        temporal_valid = torch.ones(pixels.shape[:2], dtype=torch.bool, device=pixels.device)
    else:
        temporal_valid = torch.as_tensor(sequence_mask, dtype=torch.bool, device=pixels.device)
        if temporal_valid.shape != pixels.shape[:2]:
            raise ValueError("sequence_mask must be [B,T]")
        original = original & temporal_valid.unsqueeze(-1)
    if bool((synthetic & ~original).any()):
        raise ValueError("synthetic_mask may select only originally valid keypoints")

    _pixel_inputs, observed = apply_reconstruction_mask(pixels, original, synthetic)
    boxes: Tensor | None = None
    boxes_ok: Tensor | None = None
    source_labels: np.ndarray | None = None
    if bbox_xywh is not None:
        boxes = torch.as_tensor(bbox_xywh, dtype=pixels.dtype, device=pixels.device)
        if boxes.shape != (*pixels.shape[:2], 4):
            raise ValueError(f"bbox_xywh must be [B,T,4], got {tuple(boxes.shape)}")
        inferred = torch.isfinite(boxes).all(dim=-1) & (boxes[..., 2] > 0) & (boxes[..., 3] > 0)
        if bbox_valid is None:
            boxes_ok = inferred
        else:
            boxes_ok = torch.as_tensor(bbox_valid, dtype=torch.bool, device=pixels.device)
            if boxes_ok.shape != pixels.shape[:2]:
                raise ValueError("bbox_valid must be [B,T]")
            boxes_ok = boxes_ok & inferred
    if bbox_source is not None:
        source_labels = np.asarray(bbox_source).astype(str)
        if source_labels.shape != pixels.shape[:2]:
            raise ValueError(
                f"bbox_source must be [B,T]={tuple(pixels.shape[:2])}, "
                f"got {source_labels.shape}"
            )

    base_name = "scale_normalized" if name == "normalized_angles" else name
    points_np = pixels.detach().cpu().numpy()
    original_np = original.detach().cpu().numpy()
    observed_np = observed.detach().cpu().numpy()
    boxes_np = None if boxes is None else boxes.detach().cpu().numpy()
    boxes_ok_np = None if boxes_ok is None else boxes_ok.detach().cpu().numpy()
    built: dict[str, Any] = build_observed_representation(
        points_np,
        original_np,
        observed_np,
        base_name,
        bboxes=boxes_np,
        bbox_valid=boxes_ok_np,
        bbox_sources=source_labels,
    )
    target_valid = _tensor(built["target_valid_mask"], pixels, boolean=True)
    input_valid = _tensor(built["input_mask"], pixels, boolean=True)
    if bool((synthetic & ~target_valid).any()):
        raise ValueError(
            "A synthetically masked target has no valid observed-only transform context"
        )
    if not torch.equal(input_valid, observed):
        raise ValueError(
            "An observed joint has no valid transform; check bbox/pose provenance"
        )

    inputs = _tensor(built["inputs"], pixels)
    targets = _tensor(built["targets"], pixels)
    offset = _tensor(built["inverse_offset"], pixels).unsqueeze(-2)
    scale = _tensor(built["inverse_scale"], pixels).unsqueeze(-2)
    if name == "normalized_angles":
        angles_np, angles_mask_np, angle_names = compute_trackid10_angles(
            built["inputs"], built["input_mask"]
        )
        auxiliary = torch.zeros(
            (*pixels.shape[:-1], 1), dtype=pixels.dtype, device=pixels.device
        )
        auxiliary_mask = torch.zeros_like(auxiliary, dtype=torch.bool)
        for output_index, (_, _, vertex, _) in enumerate(TRACKID10_ANGLE_TRIPLETS):
            auxiliary[..., vertex, 0] = _tensor(angles_np[..., output_index], pixels)
            auxiliary_mask[..., vertex, 0] = _tensor(
                angles_mask_np[..., output_index], pixels, boolean=True
            )
        names = list(angle_names)
    else:
        auxiliary = torch.empty((*pixels.shape[:-1], 0), dtype=pixels.dtype, device=pixels.device)
        auxiliary_mask = torch.empty_like(auxiliary, dtype=torch.bool)
        names = []

    if boxes is None:
        diagonal = torch.full(
            pixels.shape[:2], float("nan"), dtype=pixels.dtype, device=pixels.device
        )
    else:
        diagonal = bbox_diagonal(boxes).to(pixels)
        if boxes_ok is not None:
            diagonal = torch.where(boxes_ok, diagonal, torch.full_like(diagonal, float("nan")))
    result: dict[str, Tensor | str | list[str]] = {
        "inputs": inputs,
        "targets": targets,
        "pixel_targets": pixels,
        "original_valid_mask": original,
        "synthetic_mask": synthetic,
        "observed_mask": observed,
        "auxiliary": auxiliary,
        "auxiliary_mask": auxiliary_mask,
        "inverse_offset": offset,
        "inverse_scale": scale,
        "bbox_diagonal": diagonal,
        "transform_bbox_xywh": _tensor(built["transform_bbox_xywh"], pixels),
        "transform_valid": _tensor(built["transform_valid"], pixels, boolean=True),
        "pose_tight_metric_only": _tensor(
            np.zeros(pixels.shape[:2], dtype=bool)
            if source_labels is None
            else source_labels == "pose_tight_fallback",
            pixels,
            boolean=True,
        ),
        "track_valid_mask": temporal_valid,
        "representation": name,
        "angle_names": names,
        "transform_center_source": np.asarray(built["center_source"]).astype(str).tolist(),
        "transform_scale_source": np.asarray(built["scale_source"]).astype(str).tolist(),
        "bbox_metric_source": (
            np.full(pixels.shape[:2], "unspecified", dtype="U32")
            if source_labels is None
            else source_labels
        ).tolist(),
    }
    if single:
        result = {
            key: (
                value[0]
                if isinstance(value, Tensor)
                else value[0]
                if key
                in {
                    "transform_center_source",
                    "transform_scale_source",
                    "bbox_metric_source",
                }
                else value
            )
            for key, value in result.items()
        }
    return result


def inverse_reconstruction_coordinates(
    represented_coordinates: Tensor,
    inverse_offset: Tensor,
    inverse_scale: Tensor,
) -> Tensor:
    """Map represented predictions back to source pixel coordinates."""

    coordinates = torch.as_tensor(represented_coordinates)
    offset = torch.as_tensor(inverse_offset, dtype=coordinates.dtype, device=coordinates.device)
    scale = torch.as_tensor(inverse_scale, dtype=coordinates.dtype, device=coordinates.device)
    if offset.shape == coordinates.shape[:-2] + (2,):
        offset = offset.unsqueeze(-2)
    if scale.shape == coordinates.shape[:-2] + (2,):
        scale = scale.unsqueeze(-2)
    try:
        return coordinates * scale + offset
    except RuntimeError as exc:
        raise ValueError("inverse transform does not broadcast to represented coordinates") from exc


__all__ = [
    "RECONSTRUCTION_REPRESENTATIONS",
    "TRACKID3X3_ANGLE_TRIPLETS",
    "TRACKID_ANGLE_NAMES",
    "bbox_diagonal",
    "canonical_representation_name",
    "inverse_reconstruction_coordinates",
    "prepare_reconstruction_representation",
]
