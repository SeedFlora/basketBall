#!/usr/bin/env python
"""Evaluate masked 10-joint TrackID3x3 pose reconstruction in source pixels.

The public workflow is reconstruction-only.  Synthetic masks are deterministic,
naturally absent joints are never scored, and every representation is inverted
before bbox-diagonal NMJE and PCK are computed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.error_analysis import (
    build_per_joint_reconstruction_table,
    build_reconstruction_error_table,
)
from src.evaluation.metrics import reconstruction_metrics
from src.models import count_parameters, reconstruction_baseline
from src.training import (
    SUPPORTED_MASK_RATIOS,
    canonical_representation_name,
    deterministic_reconstruction_mask,
    inverse_reconstruction_coordinates,
    load_reconstruction_checkpoint,
    prepare_reconstruction_representation,
)
from src.visualization.reconstruction import (
    missingness_by_domain_joint,
    plot_coordinate_distributions,
    plot_domain_summary,
    plot_missingness_heatmap,
    plot_pck_curves,
    plot_per_joint_error,
    plot_pose_instance_counts,
    plot_reconstruction_example,
)
from src.visualization.skeleton_plot import plot_skeleton

TITLE = "Cross-Domain Robustness of Lightweight Skeleton Representations for 3x3 Basketball Player Pose Analysis"
DOMAINS = ("Indoor", "Outdoor", "Drone")
COORDINATE_KEYS = ("coords", "keypoints", "coordinates")
BBOX_KEYS = ("bbox_xywh", "bboxes", "bboxes_xywh", "bbox")
DOMAIN_KEYS = ("domains", "domain", "capture_domain")
VIDEO_KEYS = ("video_ids", "video_id", "videos")
TRACK_KEYS = ("track_ids", "track_id", "player_ids")
SEQUENCE_MASK_KEYS = ("sequence_mask", "track_valid_mask", "window_valid_mask")
UNAVAILABLE_TOKENS = {"", "none", "nan", "n/a", "na", "unknown", "unavailable", "missing"}


class ReconstructionEvaluationError(RuntimeError):
    """Raised when an artifact cannot support scientifically valid evaluation."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _nested(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _text(values: Any) -> np.ndarray:
    array = np.asarray(values)
    decoded = [
        item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in array.reshape(-1).tolist()
    ]
    return np.asarray(decoded, dtype=str).reshape(array.shape)


def _first_array(data: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray | None:
    for name in names:
        if name in data:
            return np.asarray(data[name])
    return None


def _normalise_edges(value: Any, num_joints: int) -> list[tuple[int, int]]:
    edges = np.asarray(value, dtype=np.int64)
    if edges.ndim != 2 or 2 not in edges.shape:
        raise ReconstructionEvaluationError(f"edges must be [E,2] or [2,E], got {edges.shape}")
    if edges.shape[1] != 2:
        edges = edges.T
    pairs = [(int(source), int(target)) for source, target in edges.tolist()]
    if any(source < 0 or target < 0 or source >= num_joints or target >= num_joints for source, target in pairs):
        raise ReconstructionEvaluationError("an edge references a joint outside the 10-joint layout")
    return pairs


def _metadata_for(source: Path) -> dict[str, Any]:
    path = source.with_name("metadata.json")
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionEvaluationError(f"invalid processed metadata {path}: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


def _looks_like_reconstruction_npz(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            return (
                bool(names.intersection(COORDINATE_KEYS))
                and "valid_mask" in names
                and bool(names.intersection(BBOX_KEYS))
                and bool(names.intersection(DOMAIN_KEYS))
            )
    except (OSError, ValueError):
        return False


def resolve_processed(path: str | Path | None, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve only a TrackID-shaped processed artifact, never the legacy classifier data."""

    if path is not None:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        candidate = candidate / "dataset.npz" if candidate.is_dir() else candidate
        if not candidate.is_file() or not _looks_like_reconstruction_npz(candidate):
            raise ReconstructionEvaluationError(
                f"{candidate} is not a processed TrackID reconstruction NPZ (coords, mask, bbox, domain required)"
            )
        return candidate.resolve()
    candidates = [
        project_root / "data" / "processed" / "trackid3x3.npz",
        project_root / "data" / "processed" / "dataset.npz",
        *sorted((project_root / "data" / "processed").glob("**/*.npz")),
    ]
    valid = {candidate.resolve() for candidate in candidates if candidate.is_file() and _looks_like_reconstruction_npz(candidate)}
    if not valid:
        raise ReconstructionEvaluationError(
            "No TrackID reconstruction NPZ found. Run the TrackID preprocessing workflow first."
        )
    return max(valid, key=lambda item: item.stat().st_mtime)


def load_reconstruction_data(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and validate flexible aliases of the documented TrackID NPZ contract."""

    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            raw = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise ReconstructionEvaluationError(f"could not safely load {source}: {exc}") from exc
    coordinates = _first_array(raw, COORDINATE_KEYS)
    valid = _first_array(raw, ("valid_mask", "original_valid_mask", "keypoint_mask"))
    bboxes = _first_array(raw, BBOX_KEYS)
    domains = _first_array(raw, DOMAIN_KEYS)
    if coordinates is None or valid is None or bboxes is None or domains is None:
        raise ReconstructionEvaluationError("processed data require coordinates, valid_mask, bbox_xywh, and domains")
    coordinates = np.asarray(coordinates, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    bboxes = np.asarray(bboxes, dtype=np.float32)
    if coordinates.ndim != 4 or coordinates.shape[-2:] != (10, 2):
        raise ReconstructionEvaluationError(f"coordinates must be [N,T,10,2], got {coordinates.shape}")
    if valid.shape != coordinates.shape[:-1]:
        raise ReconstructionEvaluationError("valid_mask must align with coordinates [N,T,10]")
    if bboxes.shape != (*coordinates.shape[:2], 4):
        raise ReconstructionEvaluationError("bbox_xywh must have shape [N,T,4]")
    n_samples, time_steps = coordinates.shape[:2]
    domains = _text(domains).reshape(-1)
    if domains.shape != (n_samples,):
        raise ReconstructionEvaluationError("domains must have shape [N]")
    sequence = _first_array(raw, SEQUENCE_MASK_KEYS)
    sequence = np.ones((n_samples, time_steps), dtype=bool) if sequence is None else np.asarray(sequence, dtype=bool)
    if sequence.shape != (n_samples, time_steps):
        raise ReconstructionEvaluationError("sequence_mask must have shape [N,T]")
    valid &= sequence[..., None]
    if not np.isfinite(coordinates[valid]).all():
        raise ReconstructionEvaluationError("valid source keypoints contain NaN or infinity")
    scored_frames = valid.any(axis=-1)
    inferred_valid_boxes = np.isfinite(bboxes).all(axis=-1) & (bboxes[..., 2] > 0) & (bboxes[..., 3] > 0)
    bbox_valid_raw = _first_array(raw, ("bbox_valid", "bboxes_valid"))
    valid_boxes = (
        inferred_valid_boxes
        if bbox_valid_raw is None
        else np.asarray(bbox_valid_raw, dtype=bool) & inferred_valid_boxes
    )
    if valid_boxes.shape != (n_samples, time_steps):
        raise ReconstructionEvaluationError("bbox_valid must have shape [N,T]")
    if np.any(scored_frames & ~valid_boxes):
        raise ReconstructionEvaluationError("a frame with annotated keypoints lacks a valid independent bbox")
    bbox_source_raw = _first_array(raw, ("bbox_source", "bbox_sources"))
    if bbox_source_raw is None:
        if np.any(np.char.lower(domains.astype(str)) == "drone"):
            raise ReconstructionEvaluationError(
                "Drone evaluation requires bbox_source so pose-tight boxes cannot leak into model inputs"
            )
        bbox_source = np.full((n_samples, time_steps), "mot_xywh", dtype=str)
    else:
        bbox_source = _text(bbox_source_raw).astype(str)
        if bbox_source.shape != (n_samples, time_steps):
            raise ReconstructionEvaluationError("bbox_source must have shape [N,T]")

    sample_ids = _first_array(raw, ("sample_ids", "sample_id", "window_ids"))
    if sample_ids is None:
        raise ReconstructionEvaluationError("stable sample_ids are required for deterministic masking")
    sample_ids = _text(sample_ids).reshape(-1)
    if sample_ids.shape != (n_samples,) or len(set(sample_ids.tolist())) != n_samples:
        raise ReconstructionEvaluationError("sample_ids must be unique and have shape [N]")
    splits = _first_array(raw, ("splits", "split"))
    if splits is None:
        raise ReconstructionEvaluationError("group-disjoint train/val/test split labels are required")
    splits = np.char.lower(_text(splits).reshape(-1))
    aliases = {"training": "train", "validation": "val", "valid": "val", "testing": "test"}
    splits = np.asarray([aliases.get(value, value) for value in splits], dtype=str)
    if splits.shape != (n_samples,) or not set(splits.tolist()).issubset({"train", "val", "test"}):
        raise ReconstructionEvaluationError("splits must have shape [N] and values train/val/test")
    video = _first_array(raw, VIDEO_KEYS)
    track = _first_array(raw, TRACK_KEYS)
    if video is None and track is None:
        raise ReconstructionEvaluationError("video_ids or fallback track_ids are required for leakage auditing")
    if video is not None and _text(video).reshape(-1).shape != (n_samples,):
        raise ReconstructionEvaluationError("video_ids must have shape [N]")
    if track is not None and _text(track).reshape(-1).shape != (n_samples,):
        raise ReconstructionEvaluationError("track_ids must have shape [N]")
    joints = _first_array(raw, ("joint_names",))
    if joints is None:
        raise ReconstructionEvaluationError("joint_names are required to document the exact 10-joint layout")
    joint_names = _text(joints).reshape(-1)
    if joint_names.shape != (10,):
        raise ReconstructionEvaluationError("joint_names must contain exactly the TrackID 10-joint layout")
    edges_array = _first_array(raw, ("edges", "skeleton_edges"))
    if edges_array is None:
        raise ReconstructionEvaluationError("documented skeleton edges are required")
    frame_ids = _first_array(raw, ("frame_ids", "frames"))
    if frame_ids is not None and np.asarray(frame_ids).shape != (n_samples, time_steps):
        raise ReconstructionEvaluationError("frame_ids must have shape [N,T]")
    canonical = dict(raw)
    canonical.update(
        {
            "coords": coordinates,
            "valid_mask": valid,
            "bbox_xywh": bboxes,
            "bbox_valid": valid_boxes,
            "bbox_source": bbox_source,
            "domains": domains,
            "sample_ids": sample_ids,
            "splits": splits,
            "sequence_mask": sequence,
            "video_ids": None if video is None else _text(video).reshape(-1),
            "track_ids": None if track is None else _text(track).reshape(-1),
            "joint_names": joint_names,
            "edge_pairs": np.asarray(_normalise_edges(edges_array, 10), dtype=np.int64),
        }
    )
    return canonical, _metadata_for(source)


def _usable_identity(values: np.ndarray | None) -> bool:
    if values is None or values.size == 0:
        return False
    normalized = np.char.lower(np.char.strip(values.astype(str)))
    return bool(np.all([not any(token == bad or token.startswith(f"{bad}:") for bad in UNAVAILABLE_TOKENS) for token in normalized]))


def _identity_available(value: Any) -> bool:
    token = str(value).strip().lower()
    return not any(token == bad or token.startswith(f"{bad}:") for bad in UNAVAILABLE_TOKENS)


def audit_group_leakage(data: Mapping[str, np.ndarray], metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Audit split groups, preferring source video and falling back to track."""

    meta = metadata or {}
    domains = _text(data["domains"]).reshape(-1)
    splits = _text(data["splits"]).reshape(-1)
    videos = data.get("video_ids")
    tracks = data.get("track_ids")
    video_values = None if videos is None else _text(videos).reshape(-1)
    track_values = None if tracks is None else _text(tracks).reshape(-1)
    declared_video = _first_present(
        meta.get("video_identity_available"),
        _nested(meta, "provenance.video_identity_available"),
    )
    stored_groups = data.get("split_group")
    grouping_levels = data.get("grouping_level")
    if stored_groups is not None:
        stored = _text(stored_groups).reshape(-1)
        if stored.shape != domains.shape or not _usable_identity(stored):
            return {
                "passed": False,
                "grouping_unit": "invalid_split_group",
                "reason": "stored split_group values are absent or malformed",
                "overlaps": {},
            }
        groups = stored
        if grouping_levels is not None:
            level_values = sorted(set(_text(grouping_levels).reshape(-1).tolist()))
            grouping_unit = "+".join(level_values)
        else:
            grouping_unit = "stored_split_group"
    elif video_values is not None and track_values is not None:
        # TrackID's Drone annotations lack source video identity, while the fixed
        # camera domains provide it.  Choose the strongest identifier per sample.
        identities: list[str] = []
        used_video = False
        used_track = False
        for domain, video, track in zip(domains, video_values, track_values, strict=True):
            if declared_video is not False and _identity_available(video):
                identities.append(f"{domain}::video::{video}")
                used_video = True
            elif _identity_available(track):
                identities.append(f"{domain}::track::{track}")
                used_track = True
            else:
                identities.append("")
        groups = np.asarray(identities, dtype=str)
        if not _usable_identity(groups):
            return {
                "passed": False,
                "grouping_unit": "unavailable",
                "reason": "one or more samples have neither video nor fallback track identity",
                "overlaps": {},
            }
        grouping_unit = "video" if used_video and not used_track else "track_fallback" if used_track and not used_video else "video_preferred+track_fallback"
    elif declared_video is not False and _usable_identity(video_values):
        assert video_values is not None
        grouping_unit = "video"
        groups = np.asarray([f"{domain}::video::{identity}" for domain, identity in zip(domains, video_values, strict=True)])
    elif _usable_identity(track_values):
        assert track_values is not None
        grouping_unit = "track_fallback"
        groups = np.asarray([f"{domain}::track::{identity}" for domain, identity in zip(domains, track_values, strict=True)])
    else:
        return {
            "passed": False,
            "grouping_unit": "unavailable",
            "reason": "neither usable video IDs nor fallback track IDs are present",
            "overlaps": {},
        }
    split_groups = {name: set(groups[splits == name].tolist()) for name in ("train", "val", "test")}
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    overlaps = {f"{left}/{right}": sorted(split_groups[left] & split_groups[right]) for left, right in pairs}
    offenders = {name: values for name, values in overlaps.items() if values}
    return {
        "passed": not offenders,
        "grouping_unit": grouping_unit,
        "group_counts": {name: len(values) for name, values in split_groups.items()},
        "overlap_counts": {name: len(values) for name, values in overlaps.items()},
        "overlaps": {name: values[:20] for name, values in offenders.items()},
        "video_preferred": True,
        "track_fallback_used": "track" in grouping_unit,
    }


def _checkpoint_metadata(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - arbitrary checkpoint candidates may be corrupt
        return None
    return payload if isinstance(payload, Mapping) and payload.get("task") == "keypoint_reconstruction" else None


def resolve_checkpoint(path: str | Path | None, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve the best actual reconstruction checkpoint, preferring non-synthetic NMJE."""

    if path is not None:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        if not candidate.is_file() or _checkpoint_metadata(candidate) is None:
            raise ReconstructionEvaluationError(f"{candidate} is not a keypoint-reconstruction checkpoint")
        return candidate.resolve()
    root = project_root / "outputs" / "checkpoints"
    candidates: list[tuple[tuple[int, float, int, float], Path]] = []
    for candidate in root.glob("**/*") if root.exists() else ():
        if not candidate.is_file() or candidate.suffix.lower() not in {".pt", ".pth", ".ckpt"}:
            continue
        payload = _checkpoint_metadata(candidate)
        if payload is None:
            continue
        validation = payload.get("val_metrics") if isinstance(payload.get("val_metrics"), Mapping) else {}
        score = _first_present(validation.get("nmje"), validation.get("mpjpe"), float("inf"))
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = float("inf")
        best_name = 0 if candidate.stem.lower().startswith("best") else 1
        synthetic = 1 if bool(payload.get("synthetic", False)) else 0
        candidates.append(((synthetic, score_value, best_name, -candidate.stat().st_mtime), candidate.resolve()))
    if not candidates:
        raise ReconstructionEvaluationError("No keypoint-reconstruction checkpoint was found")
    return min(candidates, key=lambda item: item[0])[1]


def _checkpoint_field(checkpoint: Mapping[str, Any], name: str, default: Any = None) -> Any:
    experiment = checkpoint.get("experiment") if isinstance(checkpoint.get("experiment"), Mapping) else {}
    paths = {
        "representation": ("representation", "config.representation", "config.features.representation"),
        "train_domain": ("train_domain", "domain", "config.train_domain", "config.data.train_domain"),
        "data_percent": ("data_percent", "data_fraction", "config.data_fraction"),
        "sequence_length": ("sequence_length", "config.data.sequence_length"),
        "seed": ("seed", "config.project.seed"),
        "mask_ratio": ("mask_ratio", "config.mask_ratio", "config.evaluation.mask_ratio"),
    }
    for path in paths.get(name, (name,)):
        value = _nested(experiment, path)
        if value is None:
            value = _nested(checkpoint, path)
        if value is not None:
            return value
    return default


def _selected_indices(data: Mapping[str, np.ndarray], split: str, test_domain: str | None) -> np.ndarray:
    selector = _text(data["splits"]).reshape(-1) == split.lower()
    if test_domain and test_domain.lower() != "all":
        selector &= np.char.lower(_text(data["domains"]).reshape(-1)) == test_domain.lower()
    selected = np.flatnonzero(selector)
    if selected.size == 0:
        raise ReconstructionEvaluationError(f"no sequences found for split={split!r}, domain={test_domain!r}")
    return selected


def _same_frame_bbox_overlap(
    data: Mapping[str, np.ndarray], selected: np.ndarray
) -> np.ndarray | None:
    """Return per-sequence max peer-box IoU only when frame grouping is observable."""

    frame_ids = _first_array(data, ("frame_index", "frame_ids", "frames"))
    source_files = _first_array(data, ("source_file", "source_files"))
    if frame_ids is None or np.asarray(frame_ids).shape[:2] != np.asarray(data["coords"]).shape[:2]:
        return None
    boxes = np.asarray(data["bbox_xywh"], dtype=float)[selected]
    valid_boxes = np.asarray(data["bbox_valid"], dtype=bool)[selected]
    frames = np.asarray(frame_ids)[selected]
    domains = _text(data["domains"])[selected]
    videos = data.get("video_ids")
    video_values = None if videos is None else _text(videos)[selected]
    source_values = None if source_files is None else _text(source_files)[selected]
    sequence = np.asarray(data["sequence_mask"], dtype=bool)[selected]
    groups: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    for sample_index in range(len(selected)):
        for time_index in np.flatnonzero(sequence[sample_index] & valid_boxes[sample_index]):
            video = "" if video_values is None else str(video_values[sample_index]).strip()
            if video:
                source = video
            elif source_values is not None:
                source = str(source_values[sample_index, time_index])
            else:
                return None
            key = (str(domains[sample_index]), source, str(frames[sample_index, time_index]))
            groups.setdefault(key, []).append((sample_index, int(time_index)))

    values: list[list[float]] = [[] for _ in range(len(selected))]
    for members in groups.values():
        if len(members) < 2:
            continue
        for first_index, (sample_index, time_index) in enumerate(members):
            x, y, width, height = boxes[sample_index, time_index]
            area = max(0.0, width) * max(0.0, height)
            best = 0.0
            for second_index, (other_sample, other_time) in enumerate(members):
                if first_index == second_index:
                    continue
                ox, oy, other_width, other_height = boxes[other_sample, other_time]
                intersection_width = max(0.0, min(x + width, ox + other_width) - max(x, ox))
                intersection_height = max(0.0, min(y + height, oy + other_height) - max(y, oy))
                intersection = intersection_width * intersection_height
                union = area + max(0.0, other_width) * max(0.0, other_height) - intersection
                if union > 0:
                    best = max(best, intersection / union)
            values[sample_index].append(best)
    return np.asarray(
        [float(np.mean(items)) if items else np.nan for items in values], dtype=float
    )


def _predict(
    coordinates: np.ndarray,
    valid_mask: np.ndarray,
    bboxes: np.ndarray,
    bbox_valid: np.ndarray,
    bbox_source: np.ndarray,
    sequence_mask: np.ndarray,
    sample_ids: np.ndarray,
    *,
    mask_ratio: float,
    seed: int,
    representation: str,
    model: torch.nn.Module | None,
    baseline: str | None,
    edges: Sequence[tuple[int, int]],
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    original = torch.from_numpy(np.asarray(valid_mask, dtype=bool))
    synthetic = deterministic_reconstruction_mask(
        original,
        mask_ratio,
        seed=seed,
        instance_ids=[str(value) for value in sample_ids],
    )
    prepared = prepare_reconstruction_representation(
        torch.from_numpy(np.asarray(coordinates, dtype=np.float32)),
        original,
        synthetic,
        representation=representation,
        bbox_xywh=torch.from_numpy(np.asarray(bboxes, dtype=np.float32)),
        bbox_valid=torch.from_numpy(np.asarray(bbox_valid, dtype=bool)),
        bbox_source=np.asarray(bbox_source).astype(str),
        sequence_mask=torch.from_numpy(np.asarray(sequence_mask, dtype=bool)),
    )
    inputs = prepared["inputs"]
    observed = prepared["observed_mask"]
    track_valid = prepared["track_valid_mask"]
    auxiliary = prepared["auxiliary"]
    auxiliary_mask = prepared["auxiliary_mask"]
    assert isinstance(inputs, torch.Tensor)
    assert isinstance(observed, torch.Tensor)
    represented_predictions: list[torch.Tensor] = []
    started = time.perf_counter()
    if model is not None:
        model.eval().to(device)
        with torch.inference_mode():
            for start in range(0, len(inputs), batch_size):
                stop = min(len(inputs), start + batch_size)
                kwargs: dict[str, Any] = {
                    "track_valid_mask": track_valid[start:stop].to(device),
                }
                if isinstance(auxiliary, torch.Tensor) and auxiliary.shape[-1] > 0:
                    kwargs["auxiliary"] = auxiliary[start:stop].to(device)
                    kwargs["auxiliary_mask"] = auxiliary_mask[start:stop].to(device)
                output = model(inputs[start:stop].to(device), observed[start:stop].to(device), **kwargs)
                prediction = output[0] if isinstance(output, tuple) else output
                represented_predictions.append(prediction.detach().cpu())
        represented = torch.cat(represented_predictions)
    else:
        assert baseline is not None
        represented = reconstruction_baseline(
            inputs,
            observed,
            method=baseline,
            edges=edges,
            track_valid_mask=track_valid,
        )
    elapsed = time.perf_counter() - started
    pixel_prediction = inverse_reconstruction_coordinates(
        represented,
        prepared["inverse_offset"],
        prepared["inverse_scale"],
    )
    pixel_inputs = torch.where(
        observed.unsqueeze(-1),
        prepared["pixel_targets"],
        torch.zeros_like(prepared["pixel_targets"]),
    )
    return {
        "prediction": pixel_prediction.detach().cpu().numpy(),
        "target": prepared["pixel_targets"].detach().cpu().numpy(),
        "masked_input": pixel_inputs.detach().cpu().numpy(),
        "represented_input": inputs.detach().cpu().numpy(),
        "represented_target": prepared["targets"].detach().cpu().numpy(),
        "synthetic_mask": synthetic.detach().cpu().numpy(),
        "observed_mask": observed.detach().cpu().numpy(),
        "original_valid_mask": prepared["original_valid_mask"].detach().cpu().numpy(),
        "transform_bbox_xywh": prepared["transform_bbox_xywh"].detach().cpu().numpy(),
        "transform_valid": prepared["transform_valid"].detach().cpu().numpy(),
        "pose_tight_metric_only": prepared["pose_tight_metric_only"].detach().cpu().numpy(),
        "transform_center_source": np.asarray(prepared["transform_center_source"]),
        "transform_scale_source": np.asarray(prepared["transform_scale_source"]),
        "bbox_metric_source": np.asarray(prepared["bbox_metric_source"]),
        "inference_time_seconds": elapsed,
    }


def _save_skeleton(
    pose: np.ndarray,
    valid: np.ndarray,
    edges: Sequence[tuple[int, int]],
    joint_names: Sequence[str],
    title: str,
    path: Path,
) -> None:
    shown = np.asarray(pose, dtype=float).copy()
    shown[~np.asarray(valid, dtype=bool)] = np.nan
    figure, _ = plot_skeleton(shown, edges, joint_names=joint_names, title=title, output_path=path)
    plt.close(figure)


def _example_location(valid: np.ndarray, synthetic: np.ndarray | None = None) -> tuple[int, int]:
    score = valid.sum(axis=-1).astype(float)
    if synthetic is not None:
        score += 1000.0 * synthetic.any(axis=-1)
    flat = int(np.argmax(score))
    return tuple(int(value) for value in np.unravel_index(flat, score.shape))


def _dataset_figures(
    data: Mapping[str, np.ndarray],
    figures_dir: Path,
    *,
    synthetic_suffix: str,
) -> list[str]:
    notes: list[str] = []
    coordinates = np.asarray(data["coords"])
    valid = np.asarray(data["valid_mask"])
    bboxes = np.asarray(data["bbox_xywh"])
    domains = _text(data["domains"]).reshape(-1)
    sequence = np.asarray(data["sequence_mask"])
    joints = _text(data["joint_names"]).reshape(-1).tolist()
    edges = [tuple(value) for value in np.asarray(data["edge_pairs"]).tolist()]
    figure, _ = plot_domain_summary(domains, title="TrackID3x3 sequences by domain" + synthetic_suffix, output_path=figures_dir / "dataset_domain_summary.png")
    plt.close(figure)
    figure, _ = plot_pose_instance_counts(domains, valid, sequence_mask=sequence, title="Pose-instance counts by domain" + synthetic_suffix, output_path=figures_dir / "pose_instance_counts.png")
    plt.close(figure)
    missing = missingness_by_domain_joint(domains, valid, sequence_mask=sequence, joint_names=joints)
    figure, _ = plot_missingness_heatmap(missing, title="Natural missingness by domain and joint" + synthetic_suffix, output_path=figures_dir / "missingness_by_domain_joint.png")
    plt.close(figure)
    zero_mask = np.zeros_like(valid, dtype=bool)
    normalized = prepare_reconstruction_representation(
        torch.from_numpy(coordinates),
        torch.from_numpy(valid),
        torch.from_numpy(zero_mask),
        representation="bbox_normalized",
        bbox_xywh=torch.from_numpy(bboxes),
        bbox_valid=torch.from_numpy(np.asarray(data["bbox_valid"], dtype=bool)),
        bbox_source=np.asarray(data["bbox_source"]).astype(str),
        sequence_mask=torch.from_numpy(sequence),
    )["targets"].numpy()
    figure, _ = plot_coordinate_distributions(coordinates, normalized, valid_mask=valid, title="Raw vs bbox-normalized coordinates" + synthetic_suffix, output_path=figures_dir / "coordinate_distributions_raw_vs_normalized.png")
    plt.close(figure)
    index, frame = _example_location(valid)
    _save_skeleton(coordinates[index, frame], valid[index, frame], edges, joints, "Example TrackID3x3 10-joint skeleton" + synthetic_suffix, figures_dir / "example_10_joint_skeleton.png")
    _save_skeleton(normalized[index, frame], valid[index, frame], edges, joints, "BBox-normalized 10-joint skeleton" + synthetic_suffix, figures_dir / "normalized_skeleton.png")

    present_domains = [domain for domain in DOMAINS if domain in set(domains.tolist())]
    if present_domains:
        figure, axes = plt.subplots(1, len(present_domains), figsize=(4.0 * len(present_domains), 4.3), squeeze=False, sharex=False, sharey=False, constrained_layout=True)
        for axis, domain in zip(axes[0], present_domains, strict=True):
            source_index = int(np.flatnonzero(domains == domain)[0])
            frame_index = int(np.argmax(valid[source_index].sum(axis=-1)))
            shown = coordinates[source_index, frame_index].copy()
            shown[~valid[source_index, frame_index]] = np.nan
            plot_skeleton(shown, edges, title=domain, ax=axis)
        figure.suptitle("Skeleton examples by capture domain" + synthetic_suffix, fontsize=12)
        figure.savefig(figures_dir / "skeleton_examples_by_domain.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)
    else:
        notes.append("skeleton_examples_by_domain: no recognized TrackID domain is present")
    return notes


def _history_figure(checkpoint_path: Path | None, checkpoint: Mapping[str, Any], figures_dir: Path, suffix: str) -> str | None:
    if checkpoint_path is None:
        return "training_curves: deterministic baseline has no training history"
    experiment = checkpoint.get("experiment") if isinstance(checkpoint.get("experiment"), Mapping) else {}
    candidates = [
        checkpoint_path.parent / "history.csv",
        PROJECT_ROOT / "outputs" / "logs" / checkpoint_path.parent.name / "history.csv",
    ]
    run_dir = experiment.get("run_dir")
    if run_dir:
        candidates.insert(0, Path(str(run_dir)) / "history.csv")
    history_path = next((path for path in candidates if path.is_file()), None)
    if history_path is None:
        return "training_curves: no matching executed history.csv"
    try:
        history = pd.read_csv(history_path)
    except (OSError, pd.errors.ParserError):
        return f"training_curves: unreadable {history_path}"
    epoch = history["epoch"] if "epoch" in history else np.arange(1, len(history) + 1)
    panels = []
    if {"train_loss", "val_loss"}.issubset(history):
        panels.append(("Loss", "train_loss", "val_loss"))
    if {"train_nmje", "val_nmje"}.issubset(history):
        panels.append(("NMJE", "train_nmje", "val_nmje"))
    if not panels:
        return "training_curves: history lacks train/validation loss or NMJE columns"
    figure, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 4.2), squeeze=False, constrained_layout=True)
    for axis, (label, train_column, val_column) in zip(axes[0], panels, strict=True):
        axis.plot(epoch, history[train_column], label="train")
        axis.plot(epoch, history[val_column], label="validation")
        axis.set(xlabel="Epoch", ylabel=label, title=label)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle("Executed training history" + suffix, fontsize=12)
    figure.savefig(figures_dir / "training_curves.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return None


def _result_figures(
    primary: Mapping[str, Any],
    ratio_rows: pd.DataFrame,
    domain_rows: pd.DataFrame,
    per_joint: pd.DataFrame,
    data: Mapping[str, np.ndarray],
    figures_dir: Path,
    *,
    suffix: str,
) -> list[str]:
    notes: list[str] = []
    predictions = np.asarray(primary["prediction"])
    targets = np.asarray(primary["target"])
    masked = np.asarray(primary["masked_input"])
    synthetic_mask = np.asarray(primary["synthetic_mask"])
    valid = np.asarray(primary["original_valid_mask"])
    edges = [tuple(value) for value in np.asarray(data["edge_pairs"]).tolist()]
    joints = _text(data["joint_names"]).reshape(-1).tolist()
    index, frame = _example_location(valid, synthetic_mask)
    title_ratio = 100.0 * float(primary["mask_ratio"])
    _save_skeleton(masked[index, frame], np.asarray(primary["observed_mask"])[index, frame], edges, joints, f"Masked skeleton ({title_ratio:.0f}% held out)" + suffix, figures_dir / "masked_skeleton.png")
    _save_skeleton(predictions[index, frame], valid[index, frame], edges, joints, "Reconstructed skeleton" + suffix, figures_dir / "reconstructed_skeleton.png")
    figure, _ = plot_reconstruction_example(
        targets[index, frame],
        masked[index, frame],
        predictions[index, frame],
        synthetic_mask=synthetic_mask[index, frame],
        original_valid_mask=valid[index, frame],
        edges=edges,
        bbox=np.asarray(primary["bboxes"])[index, frame],
        title=f"Ground truth vs reconstruction ({title_ratio:.0f}% mask)" + suffix,
        output_path=figures_dir / "ground_truth_vs_reconstruction.png",
    )
    plt.close(figure)
    figure, _ = plot_per_joint_error(per_joint, title="Per-joint reconstruction error" + suffix, output_path=figures_dir / "per_joint_reconstruction_error.png")
    plt.close(figure)
    if len(ratio_rows) >= 2:
        ordered = ratio_rows.sort_values("mask_percentage")
        figure, axis = plt.subplots(figsize=(6.3, 4.4), constrained_layout=True)
        axis.plot(ordered["mask_percentage"], ordered["nmje"], marker="o", color="#b45309")
        axis.set(xlabel="Experimentally masked keypoints (%)", ylabel="NMJE (bbox diagonal)", title="Error vs missing percentage" + suffix)
        axis.grid(alpha=0.22)
        figure.savefig(figures_dir / "error_vs_missing_percentage.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        figure, _ = plot_pck_curves(ordered, title="PCK across mask ratios" + suffix, output_path=figures_dir / "pck_curves.png")
        plt.close(figure)
    else:
        notes.append("error_vs_missing_percentage and pck_curves: require >=2 executed mask ratios")
    if "test_domain" in domain_rows and domain_rows["test_domain"].nunique() >= 2:
        domain = domain_rows.groupby("test_domain", as_index=False)["nmje"].mean()
        figure, axis = plt.subplots(figsize=(6.4, 4.3), constrained_layout=True)
        bars = axis.bar(domain["test_domain"], domain["nmje"], color="#228833")
        axis.bar_label(bars, labels=[f"{value:.4f}" for value in domain["nmje"]], padding=3)
        axis.set(xlabel="Evaluation domain", ylabel="NMJE", title="Indoor vs Outdoor vs Drone performance" + suffix)
        axis.grid(axis="y", alpha=0.2)
        figure.savefig(figures_dir / "indoor_outdoor_drone_performance.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)
    else:
        notes.append("indoor_outdoor_drone_performance: evaluation contains one domain")
    return notes


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run deterministic reconstruction evaluation and write auditable artifacts."""

    processed_path = resolve_processed(args.processed, PROJECT_ROOT)
    data, metadata = load_reconstruction_data(processed_path)
    audit = audit_group_leakage(data, metadata)
    if not audit["passed"]:
        raise ReconstructionEvaluationError(f"group leakage audit failed: {audit}")
    archive_synthetic = _first_array(data, ("synthetic",))
    synthetic = bool(metadata.get("synthetic", False)) or bool(
        archive_synthetic is not None and archive_synthetic.size and np.asarray(archive_synthetic).reshape(-1)[0]
    )
    if synthetic and not args.include_synthetic:
        raise ReconstructionEvaluationError(
            "processed artifact is synthetic; pass --include-synthetic for software validation only"
        )
    checkpoint_path: Path | None = None
    checkpoint: Mapping[str, Any] = {}
    model: torch.nn.Module | None = None
    if args.baseline is None:
        checkpoint_path = resolve_checkpoint(args.checkpoint, PROJECT_ROOT)
        model, loaded = load_reconstruction_checkpoint(checkpoint_path, map_location="cpu")
        checkpoint = loaded
        checkpoint_synthetic = bool(checkpoint.get("synthetic", False))
        synthetic = synthetic or checkpoint_synthetic
        if checkpoint_synthetic and not args.include_synthetic:
            raise ReconstructionEvaluationError(
                "checkpoint is synthetic; pass --include-synthetic for software validation only"
            )
        model_name = str(checkpoint.get("model_name", type(model).__name__))
        checkpoint_representation = canonical_representation_name(
            str(_checkpoint_field(checkpoint, "representation", "bbox_normalized"))
        )
        if args.representation and canonical_representation_name(args.representation) != checkpoint_representation:
            raise ReconstructionEvaluationError(
                "--representation must match the trained checkpoint representation"
            )
        representation = checkpoint_representation
    else:
        model_name = f"{args.baseline}_baseline"
        representation = canonical_representation_name(args.representation or "bbox_normalized")
    train_domain_value = _first_present(args.train_domain, _checkpoint_field(checkpoint, "train_domain"))
    train_domain = (
        "not_applicable"
        if args.baseline
        else str(train_domain_value) if train_domain_value is not None else "unknown"
    )
    test_domain_arg = args.test_domain
    selected = _selected_indices(data, args.split, test_domain_arg)
    observed_sequence_length = int(np.asarray(data["coords"]).shape[1])
    trained_sequence_length = _checkpoint_field(checkpoint, "sequence_length") if checkpoint else None
    if trained_sequence_length is not None:
        try:
            expected_sequence_length = int(trained_sequence_length)
        except (TypeError, ValueError) as exc:
            raise ReconstructionEvaluationError(
                f"checkpoint sequence_length is not an integer: {trained_sequence_length!r}"
            ) from exc
        if expected_sequence_length != observed_sequence_length:
            raise ReconstructionEvaluationError(
                "processed sequence length does not match the checkpoint: "
                f"T={observed_sequence_length} versus trained T={expected_sequence_length}"
            )
    requires_reliable_temporal = model_name.strip().lower() == "tcn" or args.baseline in {
        "temporal",
        "hybrid",
    }
    temporal_reliable_selected: np.ndarray | None = None
    if requires_reliable_temporal:
        if observed_sequence_length <= 1:
            raise ReconstructionEvaluationError(
                "temporal TCN/interpolation evaluation requires reliable windows with T>1; "
                "the selected artifact has T=1"
            )
        temporal_reliable = _first_array(data, ("temporal_reliable",))
        if temporal_reliable is None:
            raise ReconstructionEvaluationError(
                "temporal TCN/interpolation evaluation requires an explicit per-sequence "
                "temporal_reliable audit field"
            )
        reliable = np.asarray(temporal_reliable, dtype=bool).reshape(-1)
        if reliable.shape != np.asarray(data["coords"]).shape[:1]:
            raise ReconstructionEvaluationError("temporal_reliable must have shape [N]")
        if not np.all(reliable[selected]):
            raise ReconstructionEvaluationError(
                "temporal TCN/interpolation evaluation selected one or more sequences without "
                "reliable temporal identity"
            )
        temporal_reliable_selected = reliable[selected]
    selected_domains = _text(data["domains"])[selected].reshape(-1)
    target_domains = [domain for domain in DOMAINS if domain in set(selected_domains.tolist())]
    target_domains.extend(sorted(set(selected_domains.tolist()) - set(target_domains)))
    if not target_domains:
        raise ReconstructionEvaluationError("selected evaluation set has no domains")
    ratios = [float(value) for value in args.mask_ratios]
    unsupported = [value for value in ratios if not any(abs(value - allowed) < 1e-9 for allowed in SUPPORTED_MASK_RATIOS)]
    if unsupported:
        raise ReconstructionEvaluationError(f"mask ratios must be among {SUPPORTED_MASK_RATIOS}; got {unsupported}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ReconstructionEvaluationError("CUDA was requested but is unavailable")

    run_name = args.run_name or (
        f"{model_name}_{representation}_{train_domain.lower().replace(' ', '_')}"
    )
    smoke = bool(
        "smoke" in run_name.lower()
        or "smoke" in str(processed_path).lower()
        or checkpoint_path is not None and "smoke" in str(checkpoint_path).lower()
        or bool(checkpoint.get("smoke", False))
        or bool(_nested(checkpoint, "experiment.smoke", False))
    )
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir / run_name
    figures_dir = Path(args.figures_dir)
    if not figures_dir.is_absolute():
        figures_dir = PROJECT_ROOT / figures_dir
    figures_dir = figures_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    suffix = (
        "\n(synthetic software validation; not research results)"
        if synthetic
        else "\n(software smoke run; not a primary research result)"
        if smoke
        else ""
    )
    figure_notes = _dataset_figures(data, figures_dir, synthetic_suffix=suffix)
    history_note = _history_figure(checkpoint_path, checkpoint, figures_dir, suffix)
    if history_note:
        figure_notes.append(history_note)

    all_ratio_rows: list[dict[str, Any]] = []
    primary_candidates: list[dict[str, Any]] = []
    artifacts: list[str] = []
    total_inference = 0.0
    coordinates = np.asarray(data["coords"])[selected]
    valid = np.asarray(data["valid_mask"])[selected]
    bboxes = np.asarray(data["bbox_xywh"])[selected]
    bbox_valid = np.asarray(data["bbox_valid"], dtype=bool)[selected]
    bbox_source = np.asarray(data["bbox_source"]).astype(str)[selected]
    sequence = np.asarray(data["sequence_mask"])[selected]
    sample_ids = _text(data["sample_ids"])[selected]
    domains = _text(data["domains"])[selected]
    video = data.get("video_ids")
    track = data.get("track_ids")
    video_selected = None if video is None else _text(video)[selected]
    track_selected = None if track is None else _text(track)[selected]
    same_frame_overlap = _same_frame_bbox_overlap(data, selected)
    edges = [tuple(value) for value in np.asarray(data["edge_pairs"]).tolist()]
    joints = _text(data["joint_names"]).reshape(-1).tolist()
    parameter_count = int(count_parameters(model)) if model is not None else 0
    data_percent = _checkpoint_field(checkpoint, "data_percent", 100)
    if isinstance(data_percent, (int, float)) and float(data_percent) <= 1:
        data_percent = 100 * float(data_percent)

    for ratio in ratios:
        prediction = _predict(
            coordinates,
            valid,
            bboxes,
            bbox_valid,
            bbox_source,
            sequence,
            sample_ids,
            mask_ratio=ratio,
            seed=args.seed,
            representation=representation,
            model=model,
            baseline=args.baseline,
            edges=edges,
            batch_size=args.batch_size,
            device=device,
        )
        prediction["mask_ratio"] = ratio
        prediction["bboxes"] = bboxes
        total_inference += float(prediction["inference_time_seconds"])
        ratio_directory = output_dir / f"mask_{round(100 * ratio):02d}"
        ratio_directory.mkdir(parents=True, exist_ok=True)
        if not args.no_save_predictions:
            prediction_path = ratio_directory / "predictions.npz"
            np.savez_compressed(
                prediction_path,
                prediction=np.asarray(prediction["prediction"], dtype=np.float32),
                target=np.asarray(prediction["target"], dtype=np.float32),
                masked_input=np.asarray(prediction["masked_input"], dtype=np.float32),
                original_valid_mask=np.asarray(prediction["original_valid_mask"], dtype=bool),
                synthetic_mask=np.asarray(prediction["synthetic_mask"], dtype=bool),
                observed_mask=np.asarray(prediction["observed_mask"], dtype=bool),
                bbox_xywh=bboxes,
                bbox_valid=bbox_valid,
                bbox_source=bbox_source,
                transform_bbox_xywh=np.asarray(prediction["transform_bbox_xywh"], dtype=np.float32),
                pose_tight_metric_only=np.asarray(prediction["pose_tight_metric_only"], dtype=bool),
                sequence_mask=sequence,
                sample_ids=sample_ids,
                domains=domains,
            )
            artifacts.append(str(prediction_path.resolve()))
        error_table = build_reconstruction_error_table(
            prediction["prediction"],
            prediction["target"],
            bboxes=bboxes,
            evaluation_mask=prediction["synthetic_mask"],
            original_valid_mask=prediction["original_valid_mask"],
            sequence_mask=sequence,
            sample_ids=sample_ids,
            domains=domains,
            video_ids=video_selected,
            track_ids=track_selected,
            mask_ratio=ratio,
        )
        if same_frame_overlap is not None:
            error_table["same_frame_max_bbox_iou"] = same_frame_overlap
        error_table.to_csv(ratio_directory / "sequence_errors.csv", index=False)
        for domain in target_domains:
            domain_selector = domains == domain
            metrics = reconstruction_metrics(
                np.asarray(prediction["prediction"])[domain_selector],
                np.asarray(prediction["target"])[domain_selector],
                bboxes=bboxes[domain_selector],
                evaluation_mask=np.asarray(prediction["synthetic_mask"])[domain_selector],
                original_valid_mask=np.asarray(prediction["original_valid_mask"])[domain_selector],
                sequence_mask=sequence[domain_selector],
                joint_names=joints,
            )
            effective_train = domain if train_domain == "unknown" else train_domain
            protocol = (
                "baseline_domain_stratified"
                if args.baseline
                else "pooled_domain"
                if effective_train.lower() in {"all", "pooled", "mixed", "all_domains"}
                else "within_domain" if effective_train.lower() == domain.lower() else "cross_domain"
            )
            metric_sources = bbox_source[domain_selector]
            pose_tight_expected = metric_sources == "pose_tight_fallback"
            pose_tight_flags = np.asarray(prediction["pose_tight_metric_only"])[domain_selector]
            input_transform_audit_passed = bool(
                np.array_equal(pose_tight_flags, pose_tight_expected)
            )
            if not input_transform_audit_passed:
                raise ReconstructionEvaluationError(
                    "bbox-source transform audit failed; refusing to write metrics"
                )
            domain_dir = ratio_directory / domain.lower()
            payload: dict[str, Any] = {
                "artifact_type": "reconstruction_evaluation",
                "task": "keypoint_reconstruction",
                "title": TITLE,
                "metrics": metrics,
                "model": model_name,
                "representation": representation,
                "mask_ratio": ratio,
                "seed": args.seed,
                "protocol": protocol,
                "train_domain": effective_train,
                "test_domain": domain,
                "split": args.split,
                "grouping_unit": audit["grouping_unit"],
                "leakage_audit_passed": True,
                "leakage_audit": audit,
                "input_transform_audit_passed": input_transform_audit_passed,
                "input_transform_provenance": {
                    "policy": "official MOT boxes may normalize inputs; pose_tight_fallback is metric-denominator-only and input transforms are recomputed from observed joints after masking",
                    "bbox_metric_sources": sorted(set(metric_sources.reshape(-1).tolist())),
                    "pose_tight_metric_only_frames": int(pose_tight_flags.sum()),
                    "transform_center_sources": sorted(
                        set(
                            np.asarray(prediction["transform_center_source"])[domain_selector]
                            .reshape(-1)
                            .astype(str)
                            .tolist()
                        )
                    ),
                    "transform_scale_sources": sorted(
                        set(
                            np.asarray(prediction["transform_scale_source"])[domain_selector]
                            .reshape(-1)
                            .astype(str)
                            .tolist()
                        )
                    ),
                },
                "synthetic": synthetic,
                "smoke": smoke,
                "result_scope": (
                    "software validation only / NOT research results"
                    if synthetic
                    else "software smoke run / NOT primary research result"
                    if smoke
                    else "research result"
                ),
                "checkpoint": None if checkpoint_path is None else str(checkpoint_path.resolve()),
                "processed_dataset": str(processed_path.resolve()),
                "parameter_count": parameter_count,
                "data_percent": data_percent,
                "sequence_length": int(coordinates.shape[1]),
                "temporal_protocol_audit": {
                    "requires_reliable_temporal_windows": requires_reliable_temporal,
                    "selected_sequences_explicitly_reliable": (
                        bool(np.all(temporal_reliable_selected))
                        if temporal_reliable_selected is not None
                        else None
                    ),
                },
                "n_sequences": int(domain_selector.sum()),
                "inference_time_seconds": float(prediction["inference_time_seconds"]),
                "bbox_error_normalization": "source bbox diagonal in pixel space",
                "error_analysis_covariates": {
                    "bbox_scale": "evaluated",
                    "pose_spread_deformation_proxy": "evaluated",
                    "masked_distal_vs_non_distal": "evaluated",
                    "domain_view": "evaluated",
                    "same_frame_bbox_overlap": (
                        "evaluated" if same_frame_overlap is not None else "not evaluated: same-frame peer boxes unavailable"
                    ),
                    "temporal_motion": (
                        "not evaluated: T=1" if coordinates.shape[1] == 1 else "available from continuous sequence coordinates"
                    ),
                },
                "rgb_frames_used": False,
                "rgb_overlay_status": "NOT GENERATED: annotation-only local acquisition contains no source RGB frames; remote media size was not assumed.",
            }
            metrics_path = _write_json(domain_dir / "metrics.json", payload)
            artifacts.append(str(metrics_path.resolve()))
            all_ratio_rows.append(
                {
                    "mask_ratio": ratio,
                    "mask_percentage": 100 * ratio,
                    "test_domain": domain,
                    "nmje": metrics["nmje"],
                    "pck_at_0_05": metrics["pck_at_0_05"],
                    "pck_at_0_10": metrics["pck_at_0_10"],
                    "pck_at_0_20": metrics["pck_at_0_20"],
                }
            )
        overall = reconstruction_metrics(
            prediction["prediction"],
            prediction["target"],
            bboxes=bboxes,
            evaluation_mask=prediction["synthetic_mask"],
            original_valid_mask=prediction["original_valid_mask"],
            sequence_mask=sequence,
            joint_names=joints,
        )
        _write_json(ratio_directory / "mask_summary.json", {"task": "keypoint_reconstruction", "mask_ratio": ratio, "metrics": overall, "domain_metrics_paths": [path for path in artifacts if f"mask_{round(100 * ratio):02d}" in path and path.endswith("metrics.json")]})
        primary_candidates.append({**prediction, "metrics": overall})

    ratio_table = pd.DataFrame(all_ratio_rows)
    ratio_table.to_csv(output_dir / "missingness_metrics.csv", index=False)
    primary_index = min(
        range(len(ratios)),
        key=lambda index: abs(ratios[index] - float(args.primary_mask_ratio)),
    )
    primary = primary_candidates[primary_index]
    per_joint = build_per_joint_reconstruction_table(
        primary["prediction"],
        primary["target"],
        bboxes=bboxes,
        evaluation_mask=primary["synthetic_mask"],
        original_valid_mask=primary["original_valid_mask"],
        sequence_mask=sequence,
        joint_names=joints,
    )
    per_joint.to_csv(output_dir / "per_joint_reconstruction_error.csv", index=False)
    primary_rows = ratio_table[ratio_table["mask_ratio"] == ratios[primary_index]].reset_index(drop=True)
    plot_rows = ratio_table.groupby("mask_percentage", as_index=False).agg(
        nmje=("nmje", "mean"),
        pck_at_0_05=("pck_at_0_05", "mean"),
        pck_at_0_10=("pck_at_0_10", "mean"),
        pck_at_0_20=("pck_at_0_20", "mean"),
    )
    plot_rows["mask_ratio"] = plot_rows["mask_percentage"] / 100
    figure_notes.extend(
        _result_figures(
            primary,
            plot_rows,
            primary_rows,
            per_joint,
            data,
            figures_dir,
            suffix=suffix,
        )
    )

    index = {
        "artifact_type": "reconstruction_evaluation_index",
        "task": "keypoint_reconstruction",
        "title": TITLE,
        "model": model_name,
        "representation": representation,
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path.resolve()),
        "processed_dataset": str(processed_path.resolve()),
        "mask_ratios": ratios,
        "seed": args.seed,
        "split": args.split,
        "train_domain": train_domain,
        "test_domains": target_domains,
        "leakage_audit": audit,
        "input_transform_audit_passed": True,
        "input_transform_policy": "pose_tight_fallback bboxes are metric-denominator-only; their input transforms are derived from observed joints after masking",
        "synthetic": synthetic,
        "smoke": smoke,
        "result_scope": (
            "software validation only / NOT research results"
            if synthetic
            else "software smoke run / NOT primary research result"
            if smoke
            else "research result"
        ),
        "parameter_count": parameter_count,
        "inference_time_seconds": total_inference,
        "metrics_artifacts": artifacts,
        "figures_dir": str(figures_dir.resolve()),
        "figure_notes": figure_notes,
        "rgb_overlay_status": "NOT GENERATED: annotation-only local acquisition contains no source RGB frames; remote media size was not assumed.",
        "metric_integrity": "Predictions and targets were inverse-transformed to source pixels before bbox-diagonal NMJE/PCK; only synthetic_mask & original_valid_mask & sequence_mask were scored.",
        "error_analysis_overlap_status": (
            "evaluated from same-frame peer boxes"
            if same_frame_overlap is not None
            else "not evaluated: same-frame peer boxes unavailable"
        ),
    }
    _write_json(output_dir / "evaluation_index.json", index)
    return index


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, help="TrackID reconstruction dataset.npz")
    parser.add_argument("--checkpoint", type=Path, help="Best keypoint-reconstruction checkpoint")
    parser.add_argument("--baseline", choices=("temporal", "geometric", "hybrid"), help="Evaluate a deterministic fit-free baseline instead of a checkpoint")
    parser.add_argument("--representation", choices=("raw", "bbox_normalized", "root_centered", "scale_normalized", "normalized_angles"))
    parser.add_argument("--mask-ratios", type=float, nargs="+", default=list(SUPPORTED_MASK_RATIOS))
    parser.add_argument("--primary-mask-ratio", type=float, default=0.20)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--train-domain", help="Required metadata override only when an older checkpoint lacks train_domain")
    parser.add_argument("--test-domain", help="Indoor, Outdoor, Drone, or all (default: all)", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/logs/reconstruction_evaluation"))
    parser.add_argument("--figures-dir", type=Path, default=Path("outputs/figures/reconstruction_evaluation"))
    parser.add_argument("--run-name")
    parser.add_argument("--include-synthetic", action="store_true")
    parser.add_argument("--no-save-predictions", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        index = evaluate(args)
    except (ReconstructionEvaluationError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(f"Completed reconstruction evaluation: {index['model']} / {index['representation']}")
    print(f"Artifacts: {Path(index['metrics_artifacts'][0]).parent if index['metrics_artifacts'] else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
