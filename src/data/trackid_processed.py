"""Processed NPZ contract for TrackID3x3 masked pose reconstruction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .trackid3x3 import TRACKID10_EDGES, TRACKID10_JOINT_NAMES, PoseInstance
from .trackid_split import PoseWindow

TRACKID_PROCESSED_SCHEMA = "TrackID3x3.reconstruction.v1"
REQUIRED_TRACKID_KEYS = (
    "coords",
    "valid_mask",
    "input_mask",
    "target_mask",
    "sequence_mask",
    "bbox",
    "bbox_valid",
    "bbox_source",
    "sample_id",
    "domain",
    "video_id",
    "track_id",
    "player_number",
    "split",
    "split_group",
    "grouping_level",
    "frame_index",
    "image_size",
    "temporal_reliable",
    "rgb_available",
    "annotation_only",
    "synthetic",
    "joint_names",
    "edges",
)

_ALIASES = {
    "keypoints": "coords",
    "raw_keypoints": "coords",
    "target_keypoints": "coords",
    "original_valid_mask": "valid_mask",
    "track_valid_mask": "sequence_mask",
    "bboxes": "bbox",
    "bbox_xywh": "bbox",
    "bbox_sources": "bbox_source",
    "sample_ids": "sample_id",
    "domains": "domain",
    "video_ids": "video_id",
    "player_ids": "track_id",
    "player_numbers": "player_number",
    "track_ids": "track_id",
    "splits": "split",
    "split_groups": "split_group",
    "grouping_levels": "grouping_level",
    "frame_indices": "frame_index",
    "image_sizes": "image_size",
}

_SAMPLE_KEYS = {
    "coords",
    "valid_mask",
    "input_mask",
    "target_mask",
    "sequence_mask",
    "bbox",
    "bbox_valid",
    "bbox_source",
    "input_coords",
    "sample_id",
    "domain",
    "video_id",
    "track_id",
    "player_number",
    "split",
    "split_group",
    "grouping_level",
    "frame_index",
    "image_size",
    "mot_track_id",
    "association_confidence",
    "source_file",
    "temporal_reliable",
    "rgb_available",
    "annotation_only",
    "synthetic",
    "frame_step",
    "original_lengths",
    "raw",
    "raw_mask",
    "bbox_normalized",
    "bbox_normalized_mask",
    "bbox_inverse_offset",
    "bbox_inverse_scale",
    "bbox_transform_xywh",
    "root_centered",
    "root_centered_mask",
    "root_inverse_offset",
    "root_inverse_scale",
    "root_center_source",
    "scale_normalized",
    "scale_normalized_mask",
    "scale_inverse_offset",
    "scale_inverse_scale",
    "scale_center_source",
    "scale_source",
    "angles",
    "angles_mask",
}


class TrackIDProcessedError(ValueError):
    """Raised when a processed reconstruction archive violates its contract."""


def _stable_seed(seed: int, sample_id: str, frame_index: int) -> int:
    payload = f"{seed}\0{sample_id}\0{frame_index}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def deterministic_reconstruction_masks(
    valid_mask: np.ndarray,
    sample_ids: Sequence[str],
    *,
    missing_rate: float = 0.30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose deterministic held-out joints independently in every real frame."""

    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 3:
        raise ValueError(f"valid_mask must be [N,T,V], got {valid.shape}.")
    if len(sample_ids) != valid.shape[0]:
        raise ValueError("sample_ids length does not match valid_mask.")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be in [0,1).")
    target = np.zeros_like(valid)
    if missing_rate == 0:
        return valid.copy(), target
    for sample_index, sample_id in enumerate(sample_ids):
        for time_index in range(valid.shape[1]):
            eligible = np.flatnonzero(valid[sample_index, time_index])
            if eligible.size < 2:
                continue
            count = max(1, round(float(eligible.size) * missing_rate))
            count = min(count, int(eligible.size) - 1)
            rng = np.random.default_rng(_stable_seed(seed, str(sample_id), time_index))
            selected = eligible[rng.permutation(eligible.size)[:count]]
            target[sample_index, time_index, selected] = True
    return valid & ~target, target


def _as_windows(
    samples: Sequence[PoseInstance | PoseWindow],
) -> tuple[list[tuple[PoseInstance, ...]], list[str], np.ndarray]:
    if not samples:
        raise ValueError("Cannot preprocess an empty TrackID3x3 selection.")
    windows: list[tuple[PoseInstance, ...]] = []
    sample_ids: list[str] = []
    frame_steps: list[int] = []
    for sample in samples:
        if isinstance(sample, PoseWindow):
            window = sample.instances
            sample_id = sample.sample_id
            frame_step = sample.frame_step
        elif isinstance(sample, PoseInstance):
            window = (sample,)
            sample_id = sample.sample_id
            frame_step = 0
        else:
            raise TypeError(f"Unsupported processed sample type: {type(sample).__name__}")
        windows.append(window)
        sample_ids.append(sample_id)
        frame_steps.append(frame_step)
    lengths = {len(window) for window in windows}
    if len(lengths) != 1:
        raise ValueError(f"All processed windows must have one length, got {sorted(lengths)}.")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Processed sample IDs must be unique.")
    return windows, sample_ids, np.asarray(frame_steps, dtype=np.int64)


def build_trackid_processed(
    samples: Sequence[PoseInstance | PoseWindow],
    *,
    mask_rate: float = 0.30,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Convert static instances or reliable fixed-length windows to arrays."""

    from src.features.trackid_features import build_all_representations

    windows, sample_ids, frame_steps = _as_windows(samples)
    n_samples, sequence_length = len(windows), len(windows[0])
    coords = np.stack(
        [np.stack([item.keypoints for item in window]) for window in windows]
    ).astype(np.float32)
    valid = np.stack(
        [np.stack([item.valid_mask for item in window]) for window in windows]
    ).astype(bool)
    bbox = np.full((n_samples, sequence_length, 4), np.nan, dtype=np.float32)
    bbox_valid = np.zeros((n_samples, sequence_length), dtype=bool)
    for sample_index, window in enumerate(windows):
        for time_index, item in enumerate(window):
            if item.bbox_xywh is not None:
                bbox[sample_index, time_index] = item.bbox_xywh
                bbox_valid[sample_index, time_index] = True
    sequence_mask = np.ones((n_samples, sequence_length), dtype=bool)
    bbox_sources = np.asarray(
        [[item.bbox_source for item in window] for window in windows], dtype=str
    )
    input_mask, target_mask = deterministic_reconstruction_masks(
        valid, sample_ids, missing_rate=mask_rate, seed=seed
    )
    representations = build_all_representations(
        coords,
        valid,
        bboxes=bbox,
        bbox_valid=bbox_valid,
        bbox_sources=bbox_sources,
        observed_mask=valid,
    )

    first = [window[0] for window in windows]
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(TRACKID_PROCESSED_SCHEMA),
        "dataset_name": np.asarray("TrackID3x3"),
        "coordinate_system": np.asarray("source_image_pixels_xy"),
        "bbox_format": np.asarray("xywh"),
        "coords": coords,
        "valid_mask": valid,
        "input_mask": input_mask,
        "target_mask": target_mask,
        "input_coords": np.where(input_mask[..., None], coords, 0.0).astype(np.float32),
        "sequence_mask": sequence_mask,
        "bbox": bbox,
        "bbox_valid": bbox_valid,
        "bbox_source": bbox_sources,
        "sample_id": np.asarray(sample_ids, dtype=str),
        "domain": np.asarray([item.domain for item in first], dtype=str),
        "video_id": np.asarray([item.video_id or "" for item in first], dtype=str),
        "track_id": np.asarray([item.player_id for item in first], dtype=str),
        "player_number": np.asarray([item.player_number or "" for item in first], dtype=str),
        "split": np.asarray([item.split for item in first], dtype=str),
        "split_group": np.asarray([item.split_group for item in first], dtype=str),
        "grouping_level": np.asarray([item.grouping_level for item in first], dtype=str),
        "frame_index": np.asarray(
            [[item.frame_index for item in window] for window in windows], dtype=np.int64
        ),
        "image_size": np.asarray(
            [[item.image_size for item in window] for window in windows], dtype=np.int64
        ),
        "mot_track_id": np.asarray(
            [[item.mot_track_id or "" for item in window] for window in windows], dtype=str
        ),
        "association_confidence": np.asarray(
            [[item.association_confidence for item in window] for window in windows],
            dtype=np.float32,
        ),
        "source_file": np.asarray(
            [[item.source_file for item in window] for window in windows], dtype=str
        ),
        "temporal_reliable": np.asarray(
            [all(item.temporal_reliable for item in window) for window in windows], dtype=bool
        ),
        "rgb_available": np.zeros(n_samples, dtype=bool),
        "annotation_only": np.ones(n_samples, dtype=bool),
        "synthetic": np.zeros(n_samples, dtype=bool),
        "frame_step": frame_steps,
        "original_lengths": np.full(n_samples, sequence_length, dtype=np.int64),
        "joint_names": np.asarray(TRACKID10_JOINT_NAMES, dtype=str),
        "edges": TRACKID10_EDGES.astype(np.int64, copy=True),
        "angle_names": np.asarray(representations["angles"]["names"], dtype=str),
        "raw": np.asarray(representations["raw"]["targets"], dtype=np.float32),
        "raw_mask": np.asarray(representations["raw"]["target_valid_mask"], dtype=bool),
        "bbox_normalized": np.asarray(
            representations["bbox_normalized"]["targets"], dtype=np.float32
        ),
        "bbox_normalized_mask": np.asarray(
            representations["bbox_normalized"]["target_valid_mask"], dtype=bool
        ),
        "bbox_inverse_offset": np.asarray(
            representations["bbox_normalized"]["inverse_offset"], dtype=np.float32
        ),
        "bbox_inverse_scale": np.asarray(
            representations["bbox_normalized"]["inverse_scale"], dtype=np.float32
        ),
        "bbox_transform_xywh": np.asarray(
            representations["bbox_normalized"]["transform_bbox_xywh"], dtype=np.float32
        ),
        "root_centered": np.asarray(
            representations["root_centered"]["targets"], dtype=np.float32
        ),
        "root_centered_mask": np.asarray(
            representations["root_centered"]["target_valid_mask"], dtype=bool
        ),
        "root_inverse_offset": np.asarray(
            representations["root_centered"]["inverse_offset"], dtype=np.float32
        ),
        "root_inverse_scale": np.asarray(
            representations["root_centered"]["inverse_scale"], dtype=np.float32
        ),
        "root_center_source": np.asarray(
            representations["root_centered"]["center_source"], dtype=str
        ),
        "scale_normalized": np.asarray(
            representations["scale_normalized"]["targets"], dtype=np.float32
        ),
        "scale_normalized_mask": np.asarray(
            representations["scale_normalized"]["target_valid_mask"], dtype=bool
        ),
        "scale_inverse_offset": np.asarray(
            representations["scale_normalized"]["inverse_offset"], dtype=np.float32
        ),
        "scale_inverse_scale": np.asarray(
            representations["scale_normalized"]["inverse_scale"], dtype=np.float32
        ),
        "scale_center_source": np.asarray(
            representations["scale_normalized"]["center_source"], dtype=str
        ),
        "scale_source": np.asarray(
            representations["scale_normalized"]["scale_source"], dtype=str
        ),
        "angles": np.asarray(representations["angles"]["values"], dtype=np.float32),
        "angles_mask": np.asarray(representations["angles"]["mask"], dtype=bool),
    }
    validate_trackid_processed(arrays)
    return arrays


def add_trackid_aliases(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return a shallow mapping with singular/plural compatibility aliases."""

    result = dict(data)
    for alias, canonical in _ALIASES.items():
        if canonical in result:
            result.setdefault(alias, result[canonical])
        elif alias in result:
            result[canonical] = result[alias]
    return result


def validate_trackid_processed(data: Mapping[str, np.ndarray]) -> None:
    """Validate shapes, provenance, masks, identities, graph, and leakage."""

    values = add_trackid_aliases(data)
    missing = [key for key in REQUIRED_TRACKID_KEYS if key not in values]
    if missing:
        raise TrackIDProcessedError(f"TrackID processed dataset is missing keys: {missing}.")
    coords = np.asarray(values["coords"])
    valid = np.asarray(values["valid_mask"])
    if coords.ndim != 4 or coords.shape[-2:] != (10, 2):
        raise TrackIDProcessedError(f"coords must be float [N,T,10,2], got {coords.shape}.")
    if not np.issubdtype(coords.dtype, np.floating):
        raise TrackIDProcessedError("coords must have a floating dtype.")
    if valid.shape != coords.shape[:3] or valid.dtype != np.bool_:
        raise TrackIDProcessedError("valid_mask must be bool [N,T,10].")
    if not np.isfinite(coords).all() or np.any(coords[~valid] != 0):
        raise TrackIDProcessedError("coords must be finite and zero at invalid joints.")
    n_samples, sequence_length = coords.shape[:2]
    for key in (
        "sample_id",
        "domain",
        "video_id",
        "track_id",
        "player_number",
        "split",
        "split_group",
        "grouping_level",
    ):
        if np.asarray(values[key]).shape != (n_samples,):
            raise TrackIDProcessedError(f"{key} must have shape [{n_samples}].")
    sample_ids = np.asarray(values["sample_id"]).astype(str)
    if len(set(sample_ids.tolist())) != n_samples:
        raise TrackIDProcessedError("sample_id values must be unique.")
    domains = np.asarray(values["domain"]).astype(str)
    if not set(domains).issubset({"Indoor", "Outdoor", "Drone"}):
        raise TrackIDProcessedError(f"Unknown domain values: {sorted(set(domains))}.")
    splits = np.asarray(values["split"]).astype(str)
    if not set(splits).issubset({"train", "val", "test"}):
        raise TrackIDProcessedError(f"Unknown split values: {sorted(set(splits))}.")
    groups = np.asarray(values["split_group"]).astype(str)
    group_splits: dict[str, set[str]] = {}
    for group, split in zip(groups, splits, strict=True):
        group_splits.setdefault(group, set()).add(split)
    leakage = {group: sorted(items) for group, items in group_splits.items() if len(items) > 1}
    if leakage:
        raise TrackIDProcessedError(f"split_group leakage: {leakage}")

    for key in ("input_mask", "target_mask"):
        array = np.asarray(values[key])
        if array.shape != valid.shape or array.dtype != np.bool_:
            raise TrackIDProcessedError(f"{key} must be bool {valid.shape}.")
    input_mask = np.asarray(values["input_mask"])
    target_mask = np.asarray(values["target_mask"])
    if np.any(input_mask & target_mask) or np.any((input_mask | target_mask) & ~valid):
        raise TrackIDProcessedError("input/target masks must be disjoint subsets of valid_mask.")
    if not np.array_equal(input_mask | target_mask, valid):
        raise TrackIDProcessedError("input_mask and target_mask must partition valid_mask.")
    sequence_mask = np.asarray(values["sequence_mask"])
    if sequence_mask.shape != (n_samples, sequence_length) or sequence_mask.dtype != np.bool_:
        raise TrackIDProcessedError("sequence_mask must be bool [N,T].")

    boxes = np.asarray(values["bbox"])
    box_valid = np.asarray(values["bbox_valid"])
    sources = np.asarray(values["bbox_source"]).astype(str)
    if boxes.shape != (n_samples, sequence_length, 4):
        raise TrackIDProcessedError("bbox must be [N,T,4] source xywh.")
    if box_valid.shape != (n_samples, sequence_length) or box_valid.dtype != np.bool_:
        raise TrackIDProcessedError("bbox_valid must be bool [N,T].")
    if sources.shape != box_valid.shape:
        raise TrackIDProcessedError("bbox_source must be string [N,T].")
    if np.any(~np.isfinite(boxes[box_valid])) or np.any(boxes[..., 2:][box_valid] <= 0):
        raise TrackIDProcessedError("Valid bboxes must be finite with positive width/height.")
    expected_sources = np.where(domains[:, None] == "Drone", "pose_tight_fallback", "mot_xywh")
    expected_sources = np.broadcast_to(expected_sources, sources.shape)
    if np.any(box_valid & (sources != expected_sources)):
        raise TrackIDProcessedError("bbox_source conflicts with documented domain provenance.")

    if np.asarray(values["frame_index"]).shape != (n_samples, sequence_length):
        raise TrackIDProcessedError("frame_index must be [N,T].")
    if np.asarray(values["image_size"]).shape != (n_samples, sequence_length, 2):
        raise TrackIDProcessedError("image_size must be [N,T,2] as width,height.")
    names = np.asarray(values["joint_names"]).astype(str)
    if tuple(names.tolist()) != TRACKID10_JOINT_NAMES:
        raise TrackIDProcessedError("joint_names do not match the validated TrackID3x3 source order.")
    edges = np.asarray(values["edges"])
    if edges.ndim != 2 or edges.shape[0] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise TrackIDProcessedError("edges must be integer [2,E].")
    if edges.size and (edges.min() < 0 or edges.max() >= 10):
        raise TrackIDProcessedError("edges contain a joint index outside [0,10).")
    if "synthetic" in values:
        synthetic = np.asarray(values["synthetic"])
        if synthetic.shape != (n_samples,) or synthetic.dtype != np.bool_ or synthetic.any():
            raise TrackIDProcessedError(
                "TrackID3x3 source arrays require synthetic bool [N] with all values false."
            )
    for key, expected in (("rgb_available", False), ("annotation_only", True)):
        flag = np.asarray(values[key])
        if flag.shape != (n_samples,) or flag.dtype != np.bool_ or np.any(flag != expected):
            raise TrackIDProcessedError(f"{key} must be bool [{n_samples}] with value {expected}.")
    reliable = np.asarray(values["temporal_reliable"])
    if reliable.shape != (n_samples,) or reliable.dtype != np.bool_:
        raise TrackIDProcessedError("temporal_reliable must be bool [N].")
    if np.any((domains == "Drone") & reliable):
        raise TrackIDProcessedError("Drone samples cannot claim reliable temporal video identity.")

    for representation in ("raw", "bbox_normalized", "root_centered", "scale_normalized"):
        if representation in values:
            array = np.asarray(values[representation])
            mask = np.asarray(values.get(f"{representation}_mask"))
            if array.shape != coords.shape or mask.shape != valid.shape:
                raise TrackIDProcessedError(f"{representation} values/mask have invalid shapes.")
            if not np.isfinite(array).all() or np.any(array[~mask] != 0):
                raise TrackIDProcessedError(f"{representation} must be finite and zero when masked.")
    if "angles" in values:
        angles = np.asarray(values["angles"])
        angles_mask = np.asarray(values.get("angles_mask"))
        if angles.ndim != 3 or angles.shape[:2] != coords.shape[:2] or angles_mask.shape != angles.shape:
            raise TrackIDProcessedError("angles/angles_mask must be aligned [N,T,A].")
        if not np.isfinite(angles).all() or np.any(angles[~angles_mask] != 0):
            raise TrackIDProcessedError("angles must be finite and zero when masked.")


def _resolve_npz(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate / "dataset.npz" if candidate.is_dir() else candidate


def load_trackid_processed(
    path: str | Path = "data/processed/trackid3x3/dataset.npz",
    *,
    split: str | Sequence[str] | None = None,
    domains: str | Sequence[str] | None = None,
    aliases: bool = True,
) -> dict[str, np.ndarray]:
    """Safely load, validate, and optionally filter the reconstruction NPZ."""

    source = _resolve_npz(path)
    if not source.is_file():
        raise FileNotFoundError(f"Processed TrackID3x3 dataset not found: {source}")
    try:
        with np.load(source, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
    except (OSError, ValueError) as exc:
        raise TrackIDProcessedError(f"Could not safely load {source}: {exc}") from exc
    data = add_trackid_aliases(data)
    validate_trackid_processed(data)

    selector = np.ones(data["coords"].shape[0], dtype=bool)
    if split is not None:
        requested = {split} if isinstance(split, str) else {str(item) for item in split}
        unknown = requested - {"train", "val", "test"}
        if unknown:
            raise ValueError(f"Unknown split selection: {sorted(unknown)}")
        selector &= np.isin(data["split"].astype(str), sorted(requested))
    if domains is not None:
        requested_domains = {domains} if isinstance(domains, str) else {str(item) for item in domains}
        unknown = requested_domains - {"Indoor", "Outdoor", "Drone"}
        if unknown:
            raise ValueError(f"Unknown domain selection: {sorted(unknown)}")
        selector &= np.isin(data["domain"].astype(str), sorted(requested_domains))
    if not selector.all():
        if not selector.any():
            raise ValueError("The requested split/domain selection is empty.")
        indices = np.flatnonzero(selector)
        data = {
            key: (value[indices] if key in _SAMPLE_KEYS else value)
            for key, value in data.items()
            if key not in _ALIASES
        }
        validate_trackid_processed(data)
    if aliases:
        return add_trackid_aliases(data)
    return {key: value for key, value in data.items() if key not in _ALIASES}


def save_trackid_processed(
    output_dir: str | Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Validate and atomically write ``dataset.npz`` plus ``metadata.json``."""

    validate_trackid_processed(arrays)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset_path = destination / "dataset.npz"
    temporary = destination / "dataset.tmp.npz"
    np.savez_compressed(temporary, **dict(arrays))
    temporary.replace(dataset_path)
    metadata_path = destination / "metadata.json"
    temporary_metadata = destination / "metadata.tmp.json"
    with temporary_metadata.open("w", encoding="utf-8") as handle:
        json.dump(dict(metadata), handle, indent=2, sort_keys=True, default=str)
    temporary_metadata.replace(metadata_path)
    return dataset_path, metadata_path


try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


class TrackIDPoseDataset(Dataset):
    """Thin PyTorch adapter exposing pixels, masks, bbox, and domain metadata."""

    def __init__(
        self,
        source: str | Path | Mapping[str, np.ndarray],
        *,
        split: str | Sequence[str] | None = None,
        domains: str | Sequence[str] | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("TrackIDPoseDataset requires PyTorch.")
        if isinstance(source, (str, Path)):
            self.data = load_trackid_processed(source, split=split, domains=domains)
        else:
            self.data = add_trackid_aliases(source)
            validate_trackid_processed(self.data)
            if split is not None or domains is not None:
                raise ValueError("Filter mapping inputs before constructing TrackIDPoseDataset.")

    def __len__(self) -> int:
        return int(self.data["coords"].shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {
            "coordinates": torch.from_numpy(np.asarray(self.data["coords"][index])).float(),
            "keypoints": torch.from_numpy(np.asarray(self.data["coords"][index])).float(),
            "original_valid_mask": torch.from_numpy(
                np.asarray(self.data["valid_mask"][index])
            ).bool(),
            "bbox_xywh": torch.from_numpy(np.asarray(self.data["bbox"][index])).float(),
            "bbox_valid": torch.from_numpy(np.asarray(self.data["bbox_valid"][index])).bool(),
            "sequence_mask": torch.from_numpy(
                np.asarray(self.data["sequence_mask"][index])
            ).bool(),
            "sample_id": str(self.data["sample_id"][index]),
            "domain": str(self.data["domain"][index]),
            "video_id": str(self.data["video_id"][index]),
            "track_id": str(self.data["track_id"][index]),
            "player_number": str(self.data["player_number"][index]),
            "split": str(self.data["split"][index]),
            "split_group": str(self.data["split_group"][index]),
            "bbox_source": np.asarray(self.data["bbox_source"][index]).astype(str).tolist(),
            "temporal_reliable": bool(self.data["temporal_reliable"][index]),
            "rgb_available": bool(self.data["rgb_available"][index]),
            "synthetic": bool(self.data["synthetic"][index]),
        }
        return item


__all__ = [
    "REQUIRED_TRACKID_KEYS",
    "TRACKID_PROCESSED_SCHEMA",
    "TrackIDPoseDataset",
    "TrackIDProcessedError",
    "add_trackid_aliases",
    "build_trackid_processed",
    "deterministic_reconstruction_masks",
    "load_trackid_processed",
    "save_trackid_processed",
    "validate_trackid_processed",
]
