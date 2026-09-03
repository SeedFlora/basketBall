"""Validated processed-data loading and a PyTorch dataset adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .augmentation import AugmentationConfig, PoseAugmenter
from .splitting import assert_no_subject_leakage

REQUIRED_PROCESSED_KEYS = (
    "keypoints",
    "valid_mask",
    "labels",
    "sample_ids",
    "subject_ids",
    "splits",
    "joint_names",
    "edges",
)


def _resolve_npz(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate / "dataset.npz" if candidate.is_dir() else candidate


def validate_processed(data: Mapping[str, np.ndarray]) -> None:
    """Validate shape, label, graph, identity, and leakage invariants."""

    if "coords" in data or ("domain" in data and "bbox" in data and "labels" not in data):
        from .trackid_processed import validate_trackid_processed

        validate_trackid_processed(data)
        return

    missing = [key for key in REQUIRED_PROCESSED_KEYS if key not in data]
    if missing:
        raise ValueError(f"Processed dataset is missing required keys: {missing}.")
    keypoints = np.asarray(data["keypoints"])
    valid_mask = np.asarray(data["valid_mask"])
    if keypoints.ndim != 4:
        raise ValueError(f"keypoints must have [N,T,V,C], got {keypoints.shape}.")
    if keypoints.shape[-1] < 2:
        raise ValueError("keypoints require at least x/y coordinate channels.")
    if valid_mask.shape != keypoints.shape[:3]:
        raise ValueError(f"valid_mask {valid_mask.shape} does not match {keypoints.shape[:3]}.")
    n_samples = keypoints.shape[0]
    for key in ("labels", "sample_ids", "subject_ids", "splits"):
        if np.asarray(data[key]).shape != (n_samples,):
            raise ValueError(f"{key} must have shape [{n_samples}], got {np.asarray(data[key]).shape}.")
    labels = np.asarray(data["labels"])
    if not np.issubdtype(labels.dtype, np.integer) or not set(labels.tolist()).issubset({0, 1, 2}):
        raise ValueError("labels must be integer values in {0,1,2}.")
    sample_ids = np.asarray(data["sample_ids"]).astype(str)
    if len(set(sample_ids.tolist())) != n_samples:
        raise ValueError("Processed sample_ids are not unique.")
    splits = np.asarray(data["splits"]).astype(str)
    if not set(splits.tolist()).issubset({"train", "val", "test"}):
        raise ValueError(f"Unknown split names: {sorted(set(splits.tolist()))}.")
    assert_no_subject_leakage(
        np.asarray(data["subject_ids"]).astype(str).tolist(), splits.tolist()
    )
    joint_names = np.asarray(data["joint_names"])
    if joint_names.shape != (keypoints.shape[2],):
        raise ValueError(f"joint_names must have shape [{keypoints.shape[2]}], got {joint_names.shape}.")
    edges = np.asarray(data["edges"])
    if edges.ndim != 2 or edges.shape[0] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError(f"edges must be integer [2,E], got {edges.shape}, dtype={edges.dtype}.")
    if edges.size and (edges.min() < 0 or edges.max() >= keypoints.shape[2]):
        raise ValueError("edges contain a joint index outside [0,V).")
    if "confidence" in data and np.asarray(data["confidence"]).shape != valid_mask.shape:
        raise ValueError("confidence must match valid_mask shape [N,T,V].")
    if "original_lengths" in data and np.asarray(data["original_lengths"]).shape != (n_samples,):
        raise ValueError("original_lengths must have shape [N].")


def load_processed(path: str | Path = "data/processed/dataset.npz") -> dict[str, np.ndarray]:
    """Load a processed NPZ with pickle disabled and enforce its contract."""

    npz_path = _resolve_npz(path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {npz_path}")
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load safe processed dataset {npz_path}: {exc}") from exc
    if "coords" in data or ("domain" in data and "bbox" in data and "labels" not in data):
        from .trackid_processed import add_trackid_aliases

        data = add_trackid_aliases(data)
    validate_processed(data)
    return data


def load_metadata(path: str | Path = "data/processed") -> dict[str, Any]:
    """Load adjacent preprocessing metadata when present."""

    candidate = Path(path)
    metadata_path = candidate / "metadata.json" if candidate.is_dir() else candidate.with_name("metadata.json")
    if not metadata_path.is_file():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - project dependency, useful for schema-only installs
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


class PoseDataset(Dataset):
    """PyTorch dataset over the compact NPZ contract.

    Augmentation is rejected for any non-training selection.  Randomness is a
    pure function of ``(seed, epoch, original_sample_index)``, so results are
    stable across DataLoader worker count and fetch order.  Call ``set_epoch`` at
    each training epoch to receive a new deterministic augmentation.
    """

    def __init__(
        self,
        source: str | Path | Mapping[str, np.ndarray] = "data/processed/dataset.npz",
        *,
        split: str | Sequence[str] | None = None,
        representation: str = "normalized_angles",
        augment: bool = False,
        augmentation_config: AugmentationConfig | Mapping[str, Any] | None = None,
        seed: int = 42,
        graph: bool = False,
    ) -> None:
        if torch is None:
            raise ImportError("PoseDataset requires PyTorch.")
        self.data = load_processed(source) if isinstance(source, (str, Path)) else dict(source)
        validate_processed(self.data)
        if "labels" not in self.data:
            raise ValueError(
                "PoseDataset is the legacy action-classification adapter; use "
                "TrackIDPoseDataset for TrackID3x3 reconstruction data."
            )
        all_splits = np.asarray(self.data["splits"]).astype(str)
        if split is None:
            selected_splits: set[str] | None = None
            selector = np.ones(len(all_splits), dtype=bool)
        else:
            selected_splits = {split} if isinstance(split, str) else {str(item) for item in split}
            unknown = selected_splits - {"train", "val", "test"}
            if unknown:
                raise ValueError(f"Unknown requested splits: {sorted(unknown)}.")
            selector = np.isin(all_splits, sorted(selected_splits))
        self.indices = np.flatnonzero(selector)
        if self.indices.size == 0:
            raise ValueError(f"No samples selected for split={split!r}.")
        if augment and set(all_splits[self.indices].tolist()) != {"train"}:
            raise ValueError("Pose augmentation is permitted only for a train-only dataset selection.")
        self.representation = representation
        self.graph = bool(graph)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0
        self.augmenter = PoseAugmenter(augmentation_config, seed=seed)
        first = self._build_features(int(self.indices[0]), apply_augmentation=False)
        self.feature_dim = int(first[0].shape[-1])
        self.feature_names = first[2]
        self.num_joints = int(np.asarray(self.data["keypoints"]).shape[2])

    def __len__(self) -> int:
        return int(self.indices.size)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        self.epoch = int(epoch)

    def _build_features(
        self, source_index: int, *, apply_augmentation: bool
    ) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
        from src.features.skeleton_features import (
            build_graph_representation,
            build_representation,
        )

        points = np.asarray(self.data["keypoints"][source_index], dtype=np.float32).copy()
        mask = np.asarray(self.data["valid_mask"][source_index], dtype=bool).copy()
        if apply_augmentation:
            seed_sequence = np.random.SeedSequence([self.seed, self.epoch, source_index])
            points, mask = self.augmenter(points, mask, rng=np.random.default_rng(seed_sequence))
        builder = build_graph_representation if self.graph else build_representation
        features, feature_mask, names = builder(
            points, mask, self.representation, self.data["joint_names"]
        )
        return features, feature_mask, names, points, mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = int(self.indices[index])
        features, feature_mask, _, points, mask = self._build_features(
            source_index, apply_augmentation=self.augment
        )
        confidence = (
            np.asarray(self.data["confidence"][source_index], dtype=np.float32).copy()
            if "confidence" in self.data
            else mask.astype(np.float32)
        )
        confidence[~mask] = 0.0
        item: dict[str, Any] = {
            "features": torch.from_numpy(np.ascontiguousarray(features)).float(),
            "feature_mask": torch.from_numpy(np.ascontiguousarray(feature_mask)).bool(),
            "keypoints": torch.from_numpy(np.ascontiguousarray(points)).float(),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(mask)).bool(),
            "confidence": torch.from_numpy(np.ascontiguousarray(confidence)).float(),
            "label": torch.tensor(int(self.data["labels"][source_index]), dtype=torch.long),
            "sample_id": str(self.data["sample_ids"][source_index]),
            "subject_id": str(self.data["subject_ids"][source_index]),
            "split": str(self.data["splits"][source_index]),
        }
        if "original_lengths" in self.data:
            item["original_length"] = torch.tensor(
                int(self.data["original_lengths"][source_index]), dtype=torch.long
            )
        return item
