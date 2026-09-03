"""Pure preprocessing helpers used by the CLI and synthetic tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.features.skeleton_features import joint_names_and_edges
from src.features.temporal_features import temporal_resample

from .loader import PoseSample
from .processed import validate_processed


def preprocess_samples(
    samples: Sequence[PoseSample],
    split_assignment: Mapping[str, str],
    *,
    sequence_length: int = 32,
    coordinate_dims: int = 2,
) -> dict[str, np.ndarray]:
    """Temporal-standardise canonical samples into the compact NPZ arrays."""

    if not samples:
        raise ValueError("Cannot preprocess an empty sample collection.")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if coordinate_dims < 2:
        raise ValueError("coordinate_dims must be at least 2.")
    joint_counts = {sample.keypoints.shape[1] for sample in samples}
    if len(joint_counts) != 1:
        raise ValueError(
            f"Mixed joint layouts {sorted(joint_counts)} cannot share one dense processed tensor."
        )
    insufficient = [
        sample.sample_id for sample in samples if sample.keypoints.shape[2] < coordinate_dims
    ]
    if insufficient:
        raise ValueError(
            f"{len(insufficient)} samples have fewer than {coordinate_dims} coordinate channels; "
            f"examples: {insufficient[:5]}."
        )
    unknown_subjects = sorted(
        {sample.subject_id for sample in samples if sample.subject_id not in split_assignment}
    )
    if unknown_subjects:
        raise ValueError(f"Samples have no split assignment for subjects: {unknown_subjects[:10]}.")

    keypoint_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    confidence_rows: list[np.ndarray] = []
    confidence_available: list[bool] = []
    original_lengths: list[int] = []
    for sample in samples:
        original_lengths.append(int(sample.keypoints.shape[0]))
        points = sample.keypoints[..., :coordinate_dims]
        resampled, mask, confidence = temporal_resample(
            points, sample.valid_mask, sequence_length, sample.confidence
        )
        keypoint_rows.append(resampled)
        mask_rows.append(mask)
        confidence_available.append(confidence is not None)
        confidence_rows.append(
            confidence if confidence is not None else np.zeros(mask.shape, dtype=np.float32)
        )
    num_joints = next(iter(joint_counts))
    joint_names, edges, _ = joint_names_and_edges(num_joints)
    data = {
        "keypoints": np.stack(keypoint_rows).astype(np.float32, copy=False),
        "confidence": np.stack(confidence_rows).astype(np.float32, copy=False),
        "confidence_available": np.asarray(confidence_available, dtype=bool),
        "valid_mask": np.stack(mask_rows).astype(bool, copy=False),
        "labels": np.asarray([sample.label for sample in samples], dtype=np.int64),
        "sample_ids": np.asarray([sample.sample_id for sample in samples], dtype=np.str_),
        "subject_ids": np.asarray([sample.subject_id for sample in samples], dtype=np.str_),
        "splits": np.asarray(
            [split_assignment[sample.subject_id] for sample in samples], dtype=np.str_
        ),
        "joint_names": joint_names.astype(np.str_, copy=False),
        "edges": edges.astype(np.int64, copy=False),
        "original_lengths": np.asarray(original_lengths, dtype=np.int64),
    }
    validate_processed(data)
    return data
