"""Conservative, training-only pose augmentation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """Small perturbations that preserve action semantics."""

    jitter_std: float = 0.005
    scale_range: tuple[float, float] = (0.95, 1.05)
    translation_std: float = 0.01
    frame_dropout: float = 0.05
    joint_dropout: float = 0.02

    def __post_init__(self) -> None:
        if self.jitter_std < 0 or self.translation_std < 0:
            raise ValueError("jitter_std and translation_std must be non-negative.")
        if len(self.scale_range) != 2 or self.scale_range[0] <= 0 or self.scale_range[1] < self.scale_range[0]:
            raise ValueError("scale_range must be a positive (minimum, maximum) pair.")
        if not 0 <= self.frame_dropout < 1 or not 0 <= self.joint_dropout < 1:
            raise ValueError("dropout probabilities must be in [0,1).")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> AugmentationConfig:
        if value is None:
            return cls()
        allowed = {item.name for item in fields(cls)}
        kwargs = {key: item for key, item in value.items() if key in allowed}
        if "scale_range" in kwargs:
            kwargs["scale_range"] = tuple(float(item) for item in kwargs["scale_range"])
        return cls(**kwargs)


class PoseAugmenter:
    """Apply jitter, global scale/translation, and sparse dropout.

    Horizontal flipping is intentionally omitted because it requires an exact
    left/right joint mapping and can change handed action patterns.
    """

    def __init__(
        self,
        config: AugmentationConfig | Mapping[str, Any] | None = None,
        *,
        seed: int = 42,
    ) -> None:
        self.config = (
            config if isinstance(config, AugmentationConfig) else AugmentationConfig.from_mapping(config)
        )
        self.seed = int(seed)

    def __call__(
        self,
        keypoints: np.ndarray,
        valid_mask: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(keypoints, dtype=np.float32).copy()
        mask = np.asarray(valid_mask, dtype=bool).copy()
        if points.ndim != 3 or mask.shape != points.shape[:2]:
            raise ValueError(f"Expected [T,V,C] and [T,V], got {points.shape}, {mask.shape}.")
        generator = rng if rng is not None else np.random.default_rng(self.seed)
        coordinate_dims = min(3, points.shape[-1])
        valid_coordinates = mask[..., None]

        scale = float(generator.uniform(*self.config.scale_range))
        points[..., :coordinate_dims] *= scale
        if self.config.translation_std > 0:
            translation = generator.normal(
                0.0, self.config.translation_std, size=(1, 1, coordinate_dims)
            ).astype(np.float32)
            points[..., :coordinate_dims] += translation * valid_coordinates
        if self.config.jitter_std > 0:
            jitter = generator.normal(
                0.0, self.config.jitter_std, size=points[..., :coordinate_dims].shape
            ).astype(np.float32)
            points[..., :coordinate_dims] += jitter * valid_coordinates

        if self.config.frame_dropout > 0:
            dropped_frames = generator.random(points.shape[0]) < self.config.frame_dropout
            if np.all(dropped_frames):
                dropped_frames[int(generator.integers(0, points.shape[0]))] = False
            mask[dropped_frames, :] = False
        if self.config.joint_dropout > 0:
            dropped_joints = generator.random(mask.shape) < self.config.joint_dropout
            mask &= ~dropped_joints
        points[~mask] = 0.0
        return points.astype(np.float32, copy=False), mask


def augment_pose(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    *,
    config: AugmentationConfig | Mapping[str, Any] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Functional deterministic augmentation helper."""

    return PoseAugmenter(config, seed=seed)(keypoints, valid_mask)
