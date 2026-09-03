"""Deterministic synthetic masking for keypoint reconstruction."""

from __future__ import annotations

import hashlib
from collections.abc import Hashable, Sequence

import numpy as np
import torch
from torch import Tensor

SUPPORTED_MASK_RATIOS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4)


def validate_mask_ratio(mask_ratio: float) -> float:
    ratio = float(mask_ratio)
    if not any(abs(ratio - allowed) < 1e-9 for allowed in SUPPORTED_MASK_RATIOS):
        raise ValueError(
            f"mask_ratio must be one of {SUPPORTED_MASK_RATIOS}, got {mask_ratio!r}"
        )
    return ratio


def _instance_seed(seed: int, instance_id: Hashable) -> int:
    payload = f"{int(seed)}\0{instance_id!s}".encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"pose-mask").digest()
    return int.from_bytes(digest, "little", signed=False)


def deterministic_reconstruction_mask(
    original_valid_mask: Tensor | np.ndarray,
    mask_ratio: float,
    *,
    seed: int = 42,
    instance_ids: Sequence[Hashable] | Hashable | None = None,
) -> Tensor:
    """Select an exact deterministic fraction of valid keypoints per instance.

    ``True`` means synthetically hidden and therefore eligible for reconstruction
    scoring. Originally missing entries are never selected. Supplying stable
    ``instance_ids`` makes results invariant to batch order and worker count.
    """

    ratio = validate_mask_ratio(mask_ratio)
    source = torch.as_tensor(original_valid_mask, dtype=torch.bool)
    single = source.ndim == 2
    if single:
        source = source.unsqueeze(0)
    if source.ndim != 3:
        raise ValueError(
            f"original_valid_mask must be [T,V] or [B,T,V], got {tuple(source.shape)}"
        )
    batch = source.shape[0]
    if instance_ids is None:
        identities: list[Hashable] = list(range(batch))
    elif isinstance(instance_ids, (str, bytes)) or not isinstance(instance_ids, Sequence):
        if batch != 1:
            raise ValueError("A scalar instance_id is valid only for one instance")
        identities = [instance_ids]
    else:
        identities = list(instance_ids)
        if len(identities) != batch:
            raise ValueError(f"Expected {batch} instance_ids, got {len(identities)}")

    source_cpu = source.detach().cpu().numpy()
    selected = np.zeros_like(source_cpu, dtype=bool)
    for index, identity in enumerate(identities):
        eligible = np.flatnonzero(source_cpu[index].reshape(-1))
        if eligible.size == 0:
            continue
        count = int(np.floor(eligible.size * ratio + 0.5))
        count = max(1, min(int(eligible.size), count))
        generator = np.random.default_rng(_instance_seed(seed, identity))
        chosen = generator.choice(eligible, size=count, replace=False)
        selected[index].reshape(-1)[chosen] = True
    result = torch.from_numpy(selected).to(device=source.device)
    return result[0] if single else result


def apply_reconstruction_mask(
    coordinates: Tensor | np.ndarray,
    original_valid_mask: Tensor | np.ndarray,
    synthetic_mask: Tensor | np.ndarray,
    *,
    fill_value: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Return masked coordinates and the remaining observed mask."""

    values = torch.as_tensor(coordinates)
    original = torch.as_tensor(original_valid_mask, dtype=torch.bool, device=values.device)
    synthetic = torch.as_tensor(synthetic_mask, dtype=torch.bool, device=values.device)
    if original.shape != values.shape[:-1] or synthetic.shape != original.shape:
        raise ValueError("coordinates, original_valid_mask, and synthetic_mask shapes do not align")
    if bool((synthetic & ~original).any()):
        raise ValueError("synthetic_mask may select only originally valid keypoints")
    observed = original & ~synthetic
    masked = torch.where(
        observed.unsqueeze(-1),
        torch.nan_to_num(values),
        torch.full_like(values, float(fill_value)),
    )
    return masked, observed


class ReconstructionMasker:
    """Reusable deterministic masker configured for one experiment ratio/seed."""

    def __init__(self, mask_ratio: float, seed: int = 42, fill_value: float = 0.0) -> None:
        self.mask_ratio = validate_mask_ratio(mask_ratio)
        self.seed = int(seed)
        self.fill_value = float(fill_value)

    def __call__(
        self,
        coordinates: Tensor | np.ndarray,
        original_valid_mask: Tensor | np.ndarray,
        *,
        instance_ids: Sequence[Hashable] | Hashable | None = None,
    ) -> dict[str, Tensor]:
        original = torch.as_tensor(original_valid_mask, dtype=torch.bool)
        synthetic = deterministic_reconstruction_mask(
            original, self.mask_ratio, seed=self.seed, instance_ids=instance_ids
        )
        inputs, observed = apply_reconstruction_mask(
            coordinates, original, synthetic, fill_value=self.fill_value
        )
        return {
            "inputs": inputs,
            "targets": torch.as_tensor(coordinates).clone(),
            "original_valid_mask": original,
            "synthetic_mask": synthetic,
            "observed_mask": observed,
        }


generate_synthetic_mask = deterministic_reconstruction_mask


__all__ = [
    "SUPPORTED_MASK_RATIOS",
    "ReconstructionMasker",
    "apply_reconstruction_mask",
    "deterministic_reconstruction_mask",
    "generate_synthetic_mask",
    "validate_mask_ratio",
]
