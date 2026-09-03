"""Non-learned masked-keypoint reconstruction baselines."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from .reconstruction import TRACKID3X3_EDGES


def _validate(coordinates: Tensor, observed_mask: Tensor) -> tuple[Tensor, Tensor]:
    values = torch.as_tensor(coordinates)
    mask = torch.as_tensor(observed_mask, dtype=torch.bool, device=values.device)
    if values.ndim not in {3, 4} or mask.shape != values.shape[:-1]:
        raise ValueError("Expected coordinates [T,V,C] or [B,T,V,C] and matching mask")
    return values, mask


def temporal_interpolation_baseline(
    coordinates: Tensor,
    observed_mask: Tensor,
    track_valid_mask: Tensor | None = None,
) -> Tensor:
    """Linearly interpolate each landmark track, using nearest values at edges."""

    values, mask = _validate(coordinates, observed_mask)
    single = values.ndim == 3
    if single:
        values = values.unsqueeze(0)
        mask = mask.unsqueeze(0)
    if track_valid_mask is not None:
        valid_window = torch.as_tensor(track_valid_mask, dtype=torch.bool, device=values.device)
        if single and valid_window.ndim == 1:
            valid_window = valid_window.unsqueeze(0)
        if valid_window.shape != values.shape[:2]:
            raise ValueError(
                f"track_valid_mask must be {tuple(values.shape[:2])}, got {tuple(valid_window.shape)}"
            )
        if not bool(valid_window.all()):
            raise ValueError("Temporal interpolation requires a continuous valid track window")
    output = torch.where(mask.unsqueeze(-1), values, torch.zeros_like(values)).clone()
    frames = values.shape[1]
    timeline = torch.arange(frames, device=values.device)
    for batch in range(values.shape[0]):
        for joint in range(values.shape[2]):
            known = timeline[mask[batch, :, joint]]
            if known.numel() == 0:
                continue
            missing = timeline[~mask[batch, :, joint]]
            if missing.numel() == 0:
                continue
            right_positions = torch.searchsorted(known, missing).clamp(max=known.numel() - 1)
            left_positions = (right_positions - 1).clamp(min=0)
            right = known[right_positions]
            left = known[left_positions]
            # searchsorted gives zero for values before the first known frame.
            before = missing <= known[0]
            after = missing >= known[-1]
            left = torch.where(before, known[0], left)
            right = torch.where(before, known[0], right)
            left = torch.where(after, known[-1], left)
            right = torch.where(after, known[-1], right)
            denominator = (right - left).clamp_min(1).to(values.dtype)
            alpha = ((missing - left).to(values.dtype) / denominator).unsqueeze(-1)
            interpolated = output[batch, left, joint] * (1 - alpha) + output[batch, right, joint] * alpha
            output[batch, missing, joint] = interpolated
    return output[0] if single else output


def geometric_neighbor_baseline(
    coordinates: Tensor,
    observed_mask: Tensor,
    edges: Iterable[tuple[int, int]] = TRACKID3X3_EDGES,
) -> Tensor:
    """Fill absent landmarks from the mean of observed anatomical neighbours."""

    values, mask = _validate(coordinates, observed_mask)
    single = values.ndim == 3
    if single:
        values = values.unsqueeze(0)
        mask = mask.unsqueeze(0)
    output = torch.where(mask.unsqueeze(-1), values, torch.zeros_like(values)).clone()
    available = mask.clone()
    neighbours: list[list[int]] = [[] for _ in range(values.shape[2])]
    for source, target in edges:
        if not (0 <= source < len(neighbours) and 0 <= target < len(neighbours)):
            raise ValueError("Anatomical edge index lies outside the coordinate tensor")
        neighbours[source].append(target)
        neighbours[target].append(source)
    # Iteration lets information traverse a short missing segment while retaining
    # a transparent, deterministic rule-based baseline.
    for _ in range(values.shape[2]):
        changed = False
        for joint, adjacent in enumerate(neighbours):
            if not adjacent:
                continue
            neighbour_mask = available[:, :, adjacent]
            fillable = ~available[:, :, joint] & neighbour_mask.any(dim=-1)
            if not bool(fillable.any()):
                continue
            weights = neighbour_mask.unsqueeze(-1).to(output.dtype)
            estimate = (output[:, :, adjacent] * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1)
            output[:, :, joint] = torch.where(fillable.unsqueeze(-1), estimate, output[:, :, joint])
            available[:, :, joint] |= fillable
            changed = True
        if not changed:
            break
    return output[0] if single else output


def reconstruction_baseline(
    coordinates: Tensor,
    observed_mask: Tensor,
    method: str = "temporal",
    edges: Iterable[tuple[int, int]] = TRACKID3X3_EDGES,
    track_valid_mask: Tensor | None = None,
) -> Tensor:
    """Dispatch temporal, geometric, or temporal-then-geometric baselines."""

    token = method.strip().lower().replace("-", "_")
    if token in {"temporal", "linear", "interpolation"}:
        return temporal_interpolation_baseline(coordinates, observed_mask, track_valid_mask)
    if token in {"geometric", "neighbor", "neighbour"}:
        return geometric_neighbor_baseline(coordinates, observed_mask, edges)
    if token in {"hybrid", "temporal_geometric"}:
        temporal = temporal_interpolation_baseline(
            coordinates, observed_mask, track_valid_mask
        )
        # A track with no temporal observation remains unavailable for geometric fill.
        available = observed_mask.clone()
        if observed_mask.ndim == 2:
            track_seen = observed_mask.any(dim=0)
            available |= track_seen.unsqueeze(0).expand_as(observed_mask)
        else:
            track_seen = observed_mask.any(dim=1)
            available |= track_seen.unsqueeze(1).expand_as(observed_mask)
        return geometric_neighbor_baseline(temporal, available, edges)
    raise ValueError("method must be temporal, geometric, or hybrid")


__all__ = [
    "geometric_neighbor_baseline",
    "reconstruction_baseline",
    "temporal_interpolation_baseline",
]
