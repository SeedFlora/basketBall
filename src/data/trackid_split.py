"""Deterministic, domain-aware sampling and grouped splits for TrackID3x3."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .trackid3x3 import TRACKID3X3_DOMAINS, PoseInstance, TrackIDSchemaError

DEFAULT_TARGETS = {"Indoor": 1000, "Outdoor": 1000, "Drone": 500}
SPLIT_NAMES = ("train", "val", "test")
MANIFEST_COLUMNS = (
    "sample_id",
    "domain",
    "video_id",
    "frame_index",
    "player_id",
    "player_number",
    "split",
    "split_group",
    "grouping_level",
    "bbox_source",
    "mot_track_id",
    "association_confidence",
    "temporal_reliable",
    "source_file",
)


def _stable_seed(seed: int, *tokens: object) -> int:
    payload = "\0".join((str(seed), *(str(token) for token in tokens))).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def exact_split_counts(
    count: int, ratios: Sequence[float] = (0.70, 0.15, 0.15)
) -> tuple[int, int, int]:
    """Largest-remainder split counts whose sum is exactly ``count``."""

    if count < 0:
        raise ValueError("count must be non-negative.")
    values = np.asarray(ratios, dtype=np.float64)
    if values.shape != (3,) or np.any(values < 0) or not np.isclose(values.sum(), 1.0):
        raise ValueError("ratios must be three non-negative values summing to one.")
    exact = values * count
    result = np.floor(exact).astype(int)
    remainder = count - int(result.sum())
    order = sorted(range(3), key=lambda index: (-(exact[index] - result[index]), index))
    for index in order[:remainder]:
        result[index] += 1
    return tuple(int(value) for value in result)  # type: ignore[return-value]


def split_group_for(instance: PoseInstance) -> tuple[str, str]:
    """Return the strongest reliable grouping unit specified per domain."""

    if instance.domain == "Indoor":
        if not instance.video_id:
            raise TrackIDSchemaError(f"Indoor instance has no video ID: {instance.sample_id}")
        return f"Indoor/video/{instance.video_id}", "video"
    if instance.domain in {"Outdoor", "Drone"}:
        return f"{instance.domain}/player/{instance.player_id}", "player"
    raise TrackIDSchemaError(f"Unsupported domain {instance.domain!r}.")


def assign_domain_splits(
    instances: Sequence[PoseInstance],
    *,
    seed: int = 42,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
) -> list[PoseInstance]:
    """Assign 70/15/15 groups independently inside each domain.

    Indoor is grouped by video.  Outdoor and Drone are grouped by player/track
    because Outdoor has one pose video and Drone lacks a reliable pose-video
    identifier.  No group can occur in more than one split.
    """

    group_members: dict[str, list[PoseInstance]] = defaultdict(list)
    group_levels: dict[str, str] = {}
    group_domains: dict[str, str] = {}
    for instance in instances:
        group, level = split_group_for(instance)
        group_members[group].append(instance)
        group_levels[group] = level
        group_domains[group] = instance.domain

    assignments: dict[str, str] = {}
    for domain in TRACKID3X3_DOMAINS:
        groups = sorted(group for group, value in group_domains.items() if value == domain)
        if not groups:
            continue
        rng = np.random.default_rng(_stable_seed(seed, "split", domain))
        shuffled = [groups[index] for index in rng.permutation(len(groups))]
        counts = exact_split_counts(len(groups), ratios)
        cursor = 0
        for split, split_count in zip(SPLIT_NAMES, counts, strict=True):
            for group in shuffled[cursor : cursor + split_count]:
                assignments[group] = split
            cursor += split_count
        if cursor != len(groups):
            raise AssertionError("Grouped split assignment did not consume every group.")

    result = [
        replace(
            instance,
            split_group=(group := split_group_for(instance)[0]),
            grouping_level=group_levels[group],
            split=assignments[group],
        )
        for instance in instances
    ]
    assert_group_disjoint(result)
    return result


def assert_group_disjoint(instances: Sequence[PoseInstance]) -> None:
    """Assert that every split group belongs to exactly one split."""

    observed: dict[str, set[str]] = defaultdict(set)
    for instance in instances:
        if not instance.split_group or instance.split not in SPLIT_NAMES:
            raise AssertionError(f"Missing split assignment: {instance.sample_id}")
        observed[instance.split_group].add(instance.split)
    leakage = {group: sorted(splits) for group, splits in observed.items() if len(splits) != 1}
    if leakage:
        raise AssertionError(f"Split-group leakage detected: {leakage}")


def apply_frame_stride(
    instances: Sequence[PoseInstance], *, stride: int = 3
) -> list[PoseInstance]:
    """Keep every ``stride``-th annotated frame before instance sampling.

    The stride is applied to sorted annotated-frame ordinals, not numeric frame
    modulo.  Indoor/Outdoor streams are per video.  Drone has no reliable video
    identity, so its global sorted annotation-frame list is used only for
    redundancy reduction and never to claim a temporal track.
    """

    if stride <= 0:
        raise ValueError("stride must be positive.")
    stream_frames: dict[tuple[str, str], set[int]] = defaultdict(set)
    for instance in instances:
        stream = (
            (instance.domain, instance.video_id or "")
            if instance.domain != "Drone"
            else ("Drone", "pose_json_collection")
        )
        stream_frames[stream].add(instance.frame_index)
    retained: dict[tuple[str, str], set[int]] = {
        stream: set(sorted(frames)[::stride]) for stream, frames in stream_frames.items()
    }
    return [
        instance
        for instance in instances
        if instance.frame_index
        in retained[
            (instance.domain, instance.video_id or "")
            if instance.domain != "Drone"
            else ("Drone", "pose_json_collection")
        ]
    ]


def _balanced_take(
    candidates: Sequence[PoseInstance], target: int, *, seed: int, domain: str
) -> list[PoseInstance]:
    if target < 0:
        raise ValueError("Subset target must be non-negative.")
    if len(candidates) <= target:
        return sorted(candidates, key=lambda item: item.sample_id)
    by_group: dict[str, list[PoseInstance]] = defaultdict(list)
    for instance in candidates:
        by_group[instance.split_group].append(instance)
    groups = sorted(by_group)
    group_rng = np.random.default_rng(_stable_seed(seed, "subset-groups", domain))
    groups = [groups[index] for index in group_rng.permutation(len(groups))]
    queues: dict[str, deque[PoseInstance]] = {}
    for group in groups:
        values = sorted(by_group[group], key=lambda item: item.sample_id)
        rng = np.random.default_rng(_stable_seed(seed, "subset-items", domain, group))
        queues[group] = deque(values[index] for index in rng.permutation(len(values)))

    selected: list[PoseInstance] = []
    while len(selected) < target:
        progress = False
        for group in groups:
            if queues[group] and len(selected) < target:
                selected.append(queues[group].popleft())
                progress = True
        if not progress:
            break
    if len(selected) != target:
        raise AssertionError(f"Could select only {len(selected)} of requested {target} {domain} instances.")
    return sorted(selected, key=lambda item: item.sample_id)


def select_domain_subset(
    instances: Sequence[PoseInstance],
    *,
    targets: Mapping[str, int] = DEFAULT_TARGETS,
    stride: int = 3,
    seed: int = 42,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
) -> tuple[list[PoseInstance], dict[str, Any]]:
    """Build an exact, group-balanced subset after annotation-frame stride."""

    assigned = assign_domain_splits(instances, seed=seed, ratios=ratios)
    candidates = apply_frame_stride(assigned, stride=stride)
    selected: list[PoseInstance] = []
    report: dict[str, Any] = {}
    for domain in TRACKID3X3_DOMAINS:
        domain_all = [item for item in assigned if item.domain == domain]
        domain_candidates = [item for item in candidates if item.domain == domain]
        target = int(targets.get(domain, len(domain_candidates)))
        chosen = _balanced_take(domain_candidates, target, seed=seed, domain=domain)
        selected.extend(chosen)
        group_splits = {
            group: next(item.split for item in domain_all if item.split_group == group)
            for group in sorted({item.split_group for item in domain_all})
        }
        selected_by_group = dict(Counter(item.split_group for item in chosen))
        report[domain] = {
            "instances_before_stride": len(domain_all),
            "unique_frames_before_stride": len(
                {(item.video_id, item.frame_index) for item in domain_all}
            ),
            "instances_after_stride": len(domain_candidates),
            "unique_frames_after_stride": len(
                {(item.video_id, item.frame_index) for item in domain_candidates}
            ),
            "target_instances": target,
            "selected_instances": len(chosen),
            "selected_unique_frames": len({(item.video_id, item.frame_index) for item in chosen}),
            "selected_by_split": dict(Counter(item.split for item in chosen)),
            "group_counts_by_split": dict(Counter(group_splits.values())),
            "groups": group_splits,
            "selected_by_group": selected_by_group,
            "selected_group_balance_range": (
                [min(selected_by_group.values()), max(selected_by_group.values())]
                if selected_by_group
                else [0, 0]
            ),
        }
    selected.sort(key=lambda item: (TRACKID3X3_DOMAINS.index(item.domain), item.sample_id))
    assert_group_disjoint(selected)
    ids = [item.sample_id for item in selected]
    if len(ids) != len(set(ids)):
        raise AssertionError("Subset contains duplicate sample IDs.")
    return selected, {
        "seed": seed,
        "stride": stride,
        "split_ratios": list(ratios),
        "selection_unit": "pose_instance",
        "domains": report,
        "total_selected_instances": len(selected),
    }


@dataclass(frozen=True, slots=True)
class PoseWindow:
    """A source-reliable, fixed-step sequence from one player track."""

    instances: tuple[PoseInstance, ...]
    frame_step: int

    def __post_init__(self) -> None:
        if not self.instances:
            raise ValueError("A PoseWindow cannot be empty.")
        group = self.instances[0].temporal_group_id
        if group is None or any(item.temporal_group_id != group for item in self.instances):
            raise TrackIDSchemaError("Temporal windows require one reliable video/player track.")
        if any(item.split != self.instances[0].split for item in self.instances):
            raise TrackIDSchemaError("Temporal window crosses a split boundary.")
        deltas = np.diff([item.frame_index for item in self.instances])
        if len(deltas) and not np.all(deltas == self.frame_step):
            raise TrackIDSchemaError(
                f"Temporal window is not contiguous at frame_step={self.frame_step}: {deltas.tolist()}"
            )

    @property
    def sample_id(self) -> str:
        first, last = self.instances[0], self.instances[-1]
        return f"{first.temporal_group_id}:frames_{first.frame_index}-{last.frame_index}"


def construct_temporal_windows(
    instances: Sequence[PoseInstance],
    *,
    length: int,
    frame_step: int = 1,
    window_hop: int = 1,
) -> tuple[list[PoseWindow], dict[str, int]]:
    """Construct windows only for reliable Indoor/Outdoor track identities."""

    if length < 2:
        raise ValueError("Temporal window length must be at least two; use static instances for T=1.")
    if frame_step <= 0 or window_hop <= 0:
        raise ValueError("frame_step and window_hop must be positive.")
    grouped: dict[str, list[PoseInstance]] = defaultdict(list)
    excluded = 0
    for instance in instances:
        group = instance.temporal_group_id
        if group is None:
            excluded += 1
        else:
            grouped[group].append(instance)
    windows: list[PoseWindow] = []
    for group in sorted(grouped):
        ordered = sorted(grouped[group], key=lambda item: item.frame_index)
        run: list[PoseInstance] = []
        runs: list[list[PoseInstance]] = []
        for item in ordered:
            if run and item.frame_index - run[-1].frame_index != frame_step:
                runs.append(run)
                run = []
            run.append(item)
        if run:
            runs.append(run)
        for values in runs:
            for start in range(0, len(values) - length + 1, window_hop):
                windows.append(PoseWindow(tuple(values[start : start + length]), frame_step))
    windows.sort(key=lambda item: item.sample_id)
    return windows, {
        "windows": len(windows),
        "reliable_track_groups": len(grouped),
        "excluded_unreliable_instances": excluded,
    }


def select_temporal_windows(
    instances: Sequence[PoseInstance],
    *,
    length: int = 16,
    frame_step: int = 3,
    target_observations: Mapping[str, int] = DEFAULT_TARGETS,
    seed: int = 42,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
) -> tuple[list[PoseWindow], dict[str, Any]]:
    """Select small, non-overlapping windows before random instance thinning.

    Full assigned Indoor/Outdoor streams are first regularly subsampled at
    ``frame_step``.  Contiguous windows are then created with a hop equal to the
    window length and balanced across split groups.  Domain caps count pose
    observations (``windows * length``), not windows.  Drone is excluded because
    its pose JSON has no reliable video identity.
    """

    if length < 2:
        raise ValueError("Use select_domain_subset for static T=1 data.")
    assigned = assign_domain_splits(instances, seed=seed, ratios=ratios)
    regular = apply_frame_stride(assigned, stride=frame_step)
    all_windows, base_report = construct_temporal_windows(
        regular,
        length=length,
        frame_step=frame_step,
        window_hop=length,
    )
    selected: list[PoseWindow] = []
    domain_report: dict[str, Any] = {}
    for domain in TRACKID3X3_DOMAINS:
        available = [window for window in all_windows if window.instances[0].domain == domain]
        cap = int(target_observations.get(domain, 0))
        maximum_windows = cap // length
        if domain == "Drone":
            maximum_windows = 0
        if len(available) <= maximum_windows:
            chosen = sorted(available, key=lambda window: window.sample_id)
        else:
            by_group: dict[str, list[PoseWindow]] = defaultdict(list)
            for window in available:
                by_group[window.instances[0].split_group].append(window)
            groups = sorted(by_group)
            rng = np.random.default_rng(_stable_seed(seed, "temporal-groups", domain, length))
            groups = [groups[index] for index in rng.permutation(len(groups))]
            queues: dict[str, deque[PoseWindow]] = {}
            for group in groups:
                values = sorted(by_group[group], key=lambda window: window.sample_id)
                item_rng = np.random.default_rng(
                    _stable_seed(seed, "temporal-items", domain, group, length)
                )
                queues[group] = deque(values[index] for index in item_rng.permutation(len(values)))
            chosen = []
            while len(chosen) < maximum_windows:
                progress = False
                for group in groups:
                    if queues[group] and len(chosen) < maximum_windows:
                        chosen.append(queues[group].popleft())
                        progress = True
                if not progress:
                    break
            chosen.sort(key=lambda window: window.sample_id)
        selected.extend(chosen)
        domain_report[domain] = {
            "available_windows": len(available),
            "selected_windows": len(chosen),
            "selected_pose_observations": len(chosen) * length,
            "observation_cap": cap,
            "selected_by_split": dict(
                Counter(window.instances[0].split for window in chosen)
            ),
            "exclusion": (
                "pose JSON has no reliable video identity"
                if domain == "Drone"
                else None
            ),
        }
    selected.sort(key=lambda window: window.sample_id)
    return selected, {
        "seed": seed,
        "sequence_length": length,
        "frame_step": frame_step,
        "window_hop_observations": length,
        "overlap": False,
        "domains": domain_report,
        "base": base_report,
    }


def manifest_rows(instances: Sequence[PoseInstance]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": item.sample_id,
            "domain": item.domain,
            "video_id": item.video_id or "",
            "frame_index": item.frame_index,
            "player_id": item.player_id,
            "player_number": item.player_number or "",
            "split": item.split,
            "split_group": item.split_group,
            "grouping_level": item.grouping_level,
            "bbox_source": item.bbox_source,
            "mot_track_id": item.mot_track_id or "",
            "association_confidence": (
                "" if not np.isfinite(item.association_confidence) else item.association_confidence
            ),
            "temporal_reliable": str(item.temporal_reliable).lower(),
            "source_file": item.source_file,
        }
        for item in instances
    ]


def write_subset_manifest(path: str | Path, instances: Sequence[PoseInstance]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows(instances))
    return output


def read_subset_ids(path: str | Path) -> list[str]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
            raise TrackIDSchemaError(f"Manifest has no sample_id column: {source}")
        ids = [str(row["sample_id"]) for row in reader]
    if not ids or any(not value for value in ids):
        raise TrackIDSchemaError(f"Manifest is empty or has blank sample IDs: {source}")
    if len(ids) != len(set(ids)):
        raise TrackIDSchemaError(f"Manifest contains duplicate sample IDs: {source}")
    return ids


def select_manifest_instances(
    all_instances: Sequence[PoseInstance], manifest_path: str | Path
) -> list[PoseInstance]:
    ids = read_subset_ids(manifest_path)
    lookup = {item.sample_id: item for item in all_instances}
    missing = [sample_id for sample_id in ids if sample_id not in lookup]
    if missing:
        raise TrackIDSchemaError(
            f"Manifest refers to {len(missing)} unknown sample IDs; first: {missing[:3]}"
        )
    # Recompute deterministic grouping/splits, then verify manifest metadata at
    # script level.  Returning manifest order preserves exact reproducibility.
    return [lookup[sample_id] for sample_id in ids]


def write_subset_metadata(path: str | Path, metadata: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return output


__all__ = [
    "DEFAULT_TARGETS",
    "MANIFEST_COLUMNS",
    "PoseWindow",
    "apply_frame_stride",
    "assert_group_disjoint",
    "assign_domain_splits",
    "construct_temporal_windows",
    "exact_split_counts",
    "manifest_rows",
    "read_subset_ids",
    "select_domain_subset",
    "select_manifest_instances",
    "select_temporal_windows",
    "split_group_for",
    "write_subset_manifest",
    "write_subset_metadata",
]
