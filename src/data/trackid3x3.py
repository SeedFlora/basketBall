"""Strict adapter for the public TrackID3x3 pose annotations.

The source repository has two JSON layouts: Indoor/Outdoor store one video per
JSON file and map ``frame_<zero based index>`` to player poses, while Drone
stores one image per JSON file.  Both layouts contain exactly ten ordered 2-D
points.  This module validates those facts instead of guessing alternate
layouts or joint semantics.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

TRACKID3X3_DOMAINS = ("Indoor", "Outdoor", "Drone")
TRACKID10_JOINT_NAMES = (
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "center",
    "left_ankle",
    "right_ankle",
)

# The paper describes the centre as the midpoint of the two hips.  It is not
# represented as a COCO keypoint and therefore deliberately has no COCO index.
TRACKID10_TO_COCO17 = (0, 5, 6, 7, 8, 9, 10, None, 15, 16)
TRACKID10_EDGES = np.asarray(
    [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (3, 5),
        (2, 4),
        (4, 6),
        (1, 7),
        (2, 7),
        (7, 8),
        (7, 9),
    ],
    dtype=np.int64,
).T

_FRAME_RE = re.compile(r"frame_(\d+)\Z")
_PLAYER_NUMBER_RE = re.compile(r"_(\d+)\Z")


class TrackIDSchemaError(ValueError):
    """Raised when an annotation cannot be adapted without inventing facts."""


@dataclass(frozen=True, slots=True)
class PoseInstance:
    """One validated TrackID3x3 player pose in one annotated frame.

    ``frame_index`` follows the zero-based index used by pose JSON keys.  The
    authoritative missing-data signal is ``valid_mask``; invalid coordinates
    are stored as zero.  Bboxes are source ``xywh`` values in image pixels.
    """

    keypoints: np.ndarray
    valid_mask: np.ndarray
    domain: str
    frame_index: int
    player_id: str
    image_size: tuple[int, int]
    sample_id: str
    video_id: str | None = None
    source_file: str = ""
    bbox_xywh: np.ndarray | None = None
    bbox_source: str = "unavailable"
    mot_track_id: str | None = None
    association_confidence: float = math.nan
    temporal_reliable: bool = False
    split_group: str = ""
    grouping_level: str = ""
    split: str = ""

    def __post_init__(self) -> None:
        points = np.asarray(self.keypoints, dtype=np.float32)
        mask = np.asarray(self.valid_mask, dtype=bool)
        if points.shape != (len(TRACKID10_JOINT_NAMES), 2):
            raise TrackIDSchemaError(
                f"PoseInstance keypoints must have shape (10, 2), got {points.shape}."
            )
        if mask.shape != (len(TRACKID10_JOINT_NAMES),):
            raise TrackIDSchemaError(
                f"PoseInstance valid_mask must have shape (10,), got {mask.shape}."
            )
        if self.domain not in TRACKID3X3_DOMAINS:
            raise TrackIDSchemaError(f"Unknown TrackID3x3 domain {self.domain!r}.")
        if self.frame_index < 0:
            raise TrackIDSchemaError("frame_index must be non-negative.")
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise TrackIDSchemaError(f"Invalid image resolution {self.image_size!r}.")
        if not self.player_id or not self.sample_id:
            raise TrackIDSchemaError("player_id and sample_id must be non-empty.")

        points = points.copy()
        points[~mask] = 0.0
        points.setflags(write=False)
        mask = mask.copy()
        mask.setflags(write=False)
        object.__setattr__(self, "keypoints", points)
        object.__setattr__(self, "valid_mask", mask)

        if self.bbox_xywh is not None:
            bbox = np.asarray(self.bbox_xywh, dtype=np.float32)
            if bbox.shape != (4,) or not np.isfinite(bbox).all():
                raise TrackIDSchemaError(f"Invalid bbox shape/value for {self.sample_id}.")
            if bbox[2] <= 0 or bbox[3] <= 0:
                raise TrackIDSchemaError(f"Non-positive bbox extent for {self.sample_id}.")
            bbox = bbox.copy()
            bbox.setflags(write=False)
            object.__setattr__(self, "bbox_xywh", bbox)

    @property
    def temporal_group_id(self) -> str | None:
        """Return a track key only when source metadata supports sequencing."""

        if not self.temporal_reliable or not self.video_id:
            return None
        return f"{self.domain}/{self.video_id}/{self.player_id}"

    @property
    def player_number(self) -> str | None:
        """Return a numeric jersey suffix without treating it as globally unique."""

        match = _PLAYER_NUMBER_RE.search(self.player_id)
        return match.group(1) if match else None


@dataclass(frozen=True, slots=True)
class TrackIDLoadResult:
    """Pose instances plus source/association diagnostics."""

    instances: tuple[PoseInstance, ...]
    ground_truth_root: Path
    diagnostics: Mapping[str, Any]


def _is_ground_truth_root(path: Path) -> bool:
    return all((path / domain / "pose").is_dir() for domain in TRACKID3X3_DOMAINS)


def resolve_trackid_ground_truth(path: str | Path) -> Path:
    """Resolve a repo, ``ground_truth`` directory, or containing directory.

    Recursive discovery is accepted only when it yields one unambiguous tree
    with all three named domains.
    """

    requested = Path(path).expanduser().resolve()
    direct = [
        requested,
        requested / "ground_truth",
        requested / "TrackID3x3" / "ground_truth",
        requested / "external" / "TrackID3x3" / "ground_truth",
    ]
    matches = {candidate.resolve() for candidate in direct if _is_ground_truth_root(candidate)}
    if not matches and requested.is_dir():
        matches = {
            candidate.resolve()
            for candidate in requested.rglob("ground_truth")
            if _is_ground_truth_root(candidate)
        }
    if not matches:
        raise FileNotFoundError(
            f"No TrackID3x3 ground_truth tree with Indoor/Outdoor/Drone pose folders under {requested}."
        )
    if len(matches) != 1:
        rendered = ", ".join(str(item) for item in sorted(matches))
        raise TrackIDSchemaError(f"Ambiguous TrackID3x3 roots: {rendered}")
    return next(iter(matches))


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrackIDSchemaError(f"Could not parse {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise TrackIDSchemaError(f"JSON root must be an object: {path}")
    return value


def _resolution(metadata: Mapping[str, Any], path: Path) -> tuple[int, int]:
    value = metadata.get("resolution")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise TrackIDSchemaError(f"Missing [width, height] resolution in {path}.")
    try:
        width, height = int(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise TrackIDSchemaError(f"Non-integer resolution in {path}: {value!r}") from exc
    if width <= 0 or height <= 0:
        raise TrackIDSchemaError(f"Non-positive resolution in {path}: {value!r}")
    return width, height


def _canonical_video_id(file_name: Any, path: Path) -> str:
    if not isinstance(file_name, str) or not file_name.strip():
        raise TrackIDSchemaError(f"Missing video.file_name in {path}.")
    # Outdoor's pose filename has one trailing underscore while the matching
    # official MOT filename does not.
    return Path(file_name).stem.rstrip("_")


def _parse_frame_key(value: Any, path: Path) -> int:
    match = _FRAME_RE.fullmatch(str(value))
    if match is None:
        raise TrackIDSchemaError(f"Expected frame_<integer> key in {path}, got {value!r}.")
    return int(match.group(1))


def _pose_array(value: Any, *, path: Path, frame: int, player: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        points = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TrackIDSchemaError(
            f"Non-numeric pose in {path}, frame {frame}, player {player!r}."
        ) from exc
    if points.shape != (len(TRACKID10_JOINT_NAMES), 2):
        raise TrackIDSchemaError(
            f"Expected source pose shape (10, 2) in {path}, frame {frame}, "
            f"player {player!r}; got {points.shape}."
        )
    finite = np.isfinite(points).all(axis=-1)
    non_sentinel = ~np.all(points == 0.0, axis=-1)
    valid = finite & non_sentinel
    points = points.copy()
    points[~valid] = 0.0
    return points, valid


def _instance_id(domain: str, video_id: str | None, frame: int, player: str) -> str:
    video_token = video_id if video_id else "video_unavailable"
    return f"{domain}:{video_token}:frame_{frame}:{player}"


def _parse_video_json(path: Path, domain: str) -> list[PoseInstance]:
    root = _load_json(path)
    metadata = root.get("video")
    annotations = root.get("annotations")
    if not isinstance(metadata, Mapping) or not isinstance(annotations, Mapping):
        raise TrackIDSchemaError(
            f"{domain} video JSON requires object-valued video and annotations: {path}"
        )
    size = _resolution(metadata, path)
    video_id = _canonical_video_id(metadata.get("file_name"), path)
    result: list[PoseInstance] = []
    seen_frames: set[int] = set()
    ordered_frames = sorted(
        ((_parse_frame_key(key, path), key) for key in annotations), key=lambda item: item[0]
    )
    for frame_index, frame_key in ordered_frames:
        if frame_index in seen_frames:
            raise TrackIDSchemaError(f"Duplicate frame index {frame_index} in {path}.")
        seen_frames.add(frame_index)
        people = annotations[frame_key]
        if not isinstance(people, Mapping) or not people:
            raise TrackIDSchemaError(f"Frame {frame_key!r} is not a non-empty player map in {path}.")
        for player_value in sorted(people, key=str):
            player_id = str(player_value)
            points, mask = _pose_array(
                people[player_value], path=path, frame=frame_index, player=player_id
            )
            result.append(
                PoseInstance(
                    keypoints=points,
                    valid_mask=mask,
                    domain=domain,
                    frame_index=frame_index,
                    player_id=player_id,
                    image_size=size,
                    video_id=video_id,
                    sample_id=_instance_id(domain, video_id, frame_index, player_id),
                    source_file=path.name,
                    temporal_reliable=True,
                )
            )
    declared = metadata.get("total_frames")
    if declared is not None and int(declared) != len(seen_frames):
        raise TrackIDSchemaError(
            f"{path} declares {declared} frames but contains {len(seen_frames)} annotation frames."
        )
    return result


def _parse_drone_json(path: Path) -> list[PoseInstance]:
    root = _load_json(path)
    metadata = root.get("image")
    annotations = root.get("annotations")
    if not isinstance(metadata, Mapping) or not isinstance(annotations, Mapping):
        raise TrackIDSchemaError(
            f"Drone image JSON requires object-valued image and annotations: {path}"
        )
    size = _resolution(metadata, path)
    frame_index = _parse_frame_key(metadata.get("file_name"), path)
    if not annotations:
        raise TrackIDSchemaError(f"Drone annotations are empty in {path}.")
    result: list[PoseInstance] = []
    for player_value in sorted(annotations, key=str):
        player_id = str(player_value)
        points, mask = _pose_array(
            annotations[player_value], path=path, frame=frame_index, player=player_id
        )
        result.append(
            PoseInstance(
                keypoints=points,
                valid_mask=mask,
                domain="Drone",
                frame_index=frame_index,
                player_id=player_id,
                image_size=size,
                video_id=None,
                sample_id=_instance_id("Drone", None, frame_index, player_id),
                source_file=path.name,
                temporal_reliable=False,
            )
        )
    return result


def _pose_tight_bbox(instance: PoseInstance) -> np.ndarray:
    points = instance.keypoints[instance.valid_mask]
    if points.size == 0:
        raise TrackIDSchemaError(f"No valid points for tight bbox: {instance.sample_id}")
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    extent = upper - lower
    if np.any(extent <= 0):
        raise TrackIDSchemaError(f"Degenerate pose-tight bbox: {instance.sample_id}")
    return np.asarray([lower[0], lower[1], extent[0], extent[1]], dtype=np.float32)


def _resolve_mot_path(domain_root: Path, video_id: str) -> Path:
    mot_root = domain_root / "MOT"
    matches = [
        path
        for path in mot_root.glob("*.txt")
        if path.stem.rstrip("_").casefold() == video_id.rstrip("_").casefold()
    ]
    if len(matches) != 1:
        raise TrackIDSchemaError(
            f"Expected one official MOT file for {video_id!r} under {mot_root}, found {len(matches)}."
        )
    return matches[0]


def _load_mot(path: Path) -> dict[int, dict[str, np.ndarray]]:
    frames: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, 1):
            stripped = raw.strip()
            if not stripped:
                continue
            fields = stripped.replace(",", " ").split()
            if len(fields) < 6:
                raise TrackIDSchemaError(f"MOT row has fewer than six fields at {path}:{line_number}.")
            try:
                frame_value = float(fields[0])
                track_value = float(fields[1])
                bbox = np.asarray([float(value) for value in fields[2:6]], dtype=np.float32)
            except ValueError as exc:
                raise TrackIDSchemaError(f"Non-numeric MOT row at {path}:{line_number}.") from exc
            if not frame_value.is_integer() or not track_value.is_integer():
                raise TrackIDSchemaError(f"Non-integral frame/track id at {path}:{line_number}.")
            frame, track = int(frame_value), str(int(track_value))
            if not np.isfinite(bbox).all() or np.any(bbox[2:] <= 0):
                raise TrackIDSchemaError(f"Invalid xywh bbox at {path}:{line_number}.")
            if track in frames[frame]:
                raise TrackIDSchemaError(
                    f"Duplicate frame/track pair ({frame}, {track}) in {path}."
                )
            frames[frame][track] = bbox
    if not frames:
        raise TrackIDSchemaError(f"Official MOT file is empty: {path}")
    return dict(frames)


def _frame_offset(annotation_frames: Sequence[int], mot_frames: Iterable[int]) -> int:
    annotation_set = set(annotation_frames)
    mot_set = set(mot_frames)
    if 0 in mot_set:
        candidates = (0,)
    elif min(annotation_set, default=0) == 0 and min(mot_set) == 1:
        candidates = (1,)
    else:
        candidates = (0, 1)
    scores = {offset: sum((frame + offset) in mot_set for frame in annotation_set) for offset in candidates}
    best = max(scores.values())
    winners = [offset for offset, score in scores.items() if score == best]
    if len(winners) != 1 or best == 0:
        raise TrackIDSchemaError(f"Cannot infer an unambiguous pose-to-MOT frame offset: {scores}.")
    return winners[0]


def _iou_xywh(first: np.ndarray, second: np.ndarray) -> float:
    first_max = first[:2] + first[2:]
    second_max = second[:2] + second[2:]
    overlap = np.maximum(0.0, np.minimum(first_max, second_max) - np.maximum(first[:2], second[:2]))
    intersection = float(overlap[0] * overlap[1])
    union = float(first[2] * first[3] + second[2] * second[3] - intersection)
    return intersection / union if union > 0 else 0.0


def _pair_metrics(instance: PoseInstance, bbox: np.ndarray) -> tuple[float, float, float]:
    pose_bbox = _pose_tight_bbox(instance)
    points = instance.keypoints[instance.valid_mask]
    lower, upper = bbox[:2], bbox[:2] + bbox[2:]
    contained = np.all((points >= lower) & (points <= upper), axis=1)
    containment = float(contained.mean())
    pose_center = pose_bbox[:2] + pose_bbox[2:] / 2.0
    box_center = bbox[:2] + bbox[2:] / 2.0
    diagonal = max(float(np.linalg.norm(bbox[2:])), 1e-6)
    center_distance = float(np.linalg.norm(pose_center - box_center) / diagonal)
    iou = _iou_xywh(pose_bbox, bbox)
    return containment, center_distance, iou


def _associate_official_mot(
    instances: Sequence[PoseInstance],
    domain_root: Path,
    *,
    minimum_dominance: float,
) -> tuple[list[PoseInstance], dict[str, Any]]:
    if not instances:
        return [], {"dropped_instances": 0, "videos": {}}
    by_video: dict[str, list[PoseInstance]] = defaultdict(list)
    for instance in instances:
        if not instance.video_id:
            raise TrackIDSchemaError(f"Official MOT association requires video identity: {instance.sample_id}")
        by_video[instance.video_id].append(instance)

    attached: list[PoseInstance] = []
    dropped = 0
    video_reports: dict[str, Any] = {}
    for video_id in sorted(by_video):
        video_instances = by_video[video_id]
        mot_path = _resolve_mot_path(domain_root, video_id)
        mot = _load_mot(mot_path)
        annotation_frames = sorted({item.frame_index for item in video_instances})
        offset = _frame_offset(annotation_frames, mot)
        by_frame: dict[int, list[PoseInstance]] = defaultdict(list)
        for item in video_instances:
            by_frame[item.frame_index].append(item)

        votes: dict[str, Counter[str]] = defaultdict(Counter)
        appearances: Counter[str] = Counter()
        for frame_index in annotation_frames:
            poses = sorted(by_frame[frame_index], key=lambda item: item.player_id)
            boxes = mot.get(frame_index + offset, {})
            for pose in poses:
                appearances[pose.player_id] += 1
            if not boxes:
                continue
            tracks = sorted(boxes, key=lambda value: int(value))
            cost = np.empty((len(poses), len(tracks)), dtype=np.float64)
            metrics: dict[tuple[int, int], tuple[float, float, float]] = {}
            for pose_index, pose in enumerate(poses):
                for track_index, track in enumerate(tracks):
                    containment, distance, iou = _pair_metrics(pose, boxes[track])
                    metrics[(pose_index, track_index)] = containment, distance, iou
                    cost[pose_index, track_index] = (
                        distance + 2.0 * (1.0 - containment) + 0.25 * (1.0 - iou)
                    )
            rows, columns = linear_sum_assignment(cost)
            for pose_index, track_index in zip(rows.tolist(), columns.tolist(), strict=True):
                containment, distance, iou = metrics[(pose_index, track_index)]
                # Weak spatial pairs are evidence of ambiguity, not permission
                # to borrow the closest identity.
                if containment >= 0.5 and distance <= 1.5 and iou > 0.0:
                    votes[poses[pose_index].player_id][tracks[track_index]] += 1

        mapping: dict[str, str] = {}
        dominance: dict[str, float] = {}
        weak_players: list[str] = []
        for player in sorted(appearances):
            player_votes = votes.get(player, Counter())
            if not player_votes:
                weak_players.append(player)
                continue
            track, count = player_votes.most_common(1)[0]
            ratio = count / sum(player_votes.values())
            support = count / appearances[player]
            if ratio < minimum_dominance or support < 0.5:
                weak_players.append(player)
                continue
            mapping[player] = track
            dominance[player] = ratio
        duplicate_tracks = {
            track for track, count in Counter(mapping.values()).items() if count > 1
        }
        if duplicate_tracks:
            weak_players.extend(player for player, track in mapping.items() if track in duplicate_tracks)
            mapping = {player: track for player, track in mapping.items() if track not in duplicate_tracks}
            dominance = {player: value for player, value in dominance.items() if player in mapping}

        video_dropped = 0
        for item in video_instances:
            track = mapping.get(item.player_id)
            bbox = mot.get(item.frame_index + offset, {}).get(track) if track else None
            if bbox is None:
                video_dropped += 1
                continue
            attached.append(
                replace(
                    item,
                    bbox_xywh=bbox,
                    bbox_source="mot_xywh",
                    mot_track_id=track,
                    association_confidence=dominance[item.player_id],
                )
            )
        dropped += video_dropped
        video_reports[video_id] = {
            "mot_file": str(mot_path),
            "pose_to_mot_frame_offset": offset,
            "player_to_mot_track": mapping,
            "mapping_dominance": dominance,
            "weak_players": sorted(set(weak_players)),
            "input_instances": len(video_instances),
            "attached_instances": len(video_instances) - video_dropped,
            "dropped_instances": video_dropped,
        }
    return attached, {"dropped_instances": dropped, "videos": video_reports}


def _domain_instances(ground_truth: Path, domain: str) -> list[PoseInstance]:
    pose_root = ground_truth / domain / "pose"
    json_paths = sorted(pose_root.glob("*.json"), key=lambda path: path.name.casefold())
    if not json_paths:
        raise TrackIDSchemaError(f"No JSON pose annotations found under {pose_root}.")
    instances: list[PoseInstance] = []
    if domain == "Drone":
        for path in json_paths:
            instances.extend(_parse_drone_json(path))
    else:
        for path in json_paths:
            instances.extend(_parse_video_json(path, domain))
    ids = [item.sample_id for item in instances]
    if len(ids) != len(set(ids)):
        raise TrackIDSchemaError(f"Duplicate canonical sample IDs detected in {domain}.")
    return instances


def load_trackid3x3_with_report(
    path: str | Path,
    *,
    domains: Sequence[str] = TRACKID3X3_DOMAINS,
    attach_bboxes: bool = True,
    minimum_association_dominance: float = 0.80,
) -> TrackIDLoadResult:
    """Load validated annotations and explicit bbox-provenance diagnostics.

    Indoor/Outdoor are associated to official MOT ``xywh`` boxes by a
    video-level dominant one-to-one mapping inferred from spatial assignments.
    Weak or missing mappings are dropped and counted.  Drone has no video ID in
    its pose JSON, so only there a pose-tight fallback is used.
    """

    if not math.isfinite(minimum_association_dominance) or not (
        0.0 <= minimum_association_dominance <= 1.0
    ):
        raise ValueError("minimum_association_dominance must be finite and in [0,1].")
    ground_truth = resolve_trackid_ground_truth(path)
    requested = tuple(domains)
    unknown = sorted(set(requested) - set(TRACKID3X3_DOMAINS))
    if unknown:
        raise ValueError(f"Unknown domains: {unknown}; expected {TRACKID3X3_DOMAINS}.")
    all_instances: list[PoseInstance] = []
    reports: dict[str, Any] = {}
    for domain in requested:
        raw = _domain_instances(ground_truth, domain)
        input_count = len(raw)
        if attach_bboxes and domain in {"Indoor", "Outdoor"}:
            domain_instances, association = _associate_official_mot(
                raw,
                ground_truth / domain,
                minimum_dominance=minimum_association_dominance,
            )
        elif attach_bboxes:
            domain_instances = [
                replace(
                    item,
                    bbox_xywh=_pose_tight_bbox(item),
                    bbox_source="pose_tight_fallback",
                )
                for item in raw
            ]
            association = {
                "dropped_instances": 0,
                "reason": "Drone pose JSON has no reliable video identity for MOT linkage.",
            }
        else:
            domain_instances = raw
            association = {"dropped_instances": 0, "reason": "bbox attachment disabled"}
        all_instances.extend(domain_instances)
        frame_keys = {(item.video_id, item.frame_index) for item in raw}
        reports[domain] = {
            "pose_json_files": len(list((ground_truth / domain / "pose").glob("*.json"))),
            "annotated_frames": len(frame_keys),
            "pose_instances_input": input_count,
            "pose_instances_loaded": len(domain_instances),
            "unique_players": sorted({item.player_id for item in raw}),
            "video_ids": sorted({item.video_id for item in raw if item.video_id}),
            "resolutions": sorted({item.image_size for item in raw}),
            "missing_keypoints": int(sum((~item.valid_mask).sum() for item in raw)),
            "bbox_source_counts": dict(Counter(item.bbox_source for item in domain_instances)),
            "association": association,
        }
    all_instances.sort(key=lambda item: (item.domain, item.video_id or "", item.frame_index, item.player_id))
    diagnostics = {
        "schema": "TrackID3x3.pose_instance.v1",
        "joint_names": list(TRACKID10_JOINT_NAMES),
        "coordinate_order": ["x", "y"],
        "domains": reports,
        "total_instances": len(all_instances),
        "annotation_only": True,
        "rgb_available": False,
    }
    return TrackIDLoadResult(tuple(all_instances), ground_truth, diagnostics)


def load_trackid3x3(
    path: str | Path,
    *,
    domains: Sequence[str] = TRACKID3X3_DOMAINS,
    attach_bboxes: bool = True,
    minimum_association_dominance: float = 0.80,
) -> list[PoseInstance]:
    """Return canonical instances; see :func:`load_trackid3x3_with_report`."""

    return list(
        load_trackid3x3_with_report(
            path,
            domains=domains,
            attach_bboxes=attach_bboxes,
            minimum_association_dominance=minimum_association_dominance,
        ).instances
    )


__all__ = [
    "TRACKID3X3_DOMAINS",
    "TRACKID10_EDGES",
    "TRACKID10_JOINT_NAMES",
    "TRACKID10_TO_COCO17",
    "PoseInstance",
    "TrackIDLoadResult",
    "TrackIDSchemaError",
    "load_trackid3x3",
    "load_trackid3x3_with_report",
    "resolve_trackid_ground_truth",
]
