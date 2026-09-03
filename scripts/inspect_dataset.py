#!/usr/bin/env python
"""Inspect the actual TrackID3x3 JSON/MOT schema and write an evidence report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.storage import directory_bytes
from src.data.trackid3x3 import (
    TRACKID3X3_DOMAINS,
    TRACKID10_JOINT_NAMES,
    TRACKID10_TO_COCO17,
    TrackIDSchemaError,
    load_trackid3x3_with_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Legacy MultiSubjects_SA/Henu pickle adaptation is disabled in this public CLI.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("external/TrackID3x3"),
        help="TrackID3x3 checkout, ground_truth directory, or containing directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/trackid3x3_schema.md"),
    )
    parser.add_argument("--minimum-association-dominance", type=float, default=0.80)
    parser.add_argument(
        "--without-bboxes",
        action="store_true",
        help="Validate pose JSON only; normally official/fallback bbox provenance is audited too.",
    )
    return parser.parse_args()


def _git_commit(checkout: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _unavailable(path: Path, reason: str) -> str:
    return "\n".join(
        [
            "# TrackID3x3 schema inspection",
            "",
            "## Status: unavailable",
            "",
            f"Input checked: `{path}`",
            "",
            f"Reason: {reason}",
            "",
            "No schema, counts, joint meanings, bbox linkage, or research result is inferred.",
            "Legacy MultiSubjects_SA/Henu assumptions are disabled in this CLI.",
            "",
        ]
    )


def _display_path(path: Path) -> str:
    """Render repository-relative evidence paths when possible."""

    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _read_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a top-level JSON object: {path}")
    return value


def _first_nonempty_line(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw in handle:
            if raw.strip():
                return raw.strip()
    raise ValueError(f"Expected at least one MOT row: {path}")


def _observed_layout_lines(ground_truth_root: Path) -> list[str]:
    """Return reproducible, source-derived JSON and MOT layout evidence."""

    indoor_pose = ground_truth_root / "Indoor" / "pose" / "basket_S4T4_post.json"
    outdoor_pose = ground_truth_root / "Outdoor" / "pose" / "IMG_0104_.json"
    drone_pose = ground_truth_root / "Drone" / "pose" / "frame_1.json"
    indoor_mot = ground_truth_root / "Indoor" / "MOT" / "basket_S4T4_post.txt"
    outdoor_mot = ground_truth_root / "Outdoor" / "MOT" / "IMG_0104.txt"

    required = (indoor_pose, outdoor_pose, drone_pose, indoor_mot, outdoor_mot)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing canonical schema-evidence file(s): {missing}")

    indoor = _read_json_object(indoor_pose)
    outdoor = _read_json_object(outdoor_pose)
    drone = _read_json_object(drone_pose)

    indoor_annotations = indoor["annotations"]
    drone_annotations = drone["annotations"]
    if not isinstance(indoor_annotations, dict) or not isinstance(drone_annotations, dict):
        raise TypeError("Canonical example annotations must be object-valued.")
    indoor_frame = indoor_annotations["frame_0"]
    if not isinstance(indoor_frame, dict):
        raise TypeError("Indoor annotations.frame_0 must be object-valued.")

    indoor_player = "Pink_Jersey" if "Pink_Jersey" in indoor_frame else next(iter(indoor_frame))
    drone_player = "Black_1" if "Black_1" in drone_annotations else next(iter(drone_annotations))
    indoor_example = {
        "video": indoor["video"],
        "annotations": {"frame_0": {indoor_player: indoor_frame[indoor_player]}},
    }
    drone_example = {
        "image": drone["image"],
        "annotations": {drone_player: drone_annotations[drone_player]},
    }

    outdoor_annotations = outdoor["annotations"]
    if not isinstance(outdoor_annotations, dict):
        raise TypeError("Outdoor annotations must be object-valued.")
    outdoor_frame = outdoor_annotations["frame_0"]
    if not isinstance(outdoor_frame, dict):
        raise TypeError("Outdoor annotations.frame_0 must be object-valued.")
    outdoor_player = "Green_6" if "Green_6" in outdoor_frame else next(iter(outdoor_frame))

    indoor_row = _first_nonempty_line(indoor_mot)
    outdoor_row = _first_nonempty_line(outdoor_mot)
    indoor_columns = len(indoor_row.replace(",", " ").split())
    outdoor_columns = len(outdoor_row.replace(",", " ").split())

    lines = [
        "Actual inspected examples are:",
        "",
        f"- `{_display_path(indoor_pose)}`",
        f"- `{_display_path(outdoor_pose)}`",
        f"- `{_display_path(drone_pose)}`",
        "",
        "Indoor and Outdoor use this exact field nesting (one complete ten-pair Indoor pose is shown):",
        "",
        "```json",
        *json.dumps(indoor_example, ensure_ascii=False, indent=2).splitlines(),
        "```",
        "",
        (
            "Outdoor has the same keys and nesting; for example its `video` object is "
            f"`{json.dumps(outdoor['video'], ensure_ascii=False)}`, and an observed identity "
            f"key in `annotations.frame_0` is `{outdoor_player}`."
        ),
        "",
        (
            "Drone uses image metadata and has no `frame_N` wrapper inside `annotations` "
            "(one complete ten-pair pose is shown):"
        ),
        "",
        "```json",
        *json.dumps(drone_example, ensure_ascii=False, indent=2).splitlines(),
        "```",
        "",
        (
            f"The identity is the object key (`{indoor_player}`, `{outdoor_player}`, or "
            f"`{drone_player}` in these examples). A pose value is directly a 10-element list "
            "of two-number `[x, y]` lists; there is no nested keypoint-name, confidence, "
            "visibility, bbox, action-label, or biomechanics field."
        ),
        "",
        "## Bbox linkage and provenance",
        "",
        (
            "- Indoor and Outdoor bboxes are stored separately from pose JSON. Inspected files "
            f"include `{_display_path(indoor_mot)}` and `{_display_path(outdoor_mot)}`."
        ),
        (
            "- MOT text is headerless, comma-separated, and positional. The adapter consumes "
            "exactly columns 1-6 as `[frame, track, x, y, width, height]`. An Indoor row is "
            f"`{indoor_row}` ({indoor_columns} observed columns); an Outdoor row is "
            f"`{outdoor_row}` ({outdoor_columns} observed columns). Columns 7 onward are "
            "preserved upstream but are not consumed or assigned new semantics by this project. "
            "Pose `frame_0` aligns with MOT frame 1."
        ),
        (
            "- Player-to-MOT identities are established as a dominant one-to-one mapping over "
            "full videos after same-frame spatial assignment. Weak mappings are dropped rather "
            "than guessed."
        ),
        (
            "- Outdoor annotation suffixes remain canonical player identities; they are not "
            "silently equated to raw MOT track IDs."
        ),
        (
            "- Numeric suffixes are parsed separately as `player_number`; the full color/team "
            "plus suffix remains `track_id` because a number alone is not globally unique "
            "(for example Drone has two number 10 players)."
        ),
        (
            "- Drone pose JSON has no bbox field and cannot be linked unambiguously to a "
            "particular MOT video. Its derived box is `[min_x, min_y, max_x - min_x, "
            "max_y - min_y]` over valid pose points; each instance is marked "
            "`pose_tight_fallback`, never `mot_xywh`."
        ),
        (
            "- Leakage guard: a stored Drone pose-tight box uses the full annotated pose and is "
            "retained only as evaluation scale/provenance. After synthetic masking, bbox "
            "normalization recomputes a tight transform from observed joints only. The stored "
            "box is not an independent model-input transform."
        ),
        "",
    ]
    return lines


def _render(args: argparse.Namespace) -> str:
    result = load_trackid3x3_with_report(
        args.input,
        attach_bboxes=not args.without_bboxes,
        minimum_association_dominance=args.minimum_association_dominance,
    )
    diagnostics = result.diagnostics
    instances = list(result.instances)
    checkout = result.ground_truth_root.parent
    lines = [
        "# TrackID3x3 schema inspection",
        "",
        "## Status: source annotations validated",
        "",
        f"- Checkout: `{checkout}`",
        f"- Git commit inspected: `{_git_commit(checkout)}`",
        f"- Canonical schema: `{diagnostics['schema']}`",
        "- Artifact type: annotation-only; RGB frames/videos are not required or loaded",
        "- Synthetic: `false`",
        "- Legacy MultiSubjects_SA/Henu action-label assumptions: disabled",
        "- Official repository: <https://github.com/open-starlab/TrackID3x3>",
        "- Official paper: <https://arxiv.org/html/2503.18282v2>",
        "",
        "## Exact source pose order",
        "",
        "The JSON files contain arrays of ten x/y pairs but no embedded joint-name field. The mapping below follows the paper's ordered ten-point protocol: it enumerates head, bilateral shoulders, elbows, wrists, center, and ankles; defines center as the hip midpoint; and states that the corresponding COCO17 baseline points use nose for head. No extra hip, knee, or depth coordinate is introduced.",
        "",
        "| Source index | Canonical name | COCO17 evidence |",
        "|---:|---|---|",
    ]
    for index, (name, coco_index) in enumerate(
        zip(TRACKID10_JOINT_NAMES, TRACKID10_TO_COCO17, strict=True)
    ):
        evidence = "midpoint of COCO17 left/right hips (11, 12)" if coco_index is None else str(coco_index)
        lines.append(f"| {index} | `{name}` | {evidence} |")
    lines += [
        "",
        "Coordinates are source-image pixels in `[x, y]` order. Every inspected pose is exactly `[10, 2]`. Exact `[0, 0]` and non-finite pairs are treated as missing and represented by an explicit mask.",
        "",
        "## Empirical annotation counts",
        "",
        "| Domain | Pose JSON files | Annotated frames | Pose instances | Players | Resolution | Missing points | Bbox source |",
        "|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for domain in TRACKID3X3_DOMAINS:
        item = diagnostics["domains"][domain]
        bbox_sources = ", ".join(
            f"{key}: {value}" for key, value in sorted(item["bbox_source_counts"].items())
        ) or "not attached"
        resolutions = ", ".join(f"{width}x{height}" for width, height in item["resolutions"])
        lines.append(
            f"| {domain} | {item['pose_json_files']} | {item['annotated_frames']} | "
            f"{item['pose_instances_input']} | {len(item['unique_players'])} | {resolutions} | "
            f"{item['missing_keypoints']} | {bbox_sources} |"
        )
    total_files = sum(
        diagnostics["domains"][domain]["pose_json_files"]
        for domain in TRACKID3X3_DOMAINS
    )
    total_frames = sum(
        diagnostics["domains"][domain]["annotated_frames"]
        for domain in TRACKID3X3_DOMAINS
    )
    total_input = sum(
        diagnostics["domains"][domain]["pose_instances_input"] for domain in TRACKID3X3_DOMAINS
    )
    total_missing = sum(
        diagnostics["domains"][domain]["missing_keypoints"]
        for domain in TRACKID3X3_DOMAINS
    )
    lines += [
        f"| **Total** | **{total_files}** | **{total_frames}** | **{total_input}** | - | - | **{total_missing}** | - |",
        "",
        "These are pose-annotation counts in the checked-out JSON, not claims about all RGB frames in the full benchmark. Unique-frame and pose-instance counts are recorded separately because each annotated frame usually contains six players.",
        "",
        "## Observed JSON layouts",
        "",
        "- Indoor: 13 video JSON objects with top-level `video` and `annotations`; each `frame_N` maps six jersey identities to poses.",
        "- Outdoor: one video JSON object with 3,600 frame maps; five or six pose instances per frame. `Purple_10` is absent in 72 frames.",
        "- Drone: 500 image JSON objects with top-level `image` and `annotations`; each contains six identities. The pose JSON does not provide a reliable video identifier or frame rate.",
        "",
    ]
    lines += _observed_layout_lines(result.ground_truth_root)
    for domain in ("Indoor", "Outdoor"):
        association = diagnostics["domains"][domain]["association"]
        dominance = [
            score
            for report in association.get("videos", {}).values()
            for score in report["mapping_dominance"].values()
        ]
        if dominance:
            lines.append(
                f"- {domain}: {len(association.get('videos', {}))} pose video(s), "
                f"{association['dropped_instances']} dropped instances, mapping dominance "
                f"range `{min(dominance):.6f} to {max(dominance):.6f}`."
            )
        else:
            lines.append(f"- {domain}: bbox association not assessed (`--without-bboxes`).")
    source_counts = Counter(item.bbox_source for item in instances)
    lines += [
        "",
        f"Loaded bbox provenance totals: `{dict(sorted(source_counts.items()))}`.",
        "",
        "## Split and temporal reliability rules",
        "",
        "- Every domain is split independently 70/15/15 with seed 42.",
        "- Indoor grouping unit: video (13 groups -> 9/2/2 train/val/test).",
        "- Outdoor grouping unit: player/track (6 groups -> 4/1/1) because only one pose video is annotated.",
        "- Drone grouping unit: player/track (6 groups -> 4/1/1) because pose files lack reliable video identity.",
        "- `split_group` and `grouping_level` are retained in manifests and arrays; group disjointness is asserted.",
        "- Temporal windows are permitted only for Indoor/Outdoor video-player tracks. Drone is excluded from temporal datasets.",
        "",
        "## Storage evidence",
        "",
        f"- `ground_truth` bytes: {directory_bytes(result.ground_truth_root):,}",
        f"- Checkout `.git` overhead bytes (separate): {directory_bytes(checkout / '.git'):,}",
        f"- Pose JSON bytes: {directory_bytes(result.ground_truth_root, suffixes={'.json'}):,} bytes",
        f"- MOT text: {directory_bytes(result.ground_truth_root, suffixes={'.txt'}):,} bytes",
        "",
        "Generated from the checked-out source by `scripts/inspect_dataset.py`; no performance metric or research conclusion is reported here.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        report = _render(args)
        status = 0
    except (FileNotFoundError, TrackIDSchemaError, OSError, ValueError) as exc:
        report = _unavailable(args.input, str(exc))
        status = 0 if isinstance(exc, FileNotFoundError) else 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote: {args.output}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
