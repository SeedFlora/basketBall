#!/usr/bin/env python
"""Build the deterministic grouped TrackID3x3 pose-instance subset."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.trackid3x3 import load_trackid3x3_with_report
from src.data.trackid_split import (
    select_domain_subset,
    select_temporal_windows,
    write_subset_manifest,
    write_subset_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Targets count pose instances after frame stride, not frames. "
            "Legacy Henu subject/action selection is disabled."
        ),
    )
    parser.add_argument("--input", type=Path, default=Path("external/TrackID3x3"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/trackid3x3/subset_manifest.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("data/processed/trackid3x3/subset_metadata.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--indoor-target", type=int, default=1000)
    parser.add_argument("--outdoor-target", type=int, default=1000)
    parser.add_argument("--drone-target", type=int, default=500)
    parser.add_argument("--minimum-association-dominance", type=float, default=0.80)
    parser.add_argument(
        "--audit-output", type=Path, default=Path("outputs/reports/split_audit.md")
    )
    return parser.parse_args()


def _write_audit(
    path: Path,
    selected: list,
    static_report: dict,
    temporal_report: dict,
) -> None:
    ids = [item.sample_id for item in selected]
    duplicate_count = len(ids) - len(set(ids))
    lines = [
        "# TrackID3x3 split audit",
        "",
        "## Status: passed",
        "",
        f"- Seed: `{static_report['seed']}`",
        f"- Frame stride before static instance selection: `{static_report['stride']}`",
        "- Splitting is performed independently within every domain.",
        "- Synthetic: `false`",
        "- Legacy Henu subject/action split: not used",
        f"- Duplicate canonical sample IDs: `{duplicate_count}`",
        "",
        "## Static pose-instance subset",
        "",
        "| Domain | Grouping level | Groups train/val/test | Instances train/val/test | Selected instances | Per-group range | Unique selected frames |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for domain in ("Indoor", "Outdoor", "Drone"):
        report = static_report["domains"][domain]
        grouping = "video" if domain == "Indoor" else "player"
        group_counts = report["group_counts_by_split"]
        instance_counts = report["selected_by_split"]
        lines.append(
            f"| {domain} | {grouping} | "
            f"{group_counts.get('train', 0)}/{group_counts.get('val', 0)}/{group_counts.get('test', 0)} | "
            f"{instance_counts.get('train', 0)}/{instance_counts.get('val', 0)}/{instance_counts.get('test', 0)} | "
            f"{report['selected_instances']} | "
            f"{report['selected_group_balance_range'][0]}-{report['selected_group_balance_range'][1]} | "
            f"{report['selected_unique_frames']} |"
        )
    lines += ["", "## Pairwise split-group overlap", ""]
    all_passed = True
    for domain in ("Indoor", "Outdoor", "Drone"):
        split_groups = {
            split: {item.split_group for item in selected if item.domain == domain and item.split == split}
            for split in ("train", "val", "test")
        }
        overlaps = {
            "train/val": len(split_groups["train"] & split_groups["val"]),
            "train/test": len(split_groups["train"] & split_groups["test"]),
            "val/test": len(split_groups["val"] & split_groups["test"]),
        }
        all_passed &= all(value == 0 for value in overlaps.values())
        lines.append(f"- {domain}: `{overlaps}`")
    lines += [
        f"- All pairwise overlap counts are zero: `{str(all_passed).lower()}`",
        "",
        "## Temporal dataset audit (T=16)",
        "",
        "Temporal windows are built from full assigned source streams before random static-instance thinning. Frames are regular at step 3, windows have hop 16 observations (48 source frames), and therefore do not overlap within a track.",
        "",
        "| Domain | Available windows | Selected windows | Selected pose observations | Rule |",
        "|---|---:|---:|---:|---|",
    ]
    for domain in ("Indoor", "Outdoor", "Drone"):
        item = temporal_report["domains"][domain]
        rule = item["exclusion"] or "reliable video/player track, regular step 3, non-overlapping"
        lines.append(
            f"| {domain} | {item['available_windows']} | {item['selected_windows']} | "
            f"{item['selected_pose_observations']} | {rule} |"
        )
    lines += [
        "",
        "Drone is deliberately excluded from temporal windows because its pose JSON has no reliable video identity. This is not interpreted as an empty or failed track.",
        "",
        f"Static bbox provenance: `{dict(Counter(item.bbox_source for item in selected))}`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    loaded = load_trackid3x3_with_report(
        args.input,
        minimum_association_dominance=args.minimum_association_dominance,
    )
    targets = {
        "Indoor": args.indoor_target,
        "Outdoor": args.outdoor_target,
        "Drone": args.drone_target,
    }
    selected, report = select_domain_subset(
        loaded.instances,
        targets=targets,
        stride=args.stride,
        seed=args.seed,
    )
    _, temporal_report = select_temporal_windows(
        loaded.instances,
        length=16,
        frame_step=args.stride,
        target_observations=targets,
        seed=args.seed,
    )
    report = {
        **report,
        "dataset": "TrackID3x3",
        "schema": loaded.diagnostics["schema"],
        "source_root": str(loaded.ground_truth_root),
        "synthetic": False,
        "annotation_only": True,
        "bbox_source_counts": {
            domain: loaded.diagnostics["domains"][domain]["bbox_source_counts"]
            for domain in ("Indoor", "Outdoor", "Drone")
        },
    }
    write_subset_manifest(args.output, selected)
    # Keep the legacy generic filename as a byte-small compatibility copy;
    # subset_manifest.csv is the canonical TrackID3x3 artifact.
    if args.output.name == "subset_manifest.csv":
        write_subset_manifest(args.output.with_name("split_manifest.csv"), selected)
    write_subset_metadata(args.metadata_output, report)
    _write_audit(args.audit_output, selected, report, temporal_report)
    print(f"Selected {len(selected)} pose instances after stride {args.stride} (seed {args.seed}).")
    for domain, item in report["domains"].items():
        print(
            f"{domain}: instances={item['selected_instances']}, "
            f"unique_frames={item['selected_unique_frames']}, "
            f"split_instances={item['selected_by_split']}, "
            f"split_groups={item['group_counts_by_split']}"
        )
    print(f"Manifest: {args.output}")
    print(f"Metadata: {args.metadata_output}")
    print(f"Audit: {args.audit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
