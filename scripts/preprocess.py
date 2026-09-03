#!/usr/bin/env python
"""Preprocess TrackID3x3 static instances or source-reliable track windows."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.storage import (
    MAX_DATASET_MB,
    estimate_trackid_processed_bytes,
    project_budget_preflight,
    storage_preflight,
)
from src.data.trackid3x3 import PoseInstance, load_trackid3x3_with_report
from src.data.trackid_processed import (
    TRACKID_PROCESSED_SCHEMA,
    build_trackid_processed,
    save_trackid_processed,
)
from src.data.trackid_split import (
    PoseWindow,
    assign_domain_splits,
    select_manifest_instances,
    select_temporal_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "T=1 consumes the deterministic pose-instance manifest. T>1 builds "
            "non-overlapping windows from full assigned Indoor/Outdoor streams "
            "before random thinning; Drone is excluded. Legacy Henu preprocessing is disabled."
        ),
    )
    parser.add_argument("--input", type=Path, default=Path("external/TrackID3x3"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/trackid3x3/subset_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/trackid3x3")
    )
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--frame-step", type=int, default=3)
    parser.add_argument("--mask-rate", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--indoor-observation-cap", type=int, default=1000)
    parser.add_argument("--outdoor-observation-cap", type=int, default=1000)
    parser.add_argument("--drone-observation-cap", type=int, default=500)
    parser.add_argument("--max-artifact-mb", type=float, default=600.0)
    parser.add_argument("--max-dataset-mb", type=float, default=float(MAX_DATASET_MB))
    parser.add_argument("--minimum-free-mb", type=float, default=1024.0)
    parser.add_argument("--minimum-association-dominance", type=float, default=0.80)
    return parser.parse_args()


def _commit(checkout: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _validate_manifest_metadata(
    path: Path, selected: Sequence[PoseInstance]
) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(selected):
        raise ValueError(f"Manifest row count changed while loading: {len(rows)} vs {len(selected)}.")
    for row, item in zip(rows, selected, strict=True):
        expected = {
            "sample_id": item.sample_id,
            "domain": item.domain,
            "split": item.split,
            "split_group": item.split_group,
            "grouping_level": item.grouping_level,
            "bbox_source": item.bbox_source,
        }
        mismatches = {
            key: (row.get(key), value) for key, value in expected.items() if row.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Manifest/source assignment mismatch for {item.sample_id}: {mismatches}. "
                "Rebuild the manifest with the same seed."
            )


def _write_window_manifest(path: Path, windows: Sequence[PoseWindow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "sample_id",
        "domain",
        "video_id",
        "track_id",
        "split",
        "split_group",
        "grouping_level",
        "start_frame",
        "end_frame",
        "sequence_length",
        "frame_step",
        "bbox_source",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for window in windows:
            first, last = window.instances[0], window.instances[-1]
            writer.writerow(
                {
                    "sample_id": window.sample_id,
                    "domain": first.domain,
                    "video_id": first.video_id or "",
                    "track_id": first.player_id,
                    "split": first.split,
                    "split_group": first.split_group,
                    "grouping_level": first.grouping_level,
                    "start_frame": first.frame_index,
                    "end_frame": last.frame_index,
                    "sequence_length": len(window.instances),
                    "frame_step": window.frame_step,
                    "bbox_source": first.bbox_source,
                }
            )


def _counts(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    domains = arrays["domain"].astype(str)
    splits = arrays["split"].astype(str)
    sources = arrays["bbox_source"].astype(str)
    return {
        "samples": int(arrays["coords"].shape[0]),
        "pose_observations": int(arrays["sequence_mask"].sum()),
        "valid_keypoints": int(arrays["valid_mask"].sum()),
        "synthetically_held_out_keypoints": int(arrays["target_mask"].sum()),
        "by_domain": dict(Counter(domains.tolist())),
        "by_split": dict(Counter(splits.tolist())),
        "by_domain_split": {
            domain: dict(Counter(splits[domains == domain].tolist()))
            for domain in sorted(set(domains.tolist()))
        },
        "bbox_source_observations": dict(Counter(sources.ravel().tolist())),
        "unique_source_frames": {
            domain: len(
                {
                    (str(arrays["video_id"][index]), int(frame))
                    for index in np.flatnonzero(domains == domain)
                    for frame in arrays["frame_index"][index]
                }
            )
            for domain in sorted(set(domains.tolist()))
        },
    }


def main() -> int:
    args = parse_args()
    if args.sequence_length <= 0:
        raise ValueError("--sequence-length must be positive.")
    loaded = load_trackid3x3_with_report(
        args.input,
        minimum_association_dominance=args.minimum_association_dominance,
    )
    selection_report: dict[str, Any]
    if args.sequence_length == 1:
        if not args.manifest.is_file():
            raise FileNotFoundError(
                f"Static subset manifest not found: {args.manifest}. "
                "Run `python scripts/build_subset.py` first."
            )
        assigned = assign_domain_splits(loaded.instances, seed=args.seed)
        samples: Sequence[PoseInstance | PoseWindow] = select_manifest_instances(
            assigned, args.manifest
        )
        _validate_manifest_metadata(args.manifest, samples)  # type: ignore[arg-type]
        selection_report = {
            "mode": "static_pose_instances",
            "manifest": str(args.manifest),
            "selected_instances": len(samples),
            "frame_step": None,
        }
    else:
        caps = {
            "Indoor": args.indoor_observation_cap,
            "Outdoor": args.outdoor_observation_cap,
            "Drone": args.drone_observation_cap,
        }
        windows, temporal_report = select_temporal_windows(
            loaded.instances,
            length=args.sequence_length,
            frame_step=args.frame_step,
            target_observations=caps,
            seed=args.seed,
        )
        if not windows:
            raise ValueError(
                "No source-reliable temporal windows satisfy the requested length/frame step."
            )
        samples = windows
        window_manifest = args.output_dir / "temporal_window_manifest.csv"
        _write_window_manifest(window_manifest, windows)
        selection_report = {
            "mode": "pre_thinning_reliable_track_windows",
            "manifest": str(window_manifest),
            **temporal_report,
        }

    estimated = estimate_trackid_processed_bytes(len(samples), args.sequence_length)
    preflight = storage_preflight(
        args.output_dir,
        estimated,
        max_artifact_bytes=int(args.max_artifact_mb * 1_000_000),
        minimum_free_bytes=int(args.minimum_free_mb * 1_000_000),
    )
    project_preflight = project_budget_preflight(
        PROJECT_ROOT,
        args.output_dir,
        estimated,
        max_dataset_mb=args.max_dataset_mb,
    )
    arrays = build_trackid_processed(samples, mask_rate=args.mask_rate, seed=args.seed)
    metadata: dict[str, Any] = {
        "schema_version": TRACKID_PROCESSED_SCHEMA,
        "dataset": "TrackID3x3",
        "source_root": str(loaded.ground_truth_root),
        "source_commit": _commit(loaded.ground_truth_root.parent),
        "synthetic": False,
        "research_results": False,
        "annotation_only": True,
        "rgb_available": False,
        "coordinate_system": "source_image_pixels_xy",
        "bbox_format": "xywh",
        "bbox_policy": {
            "Indoor": "official MOT xywh after video-level spatial identity association",
            "Outdoor": "official MOT xywh; annotation suffix is player identity, not raw MOT id",
            "Drone": "pose_tight_fallback because pose JSON lacks reliable video identity",
        },
        "identity_policy": (
            "track_id preserves the full source identity (for example Green_6); player_number "
            "stores a parsed numeric suffix when present and is not assumed globally unique."
        ),
        "bbox_leakage_guard": (
            "Stored Drone pose-tight boxes are evaluation denominators/provenance. After synthetic "
            "masking, bbox-normalized Drone transforms must be recomputed from observed joints only; "
            "official Indoor/Outdoor MOT boxes are independent and may be used directly."
        ),
        "seed": args.seed,
        "default_reconstruction_mask_rate": args.mask_rate,
        "selection": selection_report,
        "counts": _counts(arrays),
        "representations": {
            "stored": [
                "raw",
                "bbox_normalized",
                "root_centered",
                "scale_normalized",
                "angles",
            ],
            "warning": (
                "Stored full-pose representations are exploratory targets. Training must mask in pixel "
                "space first and derive root/scale from the observed mask at runtime."
            ),
            "runtime_api": (
                "src.features.trackid_features.build_observed_representation(keypoints, "
                "original_valid_mask, observed_mask, representation, bboxes=..., bbox_valid=..., "
                "bbox_sources=...)"
            ),
            "inverse_api": "src.features.trackid_features.inverse_trackid_representation",
            "root_policy": (
                "observed center; both observed shoulders; observed-joint centroid; bbox-center fallback"
            ),
            "scale_policy": (
                "observed shoulder-midpoint-to-center torso; observed median radius; bbox diagonal fallback"
            ),
        },
        "array_contract": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
        "loader_aliases": {
            "keypoints": "coords",
            "bboxes/bbox_xywh": "bbox",
            "sample_ids": "sample_id",
            "domains": "domain",
            "video_ids": "video_id",
            "player_ids/track_ids": "track_id",
            "splits": "split",
        },
        "source_diagnostics": loaded.diagnostics,
        "storage_preflight": preflight,
        "project_budget_preflight": project_preflight,
    }
    dataset_path, metadata_path = save_trackid_processed(args.output_dir, arrays, metadata)
    print(
        f"Wrote {arrays['coords'].shape[0]} samples with shape "
        f"{tuple(arrays['coords'].shape)} to {dataset_path}"
    )
    print(f"Metadata: {metadata_path}")
    print(f"Artifact bytes: {dataset_path.stat().st_size:,}")
    print(f"Bbox sources: {metadata['counts']['bbox_source_observations']}")
    if args.sequence_length > 1:
        print("Drone temporal samples: 0 (no reliable pose-video identity)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
