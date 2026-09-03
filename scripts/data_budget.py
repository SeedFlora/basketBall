#!/usr/bin/env python
"""Estimate and preflight TrackID3x3 processed-dataset storage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.storage import (
    MAX_DATASET_MB,
    MAX_STORAGE_MB,
    TARGET_DATASET_MB,
    StorageBudgetError,
    current_storage_budget,
    estimate_trackid_processed_bytes,
    project_budget_preflight,
    storage_preflight,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/trackid3x3"))
    parser.add_argument("--max-artifact-mb", type=float, default=float(MAX_STORAGE_MB))
    parser.add_argument("--max-dataset-mb", type=float, default=float(MAX_DATASET_MB))
    parser.add_argument("--minimum-free-mb", type=float, default=1024.0)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    current = current_storage_budget(PROJECT_ROOT)
    print(f"Current raw data: {current['current_raw_data_mb']:.3f} MB")
    print(f"Current processed data: {current['current_processed_data_mb']:.3f} MB")
    print(f"Current output data: {current['current_output_data_mb']:.3f} MB")
    print(f"Total: {current['total_mb']:.3f} MB")
    print(f"Remaining budget: {current['remaining_budget_mb']:.3f} MB")
    print(f"MAX_DATASET_MB={MAX_DATASET_MB}; TARGET_DATASET_MB={TARGET_DATASET_MB}")
    print(
        f"Source checkout overhead: {current['source_overhead_mb']:.3f} MB; "
        f"preferred-target headroom: {current['remaining_target_mb']:.3f} MB"
    )
    estimated = estimate_trackid_processed_bytes(args.samples, args.sequence_length)
    try:
        report = storage_preflight(
            args.output_dir,
            estimated,
            max_artifact_bytes=int(args.max_artifact_mb * 1_000_000),
            minimum_free_bytes=int(args.minimum_free_mb * 1_000_000),
        )
        project_report = project_budget_preflight(
            PROJECT_ROOT,
            args.output_dir,
            estimated,
            max_dataset_mb=args.max_dataset_mb,
        )
    except StorageBudgetError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Estimated uncompressed upper bound: {estimated / 1_000_000:.3f} MB")
    print(f"Free after estimate: {int(report['free_bytes_after_estimate']) / 1_000_000:.3f} MB")
    print("Storage preflight: approved")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                {"current": current, "preflight": report, "project_preflight": project_report},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"Report: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
