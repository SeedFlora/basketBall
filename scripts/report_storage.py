#!/usr/bin/env python
"""Report TrackID3x3 annotation, checkout, processed, and output storage."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.storage import (
    build_storage_manifest,
    manifest_totals,
    write_storage_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/storage_manifest.csv"),
        help="CSV with exact columns path,category,size_mb,required,source.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    entries = build_storage_manifest(root)
    write_storage_manifest(output, entries)
    totals = manifest_totals(entries)
    raw_mb = totals["annotations"] + totals["videos"]
    processed_mb = totals["processed"]
    checkpoint_mb = totals["checkpoints"]
    footprint_mb = sum(totals.values())
    print(f"Total raw data size: {raw_mb:.3f} MB")
    print(f"Total processed data size: {processed_mb:.3f} MB")
    print(f"Checkpoint size: {checkpoint_mb:.3f} MB")
    print(f"Total experiment footprint: {footprint_mb:.3f} MB")
    print("Category breakdown:")
    for category, size_mb in totals.items():
        print(f"{category}: {size_mb:.3f} MB")
    print(f"Manifest: {output}")
    if args.json_output is not None:
        json_path = args.json_output if args.json_output.is_absolute() else root / args.json_output
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                {
                    "entries": [asdict(entry) for entry in entries],
                    "summary_mb": {
                        "total_raw_data_size": raw_mb,
                        "total_processed_data_size": processed_mb,
                        "checkpoint_size": checkpoint_mb,
                        "total_experiment_footprint": footprint_mb,
                    },
                    "totals_mb": totals,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
