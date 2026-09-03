#!/usr/bin/env python
"""Audit TrackID3x3 provenance, leakage, storage, and result integrity gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_FOOTPRINT_MB = 600.0
REQUIRED_CODE = (
    "scripts/download_dataset.py",
    "scripts/inspect_dataset.py",
    "scripts/build_subset.py",
    "scripts/preprocess.py",
    "scripts/data_budget.py",
    "scripts/report_storage.py",
    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/run_ablation.py",
    "scripts/run_cross_domain.py",
    "scripts/generate_report.py",
    "scripts/cleanup.py",
    "src/data/trackid3x3.py",
    "src/features/joint_angles.py",
    "src/models/reconstruction.py",
    "src/training/masking.py",
    "src/evaluation/metrics.py",
    "config/joint_mapping.yaml",
)
FORBIDDEN_DATASET_NAMES = {
    "MultiSubjects_SA.zip",
    "key_points_csv.rar",
    "Subjects1000.rar",
    "MultiSubjects_train_val_test.rar",
}


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink()
    )


def experiment_footprint_mb() -> float:
    roots = (
        PROJECT_ROOT / "external" / "TrackID3x3",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "outputs",
    )
    return sum(directory_bytes(path) for path in roots) / 1024**2


def check_upstream_clone() -> list[str]:
    clone = PROJECT_ROOT / "external" / "TrackID3x3"
    if not (clone / "ground_truth").is_dir():
        return ["Official TrackID3x3 annotation checkout is missing."]
    result = subprocess.run(
        ["git", "-C", str(clone), "remote", "get-url", "origin"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode or "open-starlab/TrackID3x3" not in result.stdout:
        return ["external/TrackID3x3 does not have the official upstream remote."]
    return []


def check_joint_mapping(path: Path) -> list[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    mapping = payload.get("joints", payload) if isinstance(payload, dict) else payload
    if isinstance(mapping, dict):
        keys = {int(key) for key in mapping if str(key).isdigit()}
        if keys != set(range(10)):
            return [f"Joint mapping indices must be exactly 0..9, got {sorted(keys)}."]
        return []
    if isinstance(mapping, list) and len(mapping) == 10:
        return []
    return ["config/joint_mapping.yaml does not define exactly ten indexed joints."]


def _manifest_column(table: pd.DataFrame, *names: str) -> str | None:
    return next((name for name in names if name in table.columns), None)


def check_manifest(path: Path) -> list[str]:
    failures: list[str] = []
    table = pd.read_csv(path, dtype=str).fillna("")
    sample_column = _manifest_column(table, "sample_id", "sample_ids")
    split_column = _manifest_column(table, "split", "splits")
    group_column = _manifest_column(table, "split_group", "group_id", "group_ids")
    domain_column = _manifest_column(table, "domain", "domains")
    missing = [
        label
        for label, column in {
            "sample_id": sample_column,
            "split": split_column,
            "split_group": group_column,
            "domain": domain_column,
        }.items()
        if column is None
    ]
    if missing:
        return [f"Manifest lacks columns: {missing}"]
    assert sample_column and split_column and group_column and domain_column
    if table[sample_column].eq("").any() or table[sample_column].duplicated().any():
        failures.append("Manifest sample identifiers are empty or duplicated.")
    grouped = table.groupby([domain_column, group_column])[split_column].nunique()
    if (grouped > 1).any():
        failures.append(f"Group leakage detected in {int((grouped > 1).sum())} domain/group pairs.")
    domains = set(table[domain_column])
    expected_domains = {"Indoor", "Outdoor", "Drone"}
    if domains != expected_domains:
        failures.append(f"Manifest domains are {sorted(domains)}, expected {sorted(expected_domains)}.")
    for domain, rows in table.groupby(domain_column):
        missing_splits = {"train", "val", "test"} - set(rows[split_column])
        if missing_splits:
            failures.append(f"{domain} lacks split(s): {sorted(missing_splits)}")
    return failures


def find_manifest() -> Path | None:
    processed = PROJECT_ROOT / "data" / "processed"
    paths = [
        *processed.glob("**/subset_manifest.csv"),
        *processed.glob("**/split_manifest.csv"),
    ]
    trackid = [path for path in paths if "trackid" in str(path).lower()]
    selected = trackid or paths
    if not selected:
        return None
    # ``subset_manifest.csv`` is the canonical TrackID3x3 name.  A matching
    # ``split_manifest.csv`` may coexist only as a legacy compatibility copy.
    canonical = [path for path in selected if path.name == "subset_manifest.csv"]
    return max(canonical or selected, key=lambda path: path.stat().st_mtime)


def has_real_metrics() -> bool:
    for path in (PROJECT_ROOT / "outputs" / "logs").glob("**/metrics.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("synthetic") is False and (
            "nmje" in payload
            or isinstance(payload.get("test"), dict) and "nmje" in payload["test"]
            or isinstance(payload.get("metrics"), dict) and "nmje" in payload["metrics"]
        ):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-real-results", action="store_true")
    args = parser.parse_args()
    failures = [
        f"Missing required file: {relative}"
        for relative in REQUIRED_CODE
        if not (PROJECT_ROOT / relative).is_file()
    ]
    present = {path.name for path in PROJECT_ROOT.rglob("*") if path.is_file()}
    for name in sorted(FORBIDDEN_DATASET_NAMES & present):
        failures.append(f"Forbidden legacy dataset file is present: {name}")
    requirements = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PROJECT_ROOT / "pyproject.toml", PROJECT_ROOT / "requirements.txt")
    ).lower()
    if "huggingface_hub" in requirements:
        failures.append("Hugging Face remains a project dependency.")
    failures.extend(check_upstream_clone())
    mapping_path = PROJECT_ROOT / "config" / "joint_mapping.yaml"
    if mapping_path.is_file():
        failures.extend(check_joint_mapping(mapping_path))
    manifest = find_manifest()
    if manifest:
        failures.extend(check_manifest(manifest))
    elif args.require_real_results:
        failures.append("No TrackID3x3 subset_manifest.csv exists.")
    footprint = experiment_footprint_mb()
    if footprint > MAX_FOOTPRINT_MB:
        failures.append(
            f"Experiment/data footprint {footprint:.2f} MiB exceeds {MAX_FOOTPRINT_MB:.0f} MiB."
        )
    if args.require_real_results and not has_real_metrics():
        failures.append("No executed non-synthetic reconstruction metrics artifact was found.")
    print(f"Experiment/data footprint: {footprint:.2f} MiB / {MAX_FOOTPRINT_MB:.0f} MiB")
    print(f"Manifest: {manifest if manifest else 'missing'}")
    if failures:
        print("QUALITY GATE: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("QUALITY GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
