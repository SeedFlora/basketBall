#!/usr/bin/env python
"""Safely identify removable project artifacts; deletion requires ``--confirm``."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import directory_size, human_bytes, print_storage_report

KNOWN_ARCHIVES: tuple[str, ...] = (
    "trackid3x3_annotations.zip",
    "trackid3x3_media_subset.zip",
    "MultiSubjects_SA.zip",
)
KNOWN_EXTRACTED_DIRS: tuple[str, ...] = (
    "trackid3x3_extract",
    "duplicate_sampled_frames",
)
KNOWN_TEMP_DIRS: tuple[tuple[str, ...], ...] = (
    ("data", "interim", "tmp"),
    ("data", "interim", "temp"),
    ("outputs", "tmp"),
    ("outputs", "cache"),
    (".pytest_cache",),
)
OLD_CHECKPOINT_NAMES: tuple[str, ...] = (
    "last.pt",
    "last.pth",
    "checkpoint.pt",
    "checkpoint.pth",
)
RECONSTRUCTION_TASKS = {
    "keypoint_reconstruction",
    "pose_reconstruction",
    "masked_keypoint_reconstruction",
}
LEGACY_TABLE_NAMES: tuple[str, ...] = (
    "ablation_results.csv",
    "ablation_results.md",
    "model_comparison.csv",
    "model_comparison.md",
    "robustness_results.csv",
    "robustness_results.md",
    "sample_size_results.csv",
    "sample_size_results.md",
)


@dataclass(frozen=True, slots=True)
class CleanupCandidate:
    path: Path
    category: str
    reason: str
    size_bytes: int
    protected: bool = False


def validate_target(project_root: str | Path, target: str | Path) -> Path:
    """Resolve and prove that a cleanup target is strictly inside the project."""

    root = Path(project_root).resolve()
    path = Path(target)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Refusing cleanup target outside project: {resolved}")
    return resolved


def _has_processed_evidence(root: Path) -> bool:
    processed = root / "data" / "processed"
    for metadata_path in processed.glob("**/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict) or metadata.get("synthetic") is True:
            continue
        schema = str(metadata.get("schema_version", "")).lower()
        dataset = str(metadata.get("dataset", "")).lower()
        compact = metadata_path.with_name("dataset.npz")
        if compact.is_file() and compact.stat().st_size > 0 and (
            "trackid3x3" in schema or dataset == "trackid3x3"
        ):
            return True
    return False


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _legacy_metric(payload: dict[str, object]) -> bool:
    task = str(payload.get("task", "")).lower()
    if task in RECONSTRUCTION_TASKS:
        return False
    serialized = json.dumps(payload, default=str).lower()
    return any(marker in serialized for marker in ("macro_f1", "confusion_matrix", "weighted_f1"))


def _legacy_candidates(root: Path, *, enabled: bool) -> list[CleanupCandidate]:
    """Find only evidenced pre-pivot metric/figure/checkpoint artifacts."""

    candidates: list[CleanupCandidate] = []
    logs_root = root / "outputs" / "logs"
    legacy_run_names: set[str] = set()
    if logs_root.is_dir():
        for metrics_path in logs_root.glob("*/metrics.json"):
            if not _legacy_metric(_json_object(metrics_path)):
                continue
            legacy_run_names.add(metrics_path.parent.name)
            candidates.append(
                CleanupCandidate(
                    metrics_path.parent.resolve(),
                    "legacy_task_artifact",
                    "executed pre-pivot metrics directory; opt-in required",
                    directory_size(metrics_path.parent),
                    protected=not enabled,
                )
            )
    figures_root = root / "outputs" / "figures"
    for run_name in sorted(legacy_run_names):
        directory = figures_root / run_name
        if directory.is_dir():
            candidates.append(
                CleanupCandidate(
                    directory.resolve(),
                    "legacy_task_figure",
                    "figure directory matched to an evidenced pre-pivot metrics run",
                    directory_size(directory),
                    protected=not enabled,
                )
            )
    checkpoint_root = root / "outputs" / "checkpoints"
    if checkpoint_root.is_dir():
        try:
            import torch
        except ImportError:  # pragma: no cover - project depends on torch
            torch = None
        if torch is not None:
            for path in checkpoint_root.glob("**/*"):
                if not path.is_file() or path.suffix.lower() not in {".pt", ".pth", ".ckpt"}:
                    continue
                try:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                except Exception:  # noqa: BLE001 - corrupt/foreign checkpoints are simply ineligible
                    payload = None
                if payload is None:
                    continue
                if isinstance(payload, dict) and payload.get("task") not in RECONSTRUCTION_TASKS and (
                    "class_names" in payload
                    or "num_classes" in json.dumps(payload.get("model_kwargs", {}), default=str)
                ):
                    candidates.append(
                        CleanupCandidate(
                            path.resolve(),
                            "legacy_task_checkpoint",
                            "pre-pivot checkpoint; opt-in required",
                            path.stat().st_size,
                            protected=not enabled,
                        )
                    )
    for name in LEGACY_TABLE_NAMES:
        path = root / "outputs" / "tables" / name
        if path.is_file():
            candidates.append(
                CleanupCandidate(
                    path.resolve(),
                    "legacy_task_table",
                    "known pre-pivot aggregate table; opt-in required",
                    path.stat().st_size,
                    protected=not enabled,
                )
            )
    return candidates


def discover_cleanup_candidates(
    project_root: str | Path, *, include_legacy: bool = False
) -> list[CleanupCandidate]:
    """Discover only allowlisted archives, temp paths, extracts, and old checkpoints."""

    root = Path(project_root).resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError(f"Project root lacks pyproject.toml: {root}")
    candidates: list[CleanupCandidate] = []
    processed_exists = _has_processed_evidence(root)

    raw = root / "data" / "raw"
    for name in KNOWN_ARCHIVES:
        path = raw / name
        if not path.is_file():
            continue
        is_legacy_archive = name.lower().startswith("multisubjects")
        protected = (not processed_exists) or (is_legacy_archive and not include_legacy)
        reason = (
            "known legacy source archive; --include-legacy is required"
            if is_legacy_archive and not include_legacy
            else "known source archive; protected until compact processed evidence exists"
            if protected
            else "known dataset archive after preprocessing"
        )
        candidates.append(
            CleanupCandidate(path.resolve(), "archive", reason, path.stat().st_size, protected)
        )
    for name in KNOWN_EXTRACTED_DIRS:
        path = raw / name
        if path.is_dir():
            candidates.append(
                CleanupCandidate(
                    path.resolve(),
                    "extracted_raw",
                    "known extracted raw directory",
                    directory_size(path),
                    not processed_exists,
                )
            )
    for components in KNOWN_TEMP_DIRS:
        path = root.joinpath(*components)
        if path.exists():
            candidates.append(
                CleanupCandidate(path.resolve(), "temporary", "known temporary/cache path", directory_size(path))
            )

    checkpoint_root = root / "outputs" / "checkpoints"
    if checkpoint_root.is_dir():
        by_directory: dict[Path, list[Path]] = {}
        for path in checkpoint_root.glob("**/*"):
            if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
                by_directory.setdefault(path.parent, []).append(path)
        for directory, checkpoint_files in by_directory.items():
            best_files = {
                path for path in checkpoint_files if path.stem.lower().startswith("best")
            }
            newest = max(checkpoint_files, key=lambda item: item.stat().st_mtime)
            for path in checkpoint_files:
                lower = path.name.lower()
                old_pattern = (
                    lower in OLD_CHECKPOINT_NAMES
                    or lower.startswith(("epoch_", "checkpoint_epoch"))
                )
                if not old_pattern or path in best_files:
                    continue
                # If there is no explicit best checkpoint, retain the newest file.
                if not best_files and path == newest:
                    continue
                candidates.append(
                    CleanupCandidate(
                        path.resolve(),
                        "old_checkpoint",
                        f"superseded checkpoint in {_relative_directory(directory, root)}; best/newest retained",
                        path.stat().st_size,
                    )
                )

    candidates.extend(_legacy_candidates(root, enabled=include_legacy))
    unique: dict[Path, CleanupCandidate] = {}
    for candidate in candidates:
        validated = validate_target(root, candidate.path)
        unique[validated] = candidate
    return sorted(unique.values(), key=lambda item: (item.category, str(item.path).lower()))


def _relative_directory(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def remove_candidates(
    candidates: Sequence[CleanupCandidate], project_root: str | Path
) -> tuple[int, int]:
    """Permanently remove unprotected, validated candidates and return count/bytes."""

    root = Path(project_root).resolve()
    removed_count = 0
    removed_bytes = 0
    for candidate in candidates:
        path = validate_target(root, candidate.path)
        if candidate.protected:
            print(f"PROTECTED (not removed): {path} -- {candidate.reason}")
            continue
        if not path.exists():
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed_count += 1
        removed_bytes += candidate.size_bytes
        print(f"REMOVED PERMANENTLY: {path} ({human_bytes(candidate.size_bytes)})")
    return removed_count, removed_bytes


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Permanently delete unprotected allowlisted candidates (default is dry-run)",
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Make specifically evidenced pre-pivot artifacts eligible (still dry-run without --confirm)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve()
    candidates = discover_cleanup_candidates(root, include_legacy=args.include_legacy)
    print("Storage before cleanup:")
    print_storage_report(root)
    if not candidates:
        print("No allowlisted cleanup candidates found.")
        return 0
    print("Cleanup candidates:")
    for candidate in candidates:
        status = "PROTECTED" if candidate.protected else "candidate"
        print(
            f"  [{candidate.category}; {status}] {candidate.path} -- "
            f"{human_bytes(candidate.size_bytes)}; {candidate.reason}"
        )
    removable_size = sum(item.size_bytes for item in candidates if not item.protected)
    if not args.confirm:
        print(
            f"DRY RUN: {human_bytes(removable_size)} is eligible. Nothing deleted; "
            "rerun with --confirm for permanent removal."
        )
        return 0
    removed_count, removed_bytes = remove_candidates(candidates, root)
    print(
        f"Removed {removed_count} allowlisted target(s), {human_bytes(removed_bytes)}. "
        "Deletion is permanent and was explicitly confirmed."
    )
    print("Storage after cleanup:")
    print_storage_report(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
