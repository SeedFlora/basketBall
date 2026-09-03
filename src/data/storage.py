"""Storage accounting and write preflight helpers for TrackID3x3 artifacts."""

from __future__ import annotations

import csv
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

STORAGE_CATEGORIES = (
    "annotations",
    "videos",
    "processed",
    "checkpoints",
    "figures",
    "temporary",
)
MAX_DATASET_MB = 600
TARGET_DATASET_MB = 500
# Backward-compatible descriptive aliases used by storage callers.
MAX_STORAGE_MB = MAX_DATASET_MB
TARGET_STORAGE_MB = TARGET_DATASET_MB
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class StorageBudgetError(RuntimeError):
    """Raised before a write that exceeds a configured or physical budget."""


@dataclass(frozen=True, slots=True)
class StorageEntry:
    path: str
    category: str
    size_mb: float
    required: bool
    source: str


def directory_bytes(path: str | Path, *, suffixes: set[str] | None = None) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        if suffixes is not None and root.suffix.casefold() not in suffixes:
            return 0
        return root.stat().st_size
    total = 0
    for item in root.rglob("*"):
        if not item.is_file():
            continue
        if suffixes is None or item.suffix.casefold() in suffixes:
            total += item.stat().st_size
    return total


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _entry(
    path: Path,
    category: str,
    required: bool,
    source: str,
    root: Path,
    *,
    byte_count: int | None = None,
) -> StorageEntry:
    if category not in STORAGE_CATEGORIES:
        raise ValueError(f"Unknown storage category {category!r}.")
    size = directory_bytes(path) if byte_count is None else int(byte_count)
    return StorageEntry(_relative(path, root), category, round(size / 1_000_000, 6), required, source)


def build_storage_manifest(project_root: str | Path = ".") -> list[StorageEntry]:
    """Account for source annotations, checkout overhead, and generated data."""

    root = Path(project_root).resolve()
    checkout = root / "external" / "TrackID3x3"
    ground_truth = checkout / "ground_truth"
    video_bytes = directory_bytes(checkout, suffixes=VIDEO_SUFFIXES)
    checkpoint_bytes = directory_bytes(root / "outputs", suffixes={".pt", ".pth", ".ckpt"})
    git_bytes = directory_bytes(checkout / ".git")
    checkout_overhead = max(
        0, directory_bytes(checkout) - directory_bytes(ground_truth) - git_bytes - video_bytes
    )
    processed_root = root / "data" / "processed"
    output_root = root / "outputs"
    figure_bytes = directory_bytes(output_root / "figures")
    other_outputs = max(0, directory_bytes(output_root) - checkpoint_bytes - figure_bytes)
    entries = [
        _entry(
            ground_truth,
            "annotations",
            True,
            "TrackID3x3 GitHub ground_truth annotations",
            root,
        ),
        _entry(
            root / "data" / "raw",
            "annotations",
            False,
            "Local raw staging/legacy permitted small artifacts",
            root,
        ),
        _entry(
            checkout,
            "videos",
            False,
            "TrackID3x3 RGB video payload (not required; zero when absent)",
            root,
            byte_count=video_bytes,
        ),
        _entry(
            processed_root,
            "processed",
            True,
            "All generated processed datasets, including TrackID3x3 static/smoke/temporal variants",
            root,
        ),
        _entry(
            root / "outputs",
            "checkpoints",
            False,
            "Generated model checkpoint files only",
            root,
            byte_count=checkpoint_bytes,
        ),
        _entry(
            root / "outputs" / "figures",
            "figures",
            False,
            "Generated figures",
            root,
        ),
        _entry(
            output_root,
            "temporary",
            False,
            "Reports, tables, and logs (checkpoint/figure bytes excluded)",
            root,
            byte_count=other_outputs,
        ),
        _entry(
            checkout / ".git",
            "temporary",
            False,
            "Git checkout metadata/source overhead (reported separately)",
            root,
            byte_count=git_bytes,
        ),
        _entry(
            checkout,
            "temporary",
            False,
            "Checkout source/license/court-image overhead (ground_truth/.git/videos excluded)",
            root,
            byte_count=checkout_overhead,
        ),
        _entry(
            root / "data" / "interim",
            "temporary",
            False,
            "Interim and legacy software-fixture data",
            root,
        ),
    ]
    return entries


def write_storage_manifest(
    path: str | Path, entries: Sequence[StorageEntry]
) -> Path:
    """Write exact columns: path,category,size_mb,required,source."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("path", "category", "size_mb", "required", "source")
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))
    return output


def estimate_trackid_processed_bytes(
    sample_count: int,
    sequence_length: int,
    *,
    joints: int = 10,
    coordinates: int = 2,
) -> int:
    """Conservative uncompressed estimate for all stored representations."""

    if sample_count < 0 or sequence_length <= 0 or joints <= 0 or coordinates <= 0:
        raise ValueError("Counts and dimensions must be positive (sample_count may be zero).")
    pose_values = sample_count * sequence_length * joints * coordinates
    joint_values = sample_count * sequence_length * joints
    # coords/input + raw/bbox/root/scale, plus room for transforms, bboxes,
    # angles, masks, and Unicode identity/provenance arrays.  NPZ compression
    # normally makes the result smaller; the safety factor covers metadata.
    numeric = pose_values * 4 * 6
    masks = joint_values * 8
    transforms_and_boxes = sample_count * sequence_length * 4 * 4 * 5
    strings_and_metadata = sample_count * sequence_length * 512 + 1_000_000
    return int((numeric + masks + transforms_and_boxes + strings_and_metadata) * 1.25)


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    return candidate


def storage_preflight(
    output_path: str | Path,
    estimated_bytes: int,
    *,
    max_artifact_bytes: int,
    minimum_free_bytes: int,
) -> dict[str, int | str | bool]:
    """Fail before writing if configured size or free-space limits are unsafe."""

    if min(estimated_bytes, max_artifact_bytes, minimum_free_bytes) < 0:
        raise ValueError("Storage byte limits must be non-negative.")
    destination = Path(output_path)
    volume_path = _nearest_existing_parent(destination)
    usage = shutil.disk_usage(volume_path)
    remaining = usage.free - estimated_bytes
    within_artifact_budget = estimated_bytes <= max_artifact_bytes
    preserves_free_reserve = remaining >= minimum_free_bytes
    report: dict[str, int | str | bool] = {
        "output_path": str(destination.resolve()),
        "volume_path": str(volume_path),
        "estimated_bytes": int(estimated_bytes),
        "max_artifact_bytes": int(max_artifact_bytes),
        "free_bytes_before": int(usage.free),
        "free_bytes_after_estimate": int(remaining),
        "minimum_free_bytes": int(minimum_free_bytes),
        "within_artifact_budget": within_artifact_budget,
        "preserves_free_reserve": preserves_free_reserve,
        "approved": within_artifact_budget and preserves_free_reserve,
    }
    if not report["approved"]:
        reasons = []
        if not within_artifact_budget:
            reasons.append("estimated artifact exceeds configured maximum")
        if not preserves_free_reserve:
            reasons.append("write would violate minimum free-space reserve")
        raise StorageBudgetError("Storage preflight failed: " + "; ".join(reasons))
    return report


def manifest_totals(entries: Iterable[StorageEntry]) -> dict[str, float]:
    totals = {category: 0.0 for category in STORAGE_CATEGORIES}
    for entry in entries:
        totals[entry.category] += entry.size_mb
    return {key: round(value, 6) for key, value in totals.items()}


def current_storage_budget(project_root: str | Path = ".") -> dict[str, float]:
    """Return non-overlapping project-data totals against the 600 MB maximum."""

    root = Path(project_root).resolve()
    checkout = root / "external" / "TrackID3x3"
    ground_truth_bytes = directory_bytes(checkout / "ground_truth")
    local_raw_bytes = directory_bytes(root / "data" / "raw")
    processed_bytes = directory_bytes(root / "data" / "processed")
    output_bytes = directory_bytes(root / "outputs")
    checkout_bytes = directory_bytes(checkout)
    source_overhead_bytes = max(0, checkout_bytes - ground_truth_bytes)
    interim_bytes = directory_bytes(root / "data" / "interim")
    raw_bytes = ground_truth_bytes + local_raw_bytes
    total_bytes = (
        raw_bytes + processed_bytes + output_bytes + source_overhead_bytes + interim_bytes
    )
    divisor = 1_000_000.0
    return {
        "current_raw_data_mb": raw_bytes / divisor,
        "current_processed_data_mb": processed_bytes / divisor,
        "current_output_data_mb": output_bytes / divisor,
        "source_overhead_mb": source_overhead_bytes / divisor,
        "temporary_data_mb": interim_bytes / divisor,
        "total_mb": total_bytes / divisor,
        "remaining_budget_mb": MAX_STORAGE_MB - total_bytes / divisor,
        "remaining_target_mb": TARGET_STORAGE_MB - total_bytes / divisor,
        "max_storage_mb": float(MAX_STORAGE_MB),
        "target_storage_mb": float(TARGET_STORAGE_MB),
    }


def project_budget_preflight(
    project_root: str | Path,
    output_path: str | Path,
    estimated_bytes: int,
    *,
    max_dataset_mb: float = MAX_DATASET_MB,
) -> dict[str, float | bool | str]:
    """Check the projected non-overlapping project total before replacement."""

    if estimated_bytes < 0 or max_dataset_mb <= 0:
        raise ValueError("estimated_bytes must be non-negative and max_dataset_mb positive.")
    root = Path(project_root).resolve()
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    try:
        output.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("Budgeted output_path must be inside project_root.") from exc
    current = current_storage_budget(root)
    existing_mb = directory_bytes(output) / 1_000_000.0
    projected_mb = float(current["total_mb"]) - existing_mb + estimated_bytes / 1_000_000.0
    approved = projected_mb <= max_dataset_mb
    report: dict[str, float | bool | str] = {
        "output_path": str(output.resolve()),
        "current_total_mb": float(current["total_mb"]),
        "existing_output_mb_replaced": existing_mb,
        "estimated_new_output_mb": estimated_bytes / 1_000_000.0,
        "projected_total_mb": projected_mb,
        "max_dataset_mb": float(max_dataset_mb),
        "remaining_project_budget_mb": float(max_dataset_mb) - projected_mb,
        "approved": approved,
    }
    if not approved:
        raise StorageBudgetError(
            f"Projected project data {projected_mb:.3f} MB exceeds "
            f"MAX_DATASET_MB={max_dataset_mb:.3f}."
        )
    return report


__all__ = [
    "MAX_DATASET_MB",
    "MAX_STORAGE_MB",
    "STORAGE_CATEGORIES",
    "TARGET_DATASET_MB",
    "TARGET_STORAGE_MB",
    "StorageBudgetError",
    "StorageEntry",
    "build_storage_manifest",
    "current_storage_budget",
    "directory_bytes",
    "estimate_trackid_processed_bytes",
    "manifest_totals",
    "project_budget_preflight",
    "storage_preflight",
    "write_storage_manifest",
]
