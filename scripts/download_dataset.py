#!/usr/bin/env python
"""Safely acquire the official TrackID3x3 annotation repository only.

This entry point intentionally has no Hugging Face or Henu code. It never
follows the upstream Google Drive link and never downloads videos or weights.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "open-starlab/TrackID3x3"
CLONE_URL = "https://github.com/open-starlab/TrackID3x3.git"
GITHUB_API = "https://api.github.com/repos/open-starlab/TrackID3x3"
DEFAULT_DESTINATION = PROJECT_ROOT / "external" / "TrackID3x3"
SPARSE_PATHS = ("ground_truth", "court_images", "script")
MAX_DATASET_MB = 600.0
TARGET_DATASET_MB = 500.0


def directory_bytes(path: Path) -> int:
    """Return recursive on-disk file bytes without following symlinks."""

    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink()
    )


def github_repository_size_mb() -> float | None:
    """Read GitHub's repository-size metadata; unknown means no auto-clone."""

    request = urllib.request.Request(
        GITHUB_API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "trackid3x3-budget"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except (OSError, ValueError):
        return None
    size_kib = payload.get("size")
    return float(size_kib) / 1024.0 if isinstance(size_kib, (int, float)) else None


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def clone_annotations(destination: Path) -> None:
    """Make a partial, shallow clone and materialize annotation paths only."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    gh = shutil.which("gh")
    if gh:
        command = [
            gh,
            "repo",
            "clone",
            REPOSITORY,
            str(destination),
            "--",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
        ]
    else:
        git = shutil.which("git")
        if not git:
            raise RuntimeError("Neither `gh` nor `git` is installed.")
        print("GitHub CLI is unavailable; using equivalent safe `git clone` fallback.")
        command = [
            git,
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            CLONE_URL,
            str(destination),
        ]
    _run(command)
    _run(["git", "-C", str(destination), "sparse-checkout", "set", *SPARSE_PATHS])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        remote = subprocess.run(
            ["git", "-C", str(destination), "remote", "get-url", "origin"],
            check=False,
            capture_output=True,
            text=True,
        )
        if remote.returncode or "open-starlab/TrackID3x3" not in remote.stdout:
            print(f"Refusing to reuse unverified destination: {destination}", file=sys.stderr)
            return 2
        print(f"Official TrackID3x3 clone already present: {destination}")
        print(f"Current clone size: {directory_bytes(destination) / 1024**2:.2f} MiB")
        print("No download performed.")
        return 0

    predicted_mb = github_repository_size_mb()
    current_mb = directory_bytes(PROJECT_ROOT) / 1024**2
    remaining_mb = MAX_DATASET_MB - current_mb
    print(f"Repository: {REPOSITORY}")
    print(f"Plan: shallow partial clone; sparse paths: {', '.join(SPARSE_PATHS)}")
    print("Excluded: Google Drive media, full videos, tracker outputs, pretrained weights")
    print(f"Current project footprint: {current_mb:.2f} MiB")
    print(
        "GitHub repository size metadata: "
        + (f"{predicted_mb:.2f} MiB" if predicted_mb is not None else "unknown")
    )
    print(f"Remaining hard-limit budget: {remaining_mb:.2f} MiB")
    if predicted_mb is None:
        print("Remote size is unknown; refusing automatic acquisition.", file=sys.stderr)
        return 2
    if predicted_mb > remaining_mb:
        print("Predicted clone could exceed the 600 MiB hard limit; aborted.", file=sys.stderr)
        return 2
    if args.dry_run:
        print("Dry run: preflight passed; nothing downloaded.")
        return 0
    clone_annotations(destination)
    actual_mb = directory_bytes(destination) / 1024**2
    total_mb = directory_bytes(PROJECT_ROOT) / 1024**2
    print(f"Annotation checkout size: {actual_mb:.2f} MiB")
    print(f"Project footprint after clone: {total_mb:.2f} MiB")
    if total_mb > MAX_DATASET_MB:
        raise RuntimeError("Post-clone footprint exceeded the hard limit; inspect manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
