"""Shared configuration, reproducibility, environment, and storage utilities."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    merged: dict[str, Any] = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML with an optional project-relative ``extends`` chain."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    parent = current.pop("extends", None)
    if parent is None:
        return current
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    return deep_merge(load_config(parent_path), current)


def save_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    """Write a resolved configuration."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch, and CUDA consistently."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_info(project_root: str | Path = ".") -> dict[str, Any]:
    """Capture the software and hardware context needed to reproduce a run."""
    root = Path(project_root).resolve()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    try:
        import psutil  # type: ignore

        ram_bytes = int(psutil.virtual_memory().total)
    except ImportError:
        ram_bytes = None
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "ram_bytes": ram_bytes,
        "os": platform.platform(),
        "git_commit": _git_commit(root),
    }


def directory_size(path: str | Path) -> int:
    """Return recursive on-disk file bytes for an existing path."""
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GiB"


def storage_report(project_root: str | Path = ".") -> dict[str, int]:
    """Compute the mandatory raw/processed/checkpoint/total size audit."""
    root = Path(project_root).resolve()
    outputs = root / "outputs"
    raw_bytes = directory_size(root / "external" / "TrackID3x3" / "ground_truth")
    raw_bytes += directory_size(root / "data" / "raw")
    checkpoint_bytes = (
        sum(
            item.stat().st_size
            for item in outputs.rglob("*")
            if item.is_file() and item.suffix.lower() in {".pt", ".pth"}
        )
        if outputs.exists()
        else 0
    )
    return {
        "raw_dataset_bytes": raw_bytes,
        "processed_dataset_bytes": directory_size(root / "data" / "processed"),
        "model_checkpoint_bytes": checkpoint_bytes,
        "total_project_artifact_bytes": directory_size(root),
    }


def print_storage_report(project_root: str | Path = ".") -> dict[str, int]:
    report = storage_report(project_root)
    labels = {
        "raw_dataset_bytes": "Raw dataset size",
        "processed_dataset_bytes": "Processed dataset size",
        "model_checkpoint_bytes": "Model checkpoint size",
        "total_project_artifact_bytes": "Total project artifact size",
    }
    for key, value in report.items():
        print(f"{labels[key]}: {human_bytes(value)} ({value} bytes)")
    return report


def write_json(data: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
