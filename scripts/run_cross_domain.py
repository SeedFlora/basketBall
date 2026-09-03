#!/usr/bin/env python
"""Evaluate executed reconstruction checkpoints within and across TrackID domains."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import (
    DOMAINS,
    ReconstructionEvaluationError,
    evaluate,
)
from scripts.evaluate import (
    parse_args as parse_evaluate_args,
)
from scripts.run_ablation import run_ablation


def _nested(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _checkpoint_descriptor(path: Path) -> dict[str, Any] | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - discovery safely ignores corrupt/foreign files
        return None
    if not isinstance(payload, Mapping) or payload.get("task") != "keypoint_reconstruction":
        return None
    experiment = payload.get("experiment") if isinstance(payload.get("experiment"), Mapping) else {}
    train_domain = experiment.get("train_domain", experiment.get("domain"))
    representation = experiment.get(
        "representation",
        _nested(payload, "config.representation", _nested(payload, "config.features.representation")),
    )
    return {
        "path": path.resolve(),
        "model": str(payload.get("model_name", "unknown")),
        "representation": str(representation or "unknown"),
        "train_domain": None if train_domain is None else str(train_domain),
        "synthetic": bool(payload.get("synthetic", False)),
        "smoke": bool(payload.get("smoke", False) or "smoke" in str(path).lower()),
    }


def discover_reconstruction_checkpoints(
    root: str | Path,
    *,
    include_synthetic: bool = False,
    include_smoke: bool = False,
) -> list[dict[str, Any]]:
    """Return valid reconstruction checkpoints with explicit training-domain provenance."""

    directory = Path(root)
    descriptors: list[dict[str, Any]] = []
    if not directory.exists():
        return descriptors
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".pt", ".pth", ".ckpt"}:
            continue
        descriptor = _checkpoint_descriptor(path)
        if descriptor is None or descriptor["train_domain"] not in DOMAINS:
            continue
        if descriptor["synthetic"] and not include_synthetic:
            continue
        if descriptor["smoke"] and not include_smoke:
            continue
        descriptors.append(descriptor)
    # Retain one explicit best checkpoint per run directory when duplicates exist.
    by_run: dict[tuple[Path, str, str, str], dict[str, Any]] = {}
    for descriptor in descriptors:
        path = Path(descriptor["path"])
        key = (path.parent, descriptor["model"], descriptor["representation"], descriptor["train_domain"])
        current = by_run.get(key)
        if current is None or path.stem.lower().startswith("best"):
            by_run[key] = descriptor
    return sorted(by_run.values(), key=lambda item: str(item["path"]))


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_cross_domain(args: argparse.Namespace) -> Path:
    checkpoints_root = _absolute(args.checkpoints_root)
    processed = _absolute(args.processed)
    logs_dir = _absolute(args.output_dir)
    figures_dir = _absolute(args.figures_dir)
    tables_dir = _absolute(args.tables_dir)
    descriptors = discover_reconstruction_checkpoints(
        checkpoints_root,
        include_synthetic=args.include_synthetic,
        include_smoke=args.include_smoke,
    )
    requested_domains = tuple(args.domains)
    unknown = set(requested_domains) - set(DOMAINS)
    if unknown:
        raise ReconstructionEvaluationError(f"unknown TrackID domains: {sorted(unknown)}")
    if args.models:
        descriptors = [item for item in descriptors if item["model"] in set(args.models)]
    if args.representations:
        descriptors = [
            item for item in descriptors if item["representation"] in set(args.representations)
        ]
    if not descriptors:
        raise ReconstructionEvaluationError(
            "No reconstruction checkpoint with explicit Indoor/Outdoor/Drone train_domain was found"
        )

    completed: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for descriptor in descriptors:
        checkpoint = Path(descriptor["path"])
        for test_domain in requested_domains:
            run_name = (
                f"{checkpoint.parent.name}__train_{descriptor['train_domain'].lower()}"
                f"__test_{test_domain.lower()}"
            )
            command = [
                "--processed",
                str(processed),
                "--checkpoint",
                str(checkpoint),
                "--test-domain",
                test_domain,
                "--train-domain",
                str(descriptor["train_domain"]),
                "--mask-ratios",
                *[str(value) for value in args.mask_ratios],
                "--seed",
                str(args.seed),
                "--batch-size",
                str(args.batch_size),
                "--output-dir",
                str(logs_dir),
                "--figures-dir",
                str(figures_dir),
                "--run-name",
                run_name,
            ]
            if args.device:
                command.extend(["--device", args.device])
            if args.include_synthetic:
                command.append("--include-synthetic")
            if args.no_save_predictions:
                command.append("--no-save-predictions")
            try:
                result = evaluate(parse_evaluate_args(command))
                completed.append(
                    {
                        "checkpoint": str(checkpoint),
                        "model": descriptor["model"],
                        "representation": descriptor["representation"],
                        "train_domain": descriptor["train_domain"],
                        "test_domain": test_domain,
                        "run_name": run_name,
                        "synthetic": result["synthetic"],
                    }
                )
            except Exception as exc:
                failed.append(
                    {
                        "checkpoint": str(checkpoint),
                        "test_domain": test_domain,
                        "error": str(exc),
                    }
                )
                if not args.continue_on_error:
                    raise

    ablation_paths = run_ablation(
        logs_dir,
        tables_dir,
        figures_dir,
        data_path=processed,
        include_synthetic=args.include_synthetic,
        include_smoke=args.include_smoke,
    )
    index = {
        "artifact_type": "cross_domain_evaluation_index",
        "task": "keypoint_reconstruction",
        "processed_dataset": str(processed.resolve()),
        "checkpoint_root": str(checkpoints_root.resolve()),
        "mask_ratios": list(args.mask_ratios),
        "domains": list(requested_domains),
        "seed": args.seed,
        "completed": completed,
        "failed": failed,
        "include_synthetic": args.include_synthetic,
        "include_smoke": args.include_smoke,
        "synthetic_label": "software validation only / NOT research results",
        "ablation_index": str(ablation_paths["index"].resolve()),
        "rgb_overlay": "NOT GENERATED: source RGB frames are not present locally; remote media size was not assumed.",
    }
    index_path = tables_dir / "cross_domain_evaluation_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return index_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed/trackid3x3/dataset.npz"))
    parser.add_argument("--checkpoints-root", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS))
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--representations", nargs="+")
    parser.add_argument("--mask-ratios", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.40])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/logs/reconstruction_evaluation"))
    parser.add_argument("--figures-dir", type=Path, default=Path("outputs/figures/reconstruction"))
    parser.add_argument("--tables-dir", type=Path, default=Path("outputs/tables/reconstruction"))
    parser.add_argument("--include-synthetic", action="store_true")
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-save-predictions", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        path = run_cross_domain(parse_args(argv))
    except (ReconstructionEvaluationError, ValueError, RuntimeError) as exc:
        print(f"Cross-domain evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote cross-domain evaluation index: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
