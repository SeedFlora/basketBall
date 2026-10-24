#!/usr/bin/env python
"""Replicate the executed TrackID3x3 experiments over additional seeds.

Every command mirrors the seed-42 command documented in README.md and changes only the seed
and the run name. Training writes to the usual ``outputs/logs/<name>`` and
``outputs/checkpoints/<name>``. Evaluations, including a re-evaluation of the existing seed-42
checkpoints, go to ``outputs/multiseed/evaluations`` so that ``run_ablation.py``, which builds the
canonical seed-42 tables from ``outputs/logs``, never sees a duplicate seed-42 evaluation.

The training seed and the evaluation seed are always equal, so each seed is one complete
replicate: weight initialization, batch order, the fixed per-sample training and validation
masks, the evaluation masks, and (for train fractions below one) which whole groups are kept.
The deterministic subset and the group-disjoint splits are held fixed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC = Path("data/processed/trackid3x3/dataset.npz")
TEMPORAL = Path("data/processed/trackid3x3_t16/dataset.npz")
ROTATION = Path("data/interim/trackid3x3_rotation")
EVALUATIONS = Path("outputs/multiseed/evaluations")
SCRATCH = Path("tmp/multiseed")
ALL_RATIOS = ("0.10", "0.20", "0.30", "0.40")
FAMILIES = ("core", "temporal", "dataeff", "domain", "shift", "sizematch", "rotation")
DEFAULT_FAMILIES = ("core", "temporal", "dataeff", "domain", "shift")


@dataclass
class Unit:
    """One optional training run followed by the evaluations that consume it."""

    name: str
    family: str
    train: list[str] | None
    evaluations: list[tuple[str, list[str]]] = field(default_factory=list)


def _train_command(
    name: str,
    *,
    processed: Path,
    model: str,
    representation: str,
    domain: str,
    batch_size: int,
    seed: int,
    device: str,
    train_fraction: float = 1.0,
) -> list[str]:
    return [
        "scripts/train.py",
        "--config", "config/trackid3x3.yaml",
        "--processed", str(processed),
        "--model", model,
        "--representation", representation,
        "--mask-ratio", "0.20",
        "--domain", domain,
        "--train-fraction", f"{train_fraction:g}",
        "--epochs", "30",
        "--batch-size", str(batch_size),
        "--num-workers", "0",
        "--seed", str(seed),
        "--device", device,
        "--experiment-name", name,
    ]


def _evaluate_command(
    run_name: str,
    *,
    processed: Path,
    seed: int,
    device: str,
    family: str,
    ratios: Sequence[str] = ("0.20",),
    checkpoint: str | None = None,
    baseline: str | None = None,
    test_domain: str = "all",
    batch_size: int = 256,
) -> list[str]:
    source = (
        ["--checkpoint", f"outputs/checkpoints/{checkpoint}/best.pt"]
        if checkpoint is not None
        else ["--baseline", str(baseline), "--representation", "bbox_normalized"]
    )
    return [
        "scripts/evaluate.py",
        "--processed", str(processed),
        *source,
        "--mask-ratios", *ratios,
        "--primary-mask-ratio", "0.20",
        "--split", "test",
        "--test-domain", test_domain,
        "--seed", str(seed),
        "--batch-size", str(batch_size),
        "--device", device,
        "--output-dir", str(EVALUATIONS),
        "--figures-dir", str(SCRATCH / "figures" / family),
        "--run-name", run_name,
    ]


def _trained(
    family: str,
    name: str,
    seed: int,
    device: str,
    *,
    processed: Path,
    model: str,
    representation: str,
    domain: str,
    batch_size: int,
    eval_batch_size: int = 256,
    ratios: Sequence[str] = ("0.20",),
    test_domain: str = "all",
    train_fraction: float = 1.0,
) -> Unit:
    return Unit(
        name=name,
        family=family,
        train=_train_command(
            name,
            processed=processed,
            model=model,
            representation=representation,
            domain=domain,
            batch_size=batch_size,
            seed=seed,
            device=device,
            train_fraction=train_fraction,
        ),
        evaluations=[
            (
                f"{name}_eval",
                _evaluate_command(
                    f"{name}_eval",
                    processed=processed,
                    seed=seed,
                    device=device,
                    family=family,
                    ratios=ratios,
                    checkpoint=name,
                    test_domain=test_domain,
                    batch_size=eval_batch_size,
                ),
            )
        ],
    )


def build_units(
    seeds: Sequence[int],
    families: Sequence[str],
    device: str,
    sizematch_fraction: float,
    transfer_representation: str = "bbox_normalized",
) -> list[Unit]:
    """Return the replicate plan, one unit per (experiment, seed).

    ``transfer_representation`` sets the input of the domain, sizematch, and rotation families.
    """

    units: list[Unit] = []
    short_transfer = {"bbox_normalized": "bbox", "scale_normalized": "scale"}[transfer_representation]
    for seed in seeds:
        if "core" in families:
            for short, representation, ratios in (
                ("raw", "raw", ("0.20",)),
                ("bbox", "bbox_normalized", ALL_RATIOS),
                ("root", "root_centered", ("0.20",)),
                ("scale", "scale_normalized", ("0.20",)),
                ("angles", "normalized_angles", ALL_RATIOS),
            ):
                units.append(
                    _trained(
                        "core", f"core_mlp_{short}_all_mask20_seed{seed}", seed, device,
                        processed=STATIC, model="mlp", representation=representation,
                        domain="all", batch_size=128, ratios=ratios,
                    )
                )
            units.append(
                _trained(
                    "core", f"core_graph_bbox_all_mask20_seed{seed}", seed, device,
                    processed=STATIC, model="graph", representation="bbox_normalized",
                    domain="all", batch_size=128,
                )
            )
            name = f"core_geometric_all_seed{seed}"
            units.append(
                Unit(name, "core", None, [(name, _evaluate_command(
                    name, processed=STATIC, seed=seed, device=device, family="core",
                    ratios=ALL_RATIOS, baseline="geometric",
                ))])
            )
        if "temporal" in families:
            for model in ("mlp", "graph", "tcn"):
                units.append(
                    _trained(
                        "temporal", f"core_{model}_bbox_all_t16_mask20_seed{seed}", seed, device,
                        processed=TEMPORAL, model=model, representation="bbox_normalized",
                        domain="all", batch_size=32, eval_batch_size=64,
                    )
                )
            # Training-free references for windows: spatial neighbors and linear interpolation in time.
            for baseline in ("geometric", "temporal"):
                name = f"core_{baseline}_all_t16_seed{seed}"
                units.append(
                    Unit(name, "temporal", None, [(name, _evaluate_command(
                        name, processed=TEMPORAL, seed=seed, device=device, family="temporal",
                        baseline=baseline, batch_size=64,
                    ))])
                )
        if "dataeff" in families:
            # The 100% point is re-run at the same batch size as 25% and 50% (64), so the
            # data-efficiency curve varies only the retained groups.
            for percent, fraction in ((25, 0.25), (50, 0.50), (100, 1.0)):
                units.append(
                    _trained(
                        "dataeff", f"dataeff_mlp_angles_all_data{percent}_mask20_seed{seed}",
                        seed, device, processed=STATIC, model="mlp",
                        representation="normalized_angles", domain="all", batch_size=64,
                        train_fraction=fraction,
                    )
                )
        if "domain" in families:
            for domain in ("Indoor", "Outdoor", "Drone"):
                units.append(
                    _trained(
                        "domain", f"domain_mlp_{short_transfer}_{domain.lower()}_mask20_seed{seed}", seed,
                        device, processed=STATIC, model="mlp", representation=transfer_representation,
                        domain=domain, batch_size=64,
                    )
                )
        if "shift" in families:
            for short, representation in (("raw", "raw"), ("scale", "scale_normalized")):
                units.append(
                    _trained(
                        "shift", f"shift_mlp_{short}_indoor_mask20_seed{seed}", seed, device,
                        processed=STATIC, model="mlp", representation=representation,
                        domain="Indoor", batch_size=64,
                    )
                )
        if "sizematch" in families:
            percent = round(100 * sizematch_fraction)
            for domain in ("Indoor", "Outdoor"):
                units.append(
                    _trained(
                        "sizematch",
                        f"sizematch_mlp_{short_transfer}_{domain.lower()}_data{percent}_mask20_seed{seed}",
                        seed, device, processed=STATIC, model="mlp",
                        representation=transfer_representation, domain=domain, batch_size=64,
                        train_fraction=sizematch_fraction,
                    )
                )
        if "rotation" in families:
            # Group-rotation folds written by build_rotation_folds.py; one per test group.
            for fold in sorted(path.name for path in (PROJECT_ROOT / ROTATION).glob("fold*")):
                for domain in ("Indoor", "Outdoor", "Drone"):
                    units.append(
                        _trained(
                            "rotation",
                            f"rotation_{fold}_mlp_{short_transfer}_{domain.lower()}_mask20_seed{seed}",
                            seed, device, processed=ROTATION / fold / "dataset.npz", model="mlp",
                            representation=transfer_representation, domain=domain, batch_size=64,
                        )
                    )
    return units


def _trained_already(name: str) -> bool:
    return (PROJECT_ROOT / "outputs" / "checkpoints" / name / "best.pt").is_file() and (
        PROJECT_ROOT / "outputs" / "logs" / name / "metrics.json"
    ).is_file()


def _evaluated_already(run_name: str) -> bool:
    return (PROJECT_ROOT / EVALUATIONS / run_name / "evaluation_index.json").is_file()


def _run(label: str, command: list[str]) -> tuple[bool, float]:
    log_path = PROJECT_ROOT / SCRATCH / "logs" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            [sys.executable, *command],
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode == 0, time.perf_counter() - started


def execute(unit: Unit) -> list[str]:
    """Train if needed, then evaluate; return one status line per step."""

    lines: list[str] = []
    if unit.train is not None and not _trained_already(unit.name):
        ok, seconds = _run(unit.name, unit.train)
        lines.append(f"{'ok  ' if ok else 'FAIL'} train {unit.name} ({seconds:.0f}s)")
        if not ok:
            return lines
    for run_name, command in unit.evaluations:
        if _evaluated_already(run_name):
            lines.append(f"skip eval  {run_name}")
            continue
        ok, seconds = _run(run_name, command)
        lines.append(f"{'ok  ' if ok else 'FAIL'} eval  {run_name} ({seconds:.0f}s)")
    return lines


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(DEFAULT_FAMILIES))
    parser.add_argument("--sizematch-fraction", type=float, default=0.5)
    parser.add_argument(
        "--transfer-representation",
        choices=("bbox_normalized", "scale_normalized"),
        default="bbox_normalized",
        help="Input representation of the domain, sizematch, and rotation families",
    )
    parser.add_argument("--jobs", type=int, default=4, help="Concurrent units (each is one process)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without running it")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    units = build_units(
        args.seeds, args.families, args.device, args.sizematch_fraction, args.transfer_representation
    )
    if args.dry_run:
        for unit in units:
            state = "trained" if unit.train is None or _trained_already(unit.name) else "to train"
            print(f"{unit.family:<9} {unit.name:<52} {state}; evals: {len(unit.evaluations)}")
        print(f"{len(units)} units")
        return 0
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(execute, unit) for unit in units]
        for future in as_completed(futures):
            for line in future.result():
                failures += line.startswith("FAIL")
                print(line, flush=True)
    print(f"done: {len(units)} units, {failures} failed steps")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
