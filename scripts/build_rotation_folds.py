#!/usr/bin/env python
"""Write group-rotation copies of the static TrackID3x3 dataset for the transfer control.

The canonical split tests Outdoor and Drone on a single player each. This script keeps the same
selected poses and every array unchanged except ``split``, which it reassigns at the level of
whole split groups: per domain, the sorted groups are cut into ``--folds`` chunks, and fold ``k``
tests chunk ``k``, validates on chunk ``k + 1``, and trains on the rest. Across the folds every
Outdoor and Drone player and every Indoor video is the test group exactly once. Group
disjointness holds by construction, and ``load_trackid_processed`` re-verifies it on load.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.trackid_processed import load_trackid_processed

DOMAINS = ("Indoor", "Outdoor", "Drone")


def fold_assignment(groups: np.ndarray, domains: np.ndarray, fold: int, folds: int) -> dict[str, dict[str, list[str]]]:
    """Return the test/val/train groups of one fold for every domain."""

    assignment: dict[str, dict[str, list[str]]] = {}
    for domain in DOMAINS:
        available = sorted(set(groups[domains == domain].tolist()))
        if len(available) < folds:
            raise ValueError(f"{domain} has {len(available)} groups; cannot form {folds} folds")
        chunks = [list(chunk) for chunk in np.array_split(np.asarray(available), folds)]
        test, val = chunks[fold % folds], chunks[(fold + 1) % folds]
        train = [group for group in available if group not in set(test) | set(val)]
        assignment[domain] = {"test": test, "val": val, "train": train}
    return assignment


def build_fold(source: Path, destination: Path, fold: int, folds: int) -> Path:
    with np.load(source, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    domains = arrays["domain"].astype(str)
    groups = arrays["split_group"].astype(str)
    assignment = fold_assignment(groups, domains, fold, folds)
    split = np.empty(groups.shape, dtype=arrays["split"].dtype)
    for domain, parts in assignment.items():
        for name, members in parts.items():
            split[(domains == domain) & np.isin(groups, members)] = name
    arrays["split"] = split

    destination.mkdir(parents=True, exist_ok=True)
    dataset = destination / "dataset.npz"
    np.savez_compressed(dataset, **arrays)
    load_trackid_processed(dataset, aliases=True)  # re-runs the group-leakage validation

    metadata: dict[str, Any] = json.loads(source.with_name("metadata.json").read_text(encoding="utf-8"))
    counts = metadata.setdefault("counts", {})
    counts["by_split"] = {name: int((split == name).sum()) for name in ("test", "train", "val")}
    counts["by_domain_split"] = {
        domain: {name: int(((domains == domain) & (split == name)).sum()) for name in ("test", "train", "val")}
        for domain in DOMAINS
    }
    metadata["rotation"] = {
        "fold": fold,
        "folds": folds,
        "source_dataset": str(source.relative_to(PROJECT_ROOT)),
        "policy": "whole split groups reassigned; every other array is unchanged",
        "groups": assignment,
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return dataset


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed/trackid3x3/dataset.npz"))
    parser.add_argument("--output-root", type=Path, default=Path("data/interim/trackid3x3_rotation"))
    parser.add_argument("--folds", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = PROJECT_ROOT / args.processed
    for fold in range(args.folds):
        path = build_fold(source, PROJECT_ROOT / args.output_root / f"fold{fold}", fold, args.folds)
        print(f"Wrote {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
