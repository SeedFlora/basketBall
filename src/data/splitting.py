"""Archived subject-based split helpers for pre-pivot compatibility fixtures.

Public TrackID3x3 grouping and leakage audits live in ``trackid_split``.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from .loader import INDEX_TO_LABEL, PoseSample

SPLIT_NAMES = ("train", "val", "test")


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", value))


def group_by_subject(samples: Iterable[PoseSample]) -> dict[str, list[PoseSample]]:
    """Group samples by their validated subject identifier."""

    groups: dict[str, list[PoseSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.subject_id].append(sample)
    return dict(groups)


def select_subjects(
    samples: Sequence[PoseSample], num_subjects: int | None, *, seed: int = 42
) -> list[str]:
    """Select subjects deterministically while preferring balanced action counts.

    When every subject has the same class histogram (the common one-clip-per-action
    case), selection reduces to seeded random subject sampling.  Otherwise a
    deterministic greedy objective minimises class-count deviation after every
    addition.  Samples, and especially frames, are never selected independently.
    """

    groups = group_by_subject(samples)
    subjects = sorted(groups, key=_natural_key)
    if not subjects:
        raise ValueError("Cannot select from an empty sample collection.")
    if num_subjects is None:
        num_subjects = len(subjects)
    if num_subjects <= 0:
        raise ValueError("num_subjects must be positive.")
    if num_subjects > len(subjects):
        raise ValueError(
            f"Requested {num_subjects} subjects, but only {len(subjects)} are available."
        )

    histograms = {
        subject: np.array(
            [sum(sample.label == label for sample in groups[subject]) for label in range(3)],
            dtype=np.int64,
        )
        for subject in subjects
    }
    rng = np.random.default_rng(seed)
    random_order = rng.permutation(len(subjects))
    tie_rank = {subjects[int(index)]: rank for rank, index in enumerate(random_order)}

    first_histogram = histograms[subjects[0]]
    uniform_profiles = all(np.array_equal(histograms[item], first_histogram) for item in subjects)
    if uniform_profiles:
        selected = [subjects[int(index)] for index in random_order[:num_subjects]]
        return sorted(selected, key=_natural_key)

    chosen: list[str] = []
    remaining = set(subjects)
    totals = np.zeros(3, dtype=np.int64)
    while len(chosen) < num_subjects:
        scored: list[tuple[float, int, str]] = []
        for subject in remaining:
            candidate = totals + histograms[subject]
            mean = float(candidate.mean())
            imbalance = float(np.abs(candidate - mean).sum())
            # Prefer coverage of all documented actions before optimising small
            # residual deviations.  The seeded rank resolves exact ties.
            missing_penalty = float(np.count_nonzero(candidate == 0) * (candidate.sum() + 1))
            scored.append((missing_penalty + imbalance, tie_rank[subject], subject))
        _, _, winner = min(scored)
        chosen.append(winner)
        totals += histograms[winner]
        remaining.remove(winner)
    return sorted(chosen, key=_natural_key)


def exact_split_counts(
    num_subjects: int, ratios: Sequence[float] = (0.70, 0.15, 0.15)
) -> tuple[int, int, int]:
    """Allocate all subjects with largest remainders and deterministic ties."""

    if num_subjects < 3:
        raise ValueError("At least three subjects are required for non-empty train/val/test splits.")
    ratio_array = np.asarray(ratios, dtype=np.float64)
    if ratio_array.shape != (3,) or np.any(ratio_array <= 0):
        raise ValueError("split ratios must contain three positive values.")
    if not np.isclose(ratio_array.sum(), 1.0, atol=1e-9):
        raise ValueError(f"split ratios must sum to 1.0, got {ratio_array.sum():.12g}.")
    exact = ratio_array * num_subjects
    counts = np.floor(exact).astype(int)
    remainder = num_subjects - int(counts.sum())
    # Stable argsort ensures train, then val, then test resolve equal fractions.
    order = np.argsort(-(exact - counts), kind="stable")
    for index in order[:remainder]:
        counts[int(index)] += 1
    if np.any(counts == 0):
        # Small datasets still require all three scientific partitions. Transfer
        # from the largest partition without changing the total.
        for empty in np.flatnonzero(counts == 0):
            donor = int(np.argmax(counts))
            if counts[donor] <= 1:
                raise ValueError("Not enough subjects for non-empty splits.")
            counts[donor] -= 1
            counts[int(empty)] += 1
    assert int(counts.sum()) == num_subjects
    return tuple(int(value) for value in counts)  # type: ignore[return-value]


def split_subjects(
    subject_ids: Sequence[str],
    *,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, str]:
    """Assign each unique subject to exactly one deterministic split."""

    unique = sorted({str(value) for value in subject_ids}, key=_natural_key)
    if len(unique) != len(subject_ids):
        raise ValueError("split_subjects expects each subject identifier exactly once.")
    counts = exact_split_counts(len(unique), ratios)
    order = np.random.default_rng(seed).permutation(len(unique))
    shuffled = [unique[int(index)] for index in order]
    assignment: dict[str, str] = {}
    start = 0
    for split, count in zip(SPLIT_NAMES, counts, strict=True):
        for subject in shuffled[start : start + count]:
            assignment[subject] = split
        start += count
    assert_subject_assignment(assignment)
    return assignment


def assert_subject_assignment(assignment: Mapping[str, str]) -> None:
    """Fail loudly unless train, validation and test subjects are disjoint."""

    unknown = sorted(set(assignment.values()) - set(SPLIT_NAMES))
    if unknown:
        raise AssertionError(f"Unknown split labels: {unknown}.")
    split_sets = {
        split: {subject for subject, assigned in assignment.items() if assigned == split}
        for split in SPLIT_NAMES
    }
    train_subjects = split_sets["train"]
    val_subjects = split_sets["val"]
    test_subjects = split_sets["test"]
    assert train_subjects.isdisjoint(val_subjects), "Subject leakage between train and validation."
    assert train_subjects.isdisjoint(test_subjects), "Subject leakage between train and test."
    assert val_subjects.isdisjoint(test_subjects), "Subject leakage between validation and test."


def assert_no_subject_leakage(subject_ids: Sequence[str], splits: Sequence[str]) -> None:
    """Assert that every subject appears under one and only one split label."""

    if len(subject_ids) != len(splits):
        raise AssertionError("subject_ids and splits must have equal length.")
    observed: dict[str, str] = {}
    for subject, split in zip(subject_ids, splits, strict=True):
        if split not in SPLIT_NAMES:
            raise AssertionError(f"Unknown split {split!r} for subject {subject!r}.")
        previous = observed.setdefault(str(subject), str(split))
        if previous != split:
            raise AssertionError(
                f"Subject leakage: {subject!r} occurs in both {previous!r} and {split!r}."
            )
    assert_subject_assignment(observed)


def build_manifest_rows(
    samples: Sequence[PoseSample], selected_subjects: Sequence[str], assignment: Mapping[str, str]
) -> list[dict[str, str]]:
    """Build sorted manifest rows and assert sample/subject integrity."""

    selected = set(selected_subjects)
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for sample in samples:
        if sample.subject_id not in selected:
            continue
        if sample.subject_id not in assignment:
            raise AssertionError(f"Selected subject {sample.subject_id!r} has no split assignment.")
        if sample.sample_id in seen_ids:
            raise AssertionError(f"Duplicate sample identifier {sample.sample_id!r}.")
        seen_ids.add(sample.sample_id)
        rows.append(
            {
                "subject_id": sample.subject_id,
                "sample_id": sample.sample_id,
                "action": INDEX_TO_LABEL[sample.label],
                "split": assignment[sample.subject_id],
            }
        )
    missing = selected - {row["subject_id"] for row in rows}
    if missing:
        raise AssertionError(f"Selected subjects with no samples: {sorted(missing, key=_natural_key)}.")
    assert_no_subject_leakage(
        [row["subject_id"] for row in rows], [row["split"] for row in rows]
    )
    rows.sort(key=lambda row: (SPLIT_NAMES.index(row["split"]), _natural_key(row["subject_id"]), row["sample_id"]))
    return rows


def write_selected_subjects(
    path: str | Path, selected_subjects: Sequence[str], *, seed: int, available: int
) -> None:
    """Write the reproducible subject-selection record."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": int(seed),
        "num_available_subjects": int(available),
        "num_selected_subjects": len(selected_subjects),
        "selected_subject_ids": list(selected_subjects),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_manifest(path: str | Path, rows: Sequence[Mapping[str, str]]) -> None:
    """Write the required split manifest CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "sample_id", "action", "split"])
        writer.writeheader()
        writer.writerows(rows)


def write_split_audit(
    path: str | Path, rows: Sequence[Mapping[str, str]], *, synthetic: bool = False
) -> None:
    """Write explicit, human-readable evidence of subject disjointness."""

    subjects_by_split = {
        split: {row["subject_id"] for row in rows if row["split"] == split}
        for split in SPLIT_NAMES
    }
    subject_ids = [row["subject_id"] for row in rows]
    splits = [row["split"] for row in rows]
    assert_no_subject_leakage(subject_ids, splits)
    duplicate_ids = len({row["sample_id"] for row in rows}) != len(rows)
    if duplicate_ids:
        raise AssertionError("Duplicate sample identifiers detected while writing split audit.")
    class_counts = Counter((row["split"], row["action"]) for row in rows)
    lines = ["# Subject-disjoint split audit", ""]
    if synthetic:
        lines += [
            "> **SYNTHETIC SOFTWARE FIXTURE — not research data or research evidence.**",
            "",
        ]
    lines += [
        "All assertions below passed during manifest construction.",
        "",
        f"- Train subjects: {len(subjects_by_split['train'])}",
        f"- Validation subjects: {len(subjects_by_split['val'])}",
        f"- Test subjects: {len(subjects_by_split['test'])}",
        f"- Samples: {len(rows)}",
        "- Duplicate sample identifiers: 0",
        "- `train_subjects.isdisjoint(val_subjects)`: **True**",
        "- `train_subjects.isdisjoint(test_subjects)`: **True**",
        "- `val_subjects.isdisjoint(test_subjects)`: **True**",
        "",
        "## Sample counts by split and action",
        "",
        "| Split | d (dribble) | p (lay-up) | s (shoot) |",
        "|---|---:|---:|---:|",
    ]
    for split in SPLIT_NAMES:
        lines.append(
            f"| {split} | {class_counts[(split, 'd')]} | {class_counts[(split, 'p')]} | "
            f"{class_counts[(split, 's')]} |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
