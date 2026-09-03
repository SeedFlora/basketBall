#!/usr/bin/env python
"""Archived Henu/MMAAction-style fixture generator (disabled by default).

This generator predates the TrackID3x3 reconstruction pivot.  Its COCO17
poses and d/p/s labels are fabricated solely for legacy software tests and are
not compatible with the public TrackID3x3 workflow.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

COCO17_BASE = np.array(
    [
        [0.50, 0.12], [0.47, 0.10], [0.53, 0.10], [0.43, 0.12], [0.57, 0.12],
        [0.42, 0.28], [0.58, 0.28], [0.34, 0.43], [0.66, 0.43], [0.30, 0.58],
        [0.70, 0.58], [0.45, 0.55], [0.55, 0.55], [0.43, 0.73], [0.57, 0.73],
        [0.42, 0.93], [0.58, 0.93],
    ],
    dtype=np.float32,
)


def make_clip(subject: int, label: int, frames: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Create fabricated class-coded motion for an archived software fixture."""
    phase = np.linspace(0.0, 2.0 * np.pi, frames, dtype=np.float32)
    pose = np.repeat(COCO17_BASE[None, :, :], frames, axis=0)
    pose += rng.normal(0.0, 0.004, size=pose.shape).astype(np.float32)
    pose[..., 0] += (subject % 7 - 3) * 0.004
    if label == 0:  # dribble: alternating wrist vertical movement
        pose[:, 10, 1] += 0.16 * np.sin(2.5 * phase)
        pose[:, 9, 1] -= 0.05 * np.sin(2.5 * phase)
    elif label == 1:  # lay-up: rising body, unilateral arm, knee lift
        progress = np.linspace(0.0, 1.0, frames, dtype=np.float32)
        pose[..., 1] -= 0.12 * progress[:, None]
        pose[:, 8, 1] -= 0.18 * progress
        pose[:, 10, 1] -= 0.30 * progress
        pose[:, 14, 1] -= 0.18 * np.sin(np.pi * progress)
    else:  # shoot: both wrists rise symmetrically and settle
        lift = np.sin(np.linspace(0.0, np.pi, frames, dtype=np.float32))
        pose[:, [7, 8], 1] -= 0.12 * lift[:, None]
        pose[:, [9, 10], 1] -= 0.30 * lift[:, None]
        pose[:, [9, 10], 0] += np.stack((0.05 * lift, -0.05 * lift), axis=1)
    pose = np.clip(pose, 0.0, 1.0)
    score = np.clip(rng.normal(0.94, 0.03, size=(frames, 17)), 0.0, 1.0).astype(np.float32)
    return pose[None, ...], score[None, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Use the TrackID3x3 inspect/build_subset/preprocess commands for the public "
            "workflow. The opt-in flag exists only to preserve archived software tests."
        ),
    )
    parser.add_argument(
        "--allow-legacy-henu-fixture",
        action="store_true",
        help=(
            "Explicitly acknowledge this creates fabricated, TrackID-incompatible "
            "legacy test data and no research evidence."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/legacy_henu_synthetic_fixture"),
    )
    parser.add_argument("--num-subjects", type=int, default=18)
    parser.add_argument("--clips-per-action", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not args.allow_legacy_henu_fixture:
        parser.error(
            "legacy Henu/MMAAction fixture generation is disabled for the public "
            "TrackID3x3 workflow; pass --allow-legacy-henu-fixture only for archived "
            "software validation"
        )
    if args.num_subjects < 6:
        parser.error("Use at least six subjects so every legacy subject split is meaningful.")
    if args.clips_per_action < 1:
        parser.error("--clips-per-action must be positive.")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        "WARNING: generating fabricated legacy Henu/MMAAction test data; "
        "not TrackID3x3 and not research evidence.",
        file=sys.stderr,
    )
    rng = np.random.default_rng(args.seed)
    annotations: list[dict[str, object]] = []
    action_codes = ("d", "p", "s")
    for subject in range(1, args.num_subjects + 1):
        for label, code in enumerate(action_codes):
            for clip_index in range(1, args.clips_per_action + 1):
                frames = int(rng.integers(22, 49))
                keypoint, score = make_clip(subject, label, frames, rng)
                annotations.append(
                    {
                        "frame_dir": f"{subject}_{code}_{clip_index}",
                        "label": label,
                        "total_frames": frames,
                        "img_shape": (1080, 1920),
                        "original_shape": (1080, 1920),
                        "keypoint": keypoint,
                        "keypoint_score": score,
                        "synthetic_fixture": True,
                    }
                )
    rng.shuffle(annotations)
    cut = round(len(annotations) * 0.75)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, records in (("train.pkl", annotations[:cut]), ("val.pkl", annotations[cut:])):
        payload = {
            "split": {name.removesuffix(".pkl"): [item["frame_dir"] for item in records]},
            "annotations": records,
            "fixture_metadata": {
                "synthetic": True,
                "research_results": False,
                "legacy_henu_fixture": True,
                "schema_family": "legacy_mmaction_coco17_action_classification",
                "trackid3x3_compatible": False,
                "disabled_by_default": True,
                "purpose": "archived software validation only; not research evidence",
                "action_labels": "fabricated d/p/s class codes",
                "seed": args.seed,
            },
        }
        with (output / name).open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (output / "LEGACY_HENU_FIXTURE_ONLY.txt").write_text(
        "LEGACY HENU/MMACTION SOFTWARE FIXTURE ONLY.\n"
        "TrackID3x3 compatible: false.\n"
        "The poses and d/p/s labels are fabricated; metrics are not research results.\n"
        "Use scripts/inspect_dataset.py, build_subset.py, and preprocess.py for TrackID3x3.\n",
        encoding="utf-8",
    )
    print(f"Legacy synthetic fixture only: {len(annotations)} clips, {args.num_subjects} subjects")
    print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
