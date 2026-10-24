#!/usr/bin/env python
"""Aggregate the multi-seed replicate evaluations into seed-mean tables.

Reads the evaluations written by ``run_multiseed.py`` under ``outputs/multiseed/evaluations`` and
writes ``outputs/tables/reconstruction/multiseed_*.csv`` with a Markdown twin for each.

Conventions, applied identically to every table:

- A per-seed score is the equal-domain macro mean over the test domains a slice covers
  (Indoor, Outdoor, Drone for T=1; Indoor, Outdoor for T=16), or a single domain for transfer cells.
- Seed statistics are the mean and the sample standard deviation (ddof=1) over seeds.
- Relative differences are computed from the seed means shown in the tables, so every quoted
  percentage can be re-derived from a printed number.
- A transfer penalty compares a cross-domain cell with the target domain's own within-domain
  model on the identical test set and evaluation masks.
- Paired tests pair two conditions by seed, because both share that seed's evaluation masks.

The common-denominator re-scoring divides every error by the diagonal of the ground-truth
pose-tight box (min/max over all originally valid joints), the definition Drone already uses as
its metric box. It is metric-only: the box never enters a model input.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("Indoor", "Outdoor", "Drone")
THRESHOLDS = (0.05, 0.10, 0.20)
METRICS = ("nmje", "pck_at_0_05", "pck_at_0_10", "pck_at_0_20")
JOINTS = (
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "center",
    "left_ankle",
    "right_ankle",
)

REPRESENTATIONS = (
    ("raw", "core_mlp_raw_all_mask20_seed{seed}_eval"),
    ("root_centered", "core_mlp_root_all_mask20_seed{seed}_eval"),
    ("bbox_normalized", "core_mlp_bbox_all_mask20_seed{seed}_eval"),
    ("scale_normalized", "core_mlp_scale_all_mask20_seed{seed}_eval"),
    ("normalized_angles", "core_mlp_angles_all_mask20_seed{seed}_eval"),
    ("geometric baseline", "core_geometric_all_seed{seed}"),
)
STATIC_MODELS = (
    ("geometric baseline", "core_geometric_all_seed{seed}"),
    ("mlp", "core_mlp_bbox_all_mask20_seed{seed}_eval"),
    ("graph", "core_graph_bbox_all_mask20_seed{seed}_eval"),
)
TEMPORAL_MODELS = (
    ("geometric baseline", "core_geometric_all_t16_seed{seed}"),
    ("temporal interpolation", "core_temporal_all_t16_seed{seed}"),
    ("mlp", "core_mlp_bbox_all_t16_mask20_seed{seed}_eval"),
    ("graph", "core_graph_bbox_all_t16_mask20_seed{seed}_eval"),
    ("tcn", "core_tcn_bbox_all_t16_mask20_seed{seed}_eval"),
)
PRIMARY = "core_mlp_angles_all_mask20_seed{seed}_eval"
# Transfer families; {rep} is the input representation (bbox or scale).
DOMAIN_RUN = "domain_mlp_{rep}_{domain}_mask20_seed{seed}_eval"
SIZEMATCH_RUN = "sizematch_mlp_{rep}_{domain}_data{percent}_mask20_seed{seed}_eval"
ROTATION_RUN = "rotation_fold{fold}_mlp_{rep}_{domain}_mask20_seed{seed}_eval"
TRANSFER_REPRESENTATIONS = {"bbox": "bbox_normalized", "scale": "scale_normalized"}
DATAEFF_RUN = "dataeff_mlp_angles_all_data{percent}_mask20_seed{seed}_eval"
SHIFT = (
    ("raw", "shift_mlp_raw_indoor_mask20_seed{seed}_eval"),
    ("bbox_normalized", "domain_mlp_bbox_indoor_mask20_seed{seed}_eval"),
    ("scale_normalized", "shift_mlp_scale_indoor_mask20_seed{seed}_eval"),
)


class MultiSeedError(RuntimeError):
    """Raised when a replicate artifact is missing or disagrees with its own metrics."""


class Store:
    """Read-only access to the replicate evaluations under one metric denominator.

    ``native`` reads each stored ``metrics.json`` (source box: MOT for Indoor/Outdoor, pose-tight
    for Drone). ``common`` re-scores the saved predictions with the ground-truth pose-tight box
    for every domain, after checking that the native re-computation matches the stored metrics.
    """

    def __init__(self, root: Path, seeds: Sequence[int], denominator: str = "native") -> None:
        self.root = root
        self.seeds = tuple(seeds)
        self.denominator = denominator
        self._scores: dict[tuple[str, int], dict[str, dict[str, Any]] | None] = {}

    def stored(self, run: str, ratio: int, domain: str) -> dict[str, Any] | None:
        path = self.root / run / f"mask_{ratio:02d}" / domain.lower() / "metrics.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("metrics", payload)

    def scores(self, run: str, ratio: int) -> dict[str, dict[str, Any]] | None:
        key = (run, ratio)
        if key not in self._scores:
            self._scores[key] = _score_predictions(self, run, ratio)
        return self._scores[key]

    def metrics(self, run: str, ratio: int, domain: str) -> dict[str, Any] | None:
        if self.denominator == "native":
            return self.stored(run, ratio, domain)
        scored = self.scores(run, ratio)
        return None if scored is None or domain not in scored else scored[domain]["common"]

    def macro(self, run: str, ratio: int, domains: Sequence[str]) -> dict[str, float]:
        """Equal-domain macro mean; NaN unless every requested domain is present."""

        per_domain = [self.metrics(run, ratio, domain) for domain in domains]
        if any(entry is None for entry in per_domain):
            return {metric: float("nan") for metric in METRICS}
        return {
            metric: float(np.mean([entry[metric] for entry in per_domain]))  # type: ignore[index]
            for metric in METRICS
        }

    def per_seed(
        self, template: str, ratio: int, domains: Sequence[str], **fields: Any
    ) -> dict[int, dict[str, float]]:
        return {
            seed: self.macro(template.format(seed=seed, **fields), ratio, domains)
            for seed in self.seeds
        }


def _metric_block(error: np.ndarray, joints: np.ndarray, poses: np.ndarray, sample_ids: np.ndarray) -> dict[str, Any]:
    """NMJE, PCK, per-joint, and per-pose values for one domain's scored normalized errors."""

    block: dict[str, Any] = {"nmje": float(error.mean()), "n_scored_joints": int(error.size)}
    for threshold in THRESHOLDS:
        block[f"pck_at_{threshold:.2f}".replace(".", "_")] = float((error <= threshold).mean())
    block["per_joint"] = {
        name: {"nmje": float(error[joints == index].mean()), "count": int((joints == index).sum())}
        for index, name in enumerate(JOINTS)
        if (joints == index).any()
    }
    frame = pd.DataFrame({"sample_id": sample_ids[poses], "error": error})
    block["per_pose"] = frame.groupby("sample_id", sort=False)["error"].mean()
    return block


def _score_predictions(store: Store, run: str, ratio: int) -> dict[str, dict[str, Any]] | None:
    """Score one evaluation's saved predictions under the native and the common denominator."""

    path = store.root / run / f"mask_{ratio:02d}" / "predictions.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as payload:
        prediction = payload["prediction"]
        target = payload["target"]
        valid = payload["original_valid_mask"].astype(bool)
        frames = payload["sequence_mask"].astype(bool)
        scored = payload["synthetic_mask"].astype(bool) & valid & frames[..., None]
        boxes = payload["bbox_xywh"]
        domains = payload["domains"].astype(str)
        sample_ids = payload["sample_ids"].astype(str)
    # Exact replica of src/data/trackid3x3.py:_pose_tight_bbox (float32 min/max over valid joints).
    present = valid & frames[..., None]
    lower = np.where(present[..., None], target, np.float32(np.inf)).min(axis=2)
    upper = np.where(present[..., None], target, np.float32(-np.inf)).max(axis=2)
    extent = (upper - lower).astype(np.float32)
    denominators = {
        "native": np.hypot(boxes[..., 2].astype(np.float64), boxes[..., 3].astype(np.float64)),
        "common": np.hypot(extent[..., 0].astype(np.float64), extent[..., 1].astype(np.float64)),
    }
    error = np.linalg.norm(prediction.astype(np.float64) - target.astype(np.float64), axis=-1)
    pose_index, _, joint_index = np.nonzero(scored)
    result: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        keep = domains[pose_index] == domain
        if not keep.any():
            continue
        poses, joints = pose_index[keep], joint_index[keep]
        frames_of = np.nonzero(scored)[1][keep]
        entry: dict[str, Any] = {}
        for name, diagonal in denominators.items():
            normalized = error[poses, frames_of, joints] / diagonal[poses, frames_of]
            entry[name] = _metric_block(normalized, joints, poses, sample_ids)
        stored = store.stored(run, ratio, domain)
        if stored is not None and abs(entry["native"]["nmje"] - float(stored["nmje"])) > 1e-6:
            raise MultiSeedError(
                f"{run} {domain}: recomputed native NMJE {entry['native']['nmje']:.8f} "
                f"disagrees with stored {stored['nmje']:.8f}"
            )
        entry["median_native_to_common_diagonal"] = float(
            np.median((denominators["native"] / denominators["common"])[domains == domain][frames[domains == domain]])
        )
        # Pixel size of each pose (first frame): the within-domain player-scale covariate.
        in_domain = domains == domain
        entry["pose_diagonal"] = pd.Series(denominators["common"][in_domain, 0], index=sample_ids[in_domain])
        result[domain] = entry
    return result


def _summary(values: Iterable[float]) -> tuple[float, float, int]:
    observed = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if observed.size == 0:
        return float("nan"), float("nan"), 0
    spread = float(observed.std(ddof=1)) if observed.size > 1 else float("nan")
    return float(observed.mean()), spread, int(observed.size)


def _row(label: dict[str, Any], scores: dict[int, dict[str, float]]) -> dict[str, Any]:
    row = dict(label)
    for metric in METRICS:
        mean, spread, count = _summary(score[metric] for score in scores.values())
        row[f"{metric}_mean"] = mean
        row[f"{metric}_sd"] = spread
        row["seeds"] = count
    return row


def _relative(value: float, reference: float) -> float:
    return 100.0 * (value - reference) / reference if np.isfinite(reference) and reference else float("nan")


def _paired(
    comparison: str,
    first: str,
    second: str,
    a: dict[int, dict[str, float]],
    b: dict[int, dict[str, float]],
    metric: str = "nmje",
) -> dict[str, Any]:
    """Pair two conditions by seed; negative differences favour ``first`` for NMJE."""

    seeds = [seed for seed in a if seed in b and np.isfinite(a[seed][metric]) and np.isfinite(b[seed][metric])]
    left = np.asarray([a[seed][metric] for seed in seeds])
    right = np.asarray([b[seed][metric] for seed in seeds])
    difference = left - right
    result: dict[str, Any] = {
        "comparison": comparison,
        "first": first,
        "second": second,
        "metric": metric,
        "seeds": len(seeds),
        "first_mean": float(left.mean()) if seeds else float("nan"),
        "second_mean": float(right.mean()) if seeds else float("nan"),
        "mean_difference": float(difference.mean()) if seeds else float("nan"),
        "sd_difference": float(difference.std(ddof=1)) if len(seeds) > 1 else float("nan"),
        "first_lower_in_seeds": int((difference < 0).sum()),
        "t_statistic": float("nan"),
        "p_value": float("nan"),
        "ci95_low_pct": float("nan"),
        "ci95_high_pct": float("nan"),
    }
    result["relative_difference_pct"] = _relative(result["first_mean"], result["second_mean"])
    if len(seeds) > 1 and np.any(difference != difference[0]):
        test = stats.ttest_rel(left, right)
        result["t_statistic"] = float(test.statistic)
        result["p_value"] = float(test.pvalue)
        # CI of the mean paired difference, expressed relative to the second condition's mean.
        half = float(stats.t.ppf(0.975, len(seeds) - 1)) * result["sd_difference"] / np.sqrt(len(seeds))
        result["ci95_low_pct"] = _relative(result["second_mean"] + result["mean_difference"] - half, result["second_mean"])
        result["ci95_high_pct"] = _relative(result["second_mean"] + result["mean_difference"] + half, result["second_mean"])
    return result


def _rescore(store: Store, run: str, ratio: int = 20) -> dict[str, dict[str, float]] | None:
    """Flat native and common-denominator scores for every test domain of one evaluation."""

    scored = store.scores(run, ratio)
    if scored is None:
        return None
    result: dict[str, dict[str, float]] = {}
    for domain, entry in scored.items():
        flat: dict[str, float] = {
            "scored_joints": float(entry["native"]["n_scored_joints"]),
            "median_native_to_common_diagonal": entry["median_native_to_common_diagonal"],
        }
        for metric in METRICS:
            flat[metric] = entry["native"][metric]
            flat[f"common_{metric}"] = entry["common"][metric]
        result[domain] = flat
    return result


def representation_table(store: Store) -> tuple[pd.DataFrame, dict[str, dict[int, dict[str, float]]]]:
    scores = {name: store.per_seed(template, 20, DOMAINS) for name, template in REPRESENTATIONS}
    rows = [_row({"representation": name}, scores[name]) for name, _ in REPRESENTATIONS]
    return pd.DataFrame(rows), scores


def model_tables(store: Store) -> tuple[pd.DataFrame, dict[str, dict[int, dict[str, float]]]]:
    scores: dict[str, dict[int, dict[str, float]]] = {}
    rows = []
    for name, template in STATIC_MODELS:
        key = f"T1/{name}"
        scores[key] = store.per_seed(template, 20, DOMAINS)
        rows.append(_row({"sequence_length": 1, "model": name, "domains": "Indoor+Outdoor+Drone"}, scores[key]))
    for name, template in TEMPORAL_MODELS:
        key = f"T16/{name}"
        scores[key] = store.per_seed(template, 20, DOMAINS[:2])
        rows.append(_row({"sequence_length": 16, "model": name, "domains": "Indoor+Outdoor"}, scores[key]))
    return pd.DataFrame(rows), scores


def missingness_table(store: Store) -> tuple[pd.DataFrame, dict[str, dict[int, dict[str, float]]]]:
    scores: dict[str, dict[int, dict[str, float]]] = {}
    rows = []
    for label, template in (
        ("normalized_angles MLP", PRIMARY),
        ("bbox_normalized MLP", "core_mlp_bbox_all_mask20_seed{seed}_eval"),
        ("geometric baseline", "core_geometric_all_seed{seed}"),
    ):
        for ratio in (10, 20, 30, 40):
            key = f"{label}/{ratio}"
            scores[key] = store.per_seed(template, ratio, DOMAINS)
            rows.append(_row({"model": label, "mask_percent": ratio}, scores[key]))
    return pd.DataFrame(rows), scores


def data_efficiency_table(store: Store) -> tuple[pd.DataFrame, dict[str, dict[int, dict[str, float]]]]:
    # Every point uses batch 64; the batch-128 primary run is not mixed into this curve.
    scores = {
        str(percent): store.per_seed(DATAEFF_RUN, 20, DOMAINS, percent=percent) for percent in (25, 50, 100)
    }
    rows = [_row({"train_percent": int(key)}, value) for key, value in scores.items()]
    return pd.DataFrame(rows), scores


def transfer_table(
    store: Store,
    run_template: str,
    label: str,
    *,
    replicates: Sequence[int] | None = None,
    key: str = "seed",
    **fields: Any,
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[int, dict[str, float]]]]:
    """Native and common-denominator transfer cells for one family of source models.

    Replicates are seeds by default; the group-rotation control passes folds with ``key="fold"``.
    """

    cells: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    for train in DOMAINS:
        for replicate in store.seeds if replicates is None else replicates:
            run = run_template.format(domain=train.lower(), **{key: replicate}, **fields)
            rescored = _rescore(store, run)
            for test in DOMAINS:
                entry = None if rescored is None else rescored.get(test)
                cells.setdefault((train, test), {})[replicate] = (
                    {name: float("nan") for name in ("nmje", "common_nmje", "pck_at_0_10", "common_pck_at_0_10")}
                    if entry is None
                    else entry
                )
    return cells_frame(cells, label), cells


def cells_frame(cells: dict[tuple[str, str], dict[int, dict[str, float]]], label: str) -> pd.DataFrame:
    rows = []
    for (train, test), per_replicate in cells.items():
        row: dict[str, Any] = {"source_family": label, "train_domain": train, "test_domain": test}
        for metric in ("nmje", "pck_at_0_10", "common_nmje", "common_pck_at_0_10"):
            mean, spread, count = _summary(entry.get(metric, float("nan")) for entry in per_replicate.values())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = spread
            row["replicates"] = count
        rows.append(row)
    return pd.DataFrame(rows)


def per_replicate_penalties(
    cells: dict[tuple[str, str], dict[int, dict[str, float]]], metric: str = "nmje"
) -> pd.DataFrame:
    """Spread of the transfer penalty when it is computed separately in every replicate."""

    rows = []
    for (train, test), per_replicate in cells.items():
        if train == test:
            continue
        within = cells[(test, test)]
        values = [
            _relative(entry[metric], within[replicate][metric])
            for replicate, entry in per_replicate.items()
            if replicate in within
        ]
        observed = np.asarray([value for value in values if np.isfinite(value)])
        rows.append(
            {
                "train_domain": train,
                "test_domain": test,
                "metric": metric,
                "replicates": int(observed.size),
                "penalty_min_pct": float(observed.min()) if observed.size else float("nan"),
                "penalty_max_pct": float(observed.max()) if observed.size else float("nan"),
                "penalty_sd_pct": float(observed.std(ddof=1)) if observed.size > 1 else float("nan"),
                "replicates_with_positive_penalty": int((observed > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def add_penalties(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Penalty relative to the target domain's within-domain model on the same test set."""

    within = {
        row.test_domain: row
        for row in reference.itertuples()
        if row.train_domain == row.test_domain
    }
    frame = frame.copy()
    for prefix in ("", "common_"):
        frame[f"{prefix}penalty_vs_target_within_pct"] = [
            _relative(getattr(row, f"{prefix}nmje_mean"), getattr(within[row.test_domain], f"{prefix}nmje_mean"))
            if row.test_domain in within
            else float("nan")
            for row in frame.itertuples()
        ]
    return frame


def per_joint_table(store: Store) -> pd.DataFrame:
    rows = []
    for joint in JOINTS:
        per_seed = []
        for seed in store.seeds:
            values = []
            for domain in DOMAINS:
                entry = store.metrics(PRIMARY.format(seed=seed), 20, domain)
                joint_entry = None if entry is None else entry["per_joint"].get(joint)
                values.append(float("nan") if joint_entry is None else float(joint_entry["nmje"]))
            per_seed.append(float(np.mean(values)))
        mean, spread, count = _summary(per_seed)
        rows.append({"joint": joint, "macro_nmje_mean": mean, "macro_nmje_sd": spread, "seeds": count})
    return pd.DataFrame(rows)


def strata_table(store: Store) -> pd.DataFrame:
    """Pose-level strata of the primary slice, mirroring generate_report.py, then seed statistics."""

    collected: dict[tuple[str, str], list[tuple[float, int]]] = {}
    for seed in store.seeds:
        path = store.root / PRIMARY.format(seed=seed) / "mask_20" / "sequence_errors.csv"
        if not path.is_file():
            continue
        poses = pd.read_csv(path)
        if store.denominator == "common":
            scored = store.scores(PRIMARY.format(seed=seed), 20) or {}
            common = pd.concat([entry["common"]["per_pose"] for entry in scored.values()])
            poses["nmje"] = poses["sample_id"].map(common)
        poses["nmje"] = pd.to_numeric(poses["nmje"], errors="coerce")
        poses = poses.dropna(subset=["nmje"])
        groups: list[tuple[str, Any]] = [
            ("masked distal target", np.where(pd.to_numeric(poses["masked_distal_joints"]).gt(0), "yes", "no")),
            ("player scale bin", poses["player_scale_bin"].astype(str)),
            (
                "pose-spread quartile",
                pd.qcut(pd.to_numeric(poses["pose_spread_bbox_fraction"]), q=4, labels=["Q1", "Q2", "Q3", "Q4"]),
            ),
        ]
        overlap = pd.to_numeric(poses["same_frame_max_bbox_iou"], errors="coerce")
        groups.append(
            (
                "same-frame overlap",
                pd.cut(overlap, bins=[-np.inf, 0.05, 0.20, np.inf], labels=["low", "moderate", "high"]).astype(str),
            )
        )
        for factor, labels in groups:
            frame = pd.DataFrame({"nmje": poses["nmje"].to_numpy(), "level": np.asarray(labels).astype(str)})
            for level, part in frame.groupby("level"):
                if level == "nan":
                    continue
                collected.setdefault((factor, level), []).append((float(part["nmje"].mean()), len(part)))
    rows = []
    for (factor, level), values in sorted(collected.items()):
        mean, spread, count = _summary(value for value, _ in values)
        rows.append(
            {
                "factor": factor,
                "level": level,
                "pose_mean_nmje_mean": mean,
                "pose_mean_nmje_sd": spread,
                "poses_min": min(size for _, size in values),
                "poses_max": max(size for _, size in values),
                "seeds": count,
            }
        )
    return pd.DataFrame(rows)


def _joint_extent(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Width and height of the box spanned by the masked-in joints of each pose ([N, 10, 2])."""

    low = np.where(mask[..., None], points, np.inf).min(axis=1)
    high = np.where(mask[..., None], points, -np.inf).max(axis=1)
    return (high - low).astype(np.float64)


def denominator_table(store: Store) -> pd.DataFrame:
    """How the source box, the common box, and the bbox_normalized input box relate per domain.

    Read from the Indoor-trained bbox_normalized evaluations, which cover every test domain; the
    observed span is the extent of the observed joints as a fraction of the input box.
    """

    collected: dict[str, list[tuple[float, float, float]]] = {}
    for seed in store.seeds:
        path = store.root / DOMAIN_RUN.format(rep="bbox", domain="indoor", seed=seed) / "mask_20" / "predictions.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as payload:
            target = payload["target"][:, 0]
            valid = payload["original_valid_mask"][:, 0].astype(bool)
            observed = payload["observed_mask"][:, 0].astype(bool)
            source = payload["bbox_xywh"][:, 0].astype(np.float64)
            transform = payload["transform_bbox_xywh"][:, 0].astype(np.float64)
            domains = payload["domains"].astype(str)
        tight = _joint_extent(target, valid)
        seen = _joint_extent(target, observed)
        ratio = np.hypot(source[:, 2], source[:, 3]) / np.hypot(tight[:, 0], tight[:, 1])
        for domain in DOMAINS:
            rows = domains == domain
            collected.setdefault(domain, []).append(
                (
                    float(np.median(ratio[rows])),
                    float(np.median(seen[rows, 0] / transform[rows, 2])),
                    float(np.median(seen[rows, 1] / transform[rows, 3])),
                )
            )
    rows = []
    for domain, values in collected.items():
        rows.append(
            {
                "test_domain": domain,
                "median_source_to_pose_extent_diagonal": float(np.mean([value[0] for value in values])),
                "median_observed_span_x_of_input_box": float(np.mean([value[1] for value in values])),
                "median_observed_span_y_of_input_box": float(np.mean([value[2] for value in values])),
                "seeds": len(values),
            }
        )
    return pd.DataFrame(rows)


def split_footage_table(processed: Path) -> pd.DataFrame:
    """How many test poses share a source frame with a training pose of the same domain."""

    with np.load(processed, allow_pickle=False) as payload:
        domains = payload["domain"].astype(str)
        splits = payload["split"].astype(str)
        frames = np.char.add(np.char.add(payload["source_file"][:, 0].astype(str), "#"), payload["frame_index"][:, 0].astype(str))
    rows = []
    for domain in DOMAINS:
        in_domain = domains == domain
        seen = set(frames[in_domain & (splits == "train")].tolist())
        test = frames[in_domain & (splits == "test")]
        rows.append(
            {
                "test_domain": domain,
                "test_poses": int(test.size),
                "test_poses_sharing_a_training_frame": int(sum(frame in seen for frame in test.tolist())),
            }
        )
    return pd.DataFrame(rows)


def training_budget_table(logs: Path, seeds: Sequence[int]) -> pd.DataFrame:
    """Where in the 30-epoch budget the selected checkpoint falls, over the replicate training runs."""

    prefixes = ("core_", "dataeff_", "domain_", "shift_", "sizematch_", "rotation_")
    suffixes = tuple(f"_seed{seed}" for seed in seeds)
    rows = []
    for path in sorted(logs.glob("*/metrics.json")):
        name = path.parent.name
        if not name.startswith(prefixes) or not name.endswith(suffixes):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run": name,
                "best_epoch": int(payload["best_epoch"]),
                "epochs_completed": int(payload["epochs_completed"]),
                "early_stopped": bool(payload.get("early_stopped", False)),
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "training_runs": len(frame),
        "best_epoch_at_least_28": int((frame.best_epoch >= 28).sum()),
        "best_epoch_30": int((frame.best_epoch == 30).sum()),
        "early_stopped": int(frame.early_stopped.sum()),
        "epoch_cap": 30,
    }
    return pd.DataFrame([summary])


def representation_by_domain_table(store: Store) -> pd.DataFrame:
    """Per-test-domain seed statistics behind the macro representation table."""

    rows = []
    for name, template in REPRESENTATIONS:
        for domain in DOMAINS:
            rows.append(_row({"representation": name, "test_domain": domain}, store.per_seed(template, 20, (domain,))))
    return pd.DataFrame(rows)


def within_domain_strata_table(store: Store) -> pd.DataFrame:
    """Strata formed inside each test domain and then macro-averaged, so domain cannot confound them.

    Player scale uses within-domain tertiles of the pose-extent diagonal. Same-frame overlap exists
    only where peer MOT boxes do, so it is reported for Indoor alone.
    """

    macro_factors = ("masked distal target", "player scale (within-domain tertile)")
    collected: dict[tuple[str, str], list[tuple[float, int]]] = {}
    mode = "common" if store.denominator == "common" else "native"
    for seed in store.seeds:
        run = PRIMARY.format(seed=seed)
        scored = store.scores(run, 20)
        path = store.root / run / "mask_20" / "sequence_errors.csv"
        if scored is None or not path.is_file():
            continue
        covariates = pd.read_csv(path).set_index("sample_id")[["masked_distal_joints", "same_frame_max_bbox_iou"]]
        cells: dict[tuple[str, str, str], tuple[float, int]] = {}
        for domain, entry in scored.items():
            poses = pd.DataFrame({"nmje": entry[mode]["per_pose"]})
            poses["diagonal"] = entry["pose_diagonal"].reindex(poses.index)
            poses = poses.join(covariates)
            levels: dict[str, Any] = {
                macro_factors[0]: np.where(pd.to_numeric(poses["masked_distal_joints"]).gt(0), "yes", "no"),
                macro_factors[1]: pd.qcut(poses["diagonal"], q=3, labels=["small", "medium", "large"]).astype(str),
            }
            if domain == "Indoor":
                overlap = pd.to_numeric(poses["same_frame_max_bbox_iou"], errors="coerce")
                levels["same-frame overlap (Indoor only)"] = pd.cut(
                    overlap, bins=[-np.inf, 0.05, 0.20, np.inf], labels=["low", "moderate", "high"]
                ).astype(str)
            for factor, labels in levels.items():
                frame = pd.DataFrame({"nmje": poses["nmje"].to_numpy(), "level": np.asarray(labels).astype(str)})
                for level, part in frame.groupby("level"):
                    if level != "nan":
                        cells[(factor, level, domain)] = (float(part["nmje"].mean()), len(part))
        for factor, level in {(factor, level) for factor, level, _ in cells}:
            domains = DOMAINS if factor in macro_factors else ("Indoor",)
            parts = [cells.get((factor, level, domain)) for domain in domains]
            if all(part is not None for part in parts):
                collected.setdefault((factor, level), []).append(
                    (float(np.mean([part[0] for part in parts])), sum(part[1] for part in parts))  # type: ignore[index]
                )
    rows = []
    for (factor, level), values in sorted(collected.items()):
        mean, spread, count = _summary(value for value, _ in values)
        rows.append(
            {
                "factor": factor,
                "level": level,
                "aggregation": "macro over within-domain strata" if factor in macro_factors else "Indoor only",
                "pose_mean_nmje_mean": mean,
                "pose_mean_nmje_sd": spread,
                "poses_min": min(size for _, size in values),
                "poses_max": max(size for _, size in values),
                "seeds": count,
            }
        )
    return pd.DataFrame(rows)


def runs_table(store: Store, templates: Iterable[str], ratios: Sequence[int] = (10, 20, 30, 40)) -> pd.DataFrame:
    """Long per-seed evidence: one row per run, ratio, and test domain."""

    rows = []
    for template in templates:
        for seed in store.seeds:
            run = template.format(seed=seed)
            for ratio in ratios:
                for domain in DOMAINS:
                    entry = store.metrics(run, ratio, domain)
                    if entry is None:
                        continue
                    rows.append(
                        {
                            "run": run,
                            "seed": seed,
                            "mask_percent": ratio,
                            "test_domain": domain,
                            "scored_joints": int(entry["n_scored_joints"]),
                            **{metric: float(entry[metric]) for metric in METRICS},
                        }
                    )
    return pd.DataFrame(rows)


def _markdown(frame: pd.DataFrame) -> str:
    def cell(value: Any) -> str:
        if isinstance(value, float):
            return "" if not np.isfinite(value) else f"{value:.4f}"
        return str(value)

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body]) + "\n"


def _write(frame: pd.DataFrame, directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    frame.to_csv(path, index=False)
    (directory / f"{name}.md").write_text(_markdown(frame), encoding="utf-8")
    return path


def aggregate(args: argparse.Namespace) -> list[Path]:
    store = Store(PROJECT_ROOT / args.evaluations, args.seeds, args.denominator)
    tables = PROJECT_ROOT / args.tables_dir
    suffix = "" if args.denominator == "native" else f"_{args.denominator}"

    def write(frame: pd.DataFrame, name: str) -> Path:
        return _write(frame, tables, name + suffix)

    written: list[Path] = []
    tests: list[dict[str, Any]] = []

    representation, rep = representation_table(store)
    written.append(write(representation, "multiseed_representation"))
    for first, second in (
        ("raw", "bbox_normalized"),
        ("root_centered", "geometric baseline"),
        ("scale_normalized", "root_centered"),
        ("scale_normalized", "bbox_normalized"),
        ("normalized_angles", "scale_normalized"),
        ("normalized_angles", "geometric baseline"),
        ("scale_normalized", "geometric baseline"),
    ):
        tests.append(_paired("representation, T=1 macro", first, second, rep[first], rep[second]))

    models, mod = model_tables(store)
    written.append(write(models, "multiseed_models"))
    for first, second in (
        ("T1/mlp", "T1/graph"),
        ("T1/graph", "T1/geometric baseline"),
        ("T16/mlp", "T16/graph"),
        ("T16/tcn", "T16/mlp"),
        ("T16/graph", "T16/geometric baseline"),
        ("T16/mlp", "T16/temporal interpolation"),
        ("T16/graph", "T16/temporal interpolation"),
        ("T16/tcn", "T16/temporal interpolation"),
    ):
        tests.append(_paired("model family", first, second, mod[first], mod[second]))

    missing, mis = missingness_table(store)
    written.append(write(missing, "multiseed_missingness"))
    for ratio in (10, 30, 40):
        key, reference = f"normalized_angles MLP/{ratio}", "normalized_angles MLP/20"
        tests.append(_paired("masking rate", key, reference, mis[key], mis[reference]))

    efficiency, eff = data_efficiency_table(store)
    written.append(write(efficiency, "multiseed_data_efficiency"))
    for first, second in (("25", "100"), ("50", "100"), ("25", "50")):
        tests.append(_paired("training groups (percent)", first, second, eff[first], eff[second]))

    def transfer_tests(family: str, cells: dict[tuple[str, str], dict[int, dict[str, float]]]) -> None:
        for (train, test), per_replicate in cells.items():
            if train == test:
                continue
            for prefix, denominator in (("", "native"), ("common_", "common")):
                tests.append(
                    _paired(
                        f"transfer penalty, {family}, {denominator} denominator",
                        f"{train}->{test}",
                        f"{test}->{test}",
                        per_replicate,
                        cells[(test, test)],
                        metric=f"{prefix}nmje",
                    )
                )

    transfer_frames: list[pd.DataFrame] = []
    spread_frames: list[pd.DataFrame] = []
    for rep, representation in TRANSFER_REPRESENTATIONS.items():
        families: list[tuple[str, dict[tuple[str, str], dict[int, dict[str, float]]]]] = []
        if (store.root / DOMAIN_RUN.format(rep=rep, domain="indoor", seed=store.seeds[0])).is_dir():
            _, full = transfer_table(store, DOMAIN_RUN, "five seeds", rep=rep)
            families.append(("five seeds", full))
            matched_probe = SIZEMATCH_RUN.format(rep=rep, domain="indoor", percent=args.sizematch_percent, seed=store.seeds[0])
            if (store.root / matched_probe).is_dir():
                _, matched = transfer_table(store, SIZEMATCH_RUN, "size-matched", rep=rep, percent=args.sizematch_percent)
                # Drone (334 training poses) is the reference size, so its row is the full Drone model.
                matched.update({("Drone", test): full[("Drone", test)] for test in DOMAINS})
                families.append((f"size-matched ({args.sizematch_percent}% of Indoor/Outdoor groups)", matched))
        folds = sorted(
            int(path.name.removeprefix("rotation_fold").split("_", 1)[0])
            for path in store.root.glob(f"rotation_fold*_mlp_{rep}_indoor_mask20_seed{args.rotation_seed}_eval")
        )
        if folds:
            _, rotated = transfer_table(
                store, ROTATION_RUN, "rotation", replicates=folds, key="fold", rep=rep, seed=args.rotation_seed
            )
            families.append((f"group rotation ({len(folds)} folds, seed {args.rotation_seed})", rotated))
        for family, cells in families:
            frame = cells_frame(cells, family)
            transfer_frames.append(add_penalties(frame, frame).assign(representation=representation))
            for metric in ("nmje", "common_nmje"):
                spread_frames.append(
                    per_replicate_penalties(cells, metric).assign(representation=representation, source_family=family)
                )
            transfer_tests(f"{representation}, {family}", cells)
    if transfer_frames:
        written.append(write(pd.concat(transfer_frames, ignore_index=True), "multiseed_transfer"))
        written.append(write(pd.concat(spread_frames, ignore_index=True), "multiseed_penalty_spread"))

    shift = {name: store.per_seed(template, 20, ("Outdoor",)) for name, template in SHIFT}
    shift_rows = [
        _row({"representation": name, "train_domain": "Indoor", "test_domain": "Outdoor"}, shift[name])
        for name, _ in SHIFT
    ]
    written.append(write(pd.DataFrame(shift_rows), "multiseed_normalization_shift"))
    for first, second in (("raw", "scale_normalized"), ("bbox_normalized", "scale_normalized")):
        tests.append(_paired("normalization shift Indoor->Outdoor", first, second, shift[first], shift[second]))

    written.append(write(per_joint_table(store), "multiseed_per_joint"))
    written.append(write(strata_table(store), "multiseed_strata"))
    written.append(write(within_domain_strata_table(store), "multiseed_strata_within_domain"))
    written.append(write(representation_by_domain_table(store), "multiseed_representation_by_domain"))
    written.append(write(denominator_table(store), "multiseed_denominator"))
    written.append(write(split_footage_table(PROJECT_ROOT / args.processed), "multiseed_split_footage"))
    written.append(write(training_budget_table(PROJECT_ROOT / "outputs" / "logs", store.seeds), "multiseed_training_budget"))
    written.append(write(pd.DataFrame(tests), "multiseed_paired_tests"))
    templates = [template for _, template in (*REPRESENTATIONS, *STATIC_MODELS[2:], *TEMPORAL_MODELS, *SHIFT[::2])]
    templates += [DATAEFF_RUN.replace("{percent}", str(percent)) for percent in (25, 50, 100)]
    templates += [
        DOMAIN_RUN.replace("{rep}", rep).replace("{domain}", domain.lower())
        for rep in TRANSFER_REPRESENTATIONS
        for domain in DOMAINS
    ]
    # Per-run values behind Delta_size and Delta_rot (rotation runs exist for one seed only).
    templates += [
        SIZEMATCH_RUN.replace("{rep}", rep).replace("{domain}", domain.lower()).replace("{percent}", str(args.sizematch_percent))
        for rep in TRANSFER_REPRESENTATIONS
        for domain in DOMAINS[:2]
    ]
    templates += [
        ROTATION_RUN.replace("{rep}", rep).replace("{domain}", domain.lower()).replace("{fold}", str(fold))
        for rep in TRANSFER_REPRESENTATIONS
        for fold in range(6)
        for domain in DOMAINS
    ]
    written.append(write(runs_table(store, templates), "multiseed_runs"))
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, default=Path("outputs/multiseed/evaluations"))
    parser.add_argument("--tables-dir", type=Path, default=Path("outputs/tables/reconstruction"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--sizematch-percent", type=int, default=50)
    parser.add_argument("--rotation-seed", type=int, default=42)
    parser.add_argument("--processed", type=Path, default=Path("data/processed/trackid3x3/dataset.npz"))
    parser.add_argument(
        "--denominator",
        choices=("native", "common"),
        default="native",
        help="native: stored source-box metrics; common: GT pose-tight box for every domain",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        written = aggregate(parse_args(argv))
    except (MultiSeedError, KeyError, ValueError) as exc:
        print(f"Multi-seed aggregation failed: {exc}", file=sys.stderr)
        return 2
    for path in written:
        shown = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
        print(f"Wrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
