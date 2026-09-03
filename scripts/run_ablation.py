#!/usr/bin/env python
"""Aggregate executed TrackID3x3 reconstruction experiments without inventing results.

Only leakage-audited masked-keypoint reconstruction artifacts enter the public
tables and figures.  A legacy discovery helper remains available for older code,
but the command-line workflow is reconstruction-only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import RECONSTRUCTION_SUMMARY_METRICS, SUMMARY_METRICS
from src.visualization.reconstruction import (
    plot_cross_domain_matrix,
    plot_nmje_by_model,
    plot_pck_curves,
    plot_reconstruction_data_efficiency,
)

TITLE = "Cross-Domain Robustness of Lightweight Skeleton Representations for 3x3 Basketball Player Pose Analysis"
RECONSTRUCTION_TASKS = {
    "keypoint_reconstruction",
    "pose_reconstruction",
    "masked_keypoint_reconstruction",
}
DOMAINS = ("Indoor", "Outdoor", "Drone")
RECONSTRUCTION_METRICS = (*RECONSTRUCTION_SUMMARY_METRICS, "mean_joint_error_coordinate_units")
EXPLORATORY_SCOPE = "single-seed exploratory research result"
DRONE_EXPLORATORY_SCOPE = (
    "single-seed exploratory; Drone uses distinct metric-bbox provenance"
)

REPRESENTATION_DEFINITIONS: tuple[dict[str, str], ...] = (
    {
        "representation": "raw",
        "definition": "Source pixel x/y coordinates; missing inputs are zero-filled after masking.",
        "inverse_for_metrics": "Identity; metrics are computed in source pixels.",
    },
    {
        "representation": "bbox_normalized",
        "definition": "Per-frame (x-x0)/w and (y-y0)/h; official MOT boxes are used directly, while pose-tight fallback frames use a post-mask observed-joint box.",
        "inverse_for_metrics": "Multiply by the exact leakage-safe input-transform width/height and add its origin; use the separately stored source bbox only for metric normalization.",
    },
    {
        "representation": "root_centered",
        "definition": "Coordinates centred using an observed-only anatomical/bbox fallback centre.",
        "inverse_for_metrics": "Add the stored observed-only centre.",
    },
    {
        "representation": "scale_normalized",
        "definition": "Observed-only centred coordinates divided by an observed-only scale.",
        "inverse_for_metrics": "Multiply by the stored scale and add the stored centre.",
    },
    {
        "representation": "normalized_angles",
        "definition": "Scale-normalized coordinates with supported observed-joint angles as auxiliary context.",
        "inverse_for_metrics": "Invert only predicted x/y coordinates; angles are context, not targets.",
    },
)


def _nested(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _load_json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _as_finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _canonical_domain(value: Any) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    lookup = {name.lower(): name for name in DOMAINS}
    return lookup.get(token.lower(), token or None)


def _canonical_representation(value: Any) -> str:
    token = str(value or "unknown").strip().lower().replace("-", "_").replace("+", "_")
    aliases = {"normalized": "bbox_normalized", "bbox": "bbox_normalized", "root": "root_centered"}
    return aliases.get(token, token)


def _normalise_reconstruction_metrics(mapping: Mapping[str, Any]) -> dict[str, float]:
    aliases: dict[str, tuple[str, ...]] = {
        "nmje": ("nmje", "NMJE", "normalized_mean_joint_error"),
        "pck_at_0_05": ("pck_at_0_05", "PCK@0.05", "pck_0.05", "pck05"),
        "pck_at_0_10": ("pck_at_0_10", "PCK@0.10", "pck_0.10", "pck10"),
        "pck_at_0_20": ("pck_at_0_20", "PCK@0.20", "pck_0.20", "pck20"),
        "mean_joint_error_coordinate_units": (
            "mean_joint_error_coordinate_units",
            "mpjpe",
            "mean_joint_error",
        ),
    }
    pck = mapping.get("pck") if isinstance(mapping.get("pck"), Mapping) else {}
    result: dict[str, float] = {}
    for canonical, names in aliases.items():
        candidates = [mapping.get(name) for name in names]
        if canonical.startswith("pck_at_"):
            threshold = canonical.removeprefix("pck_at_").replace("_", ".")
            candidates.extend([pck.get(threshold), pck.get(f"{float(threshold):.2f}")])
        value = next((number for item in candidates if (number := _as_finite(item)) is not None), None)
        if value is not None:
            result[canonical] = value
    return result


def _reconstruction_metric_bundle(payload: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]] | None:
    candidates = (
        "test",
        "test_metrics",
        "evaluation.metrics",
        "evaluation",
        "metrics",
        "overall",
    )
    for path in candidates:
        value = _nested(payload, path)
        if isinstance(value, Mapping) and "metrics" in value and isinstance(value["metrics"], Mapping):
            value = value["metrics"]
        if isinstance(value, Mapping) and "nmje" in _normalise_reconstruction_metrics(value):
            return path, value
    if "nmje" in _normalise_reconstruction_metrics(payload):
        return "root", payload
    return None


def _task(payload: Mapping[str, Any]) -> str:
    return str(
        _first_present(
            payload.get("task"),
            _nested(payload, "experiment.task"),
            _nested(payload, "config.task"),
            _nested(payload, "config.project.task"),
            "",
        )
    ).strip().lower()


def _audit_passed(payload: Mapping[str, Any]) -> bool:
    values = (
        payload.get("leakage_audit_passed"),
        _nested(payload, "leakage_audit.passed"),
        _nested(payload, "split_audit.passed"),
        _nested(payload, "experiment.leakage_audit_passed"),
        _nested(payload, "experiment.leakage_audit.passed"),
    )
    return any(value is True for value in values)


def discover_reconstruction_records(
    logs_root: str | Path,
    *,
    include_synthetic: bool = False,
    include_smoke: bool = False,
    require_leakage_audit: bool = True,
) -> pd.DataFrame:
    """Discover executed TrackID reconstruction test artifacts only.

    Classification artifacts, validation-only trainer summaries, malformed JSON,
    and records lacking an explicit leakage audit are excluded.
    """

    root = Path(logs_root)
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame()
    for path in sorted(root.rglob("metrics.json")):
        payload = _load_json(path)
        if payload is None or _task(payload) not in RECONSTRUCTION_TASKS:
            continue
        bundle = _reconstruction_metric_bundle(payload)
        if bundle is None:
            continue
        metric_source, metric_values = bundle
        synthetic = bool(
            _first_present(payload.get("synthetic"), _nested(payload, "provenance.synthetic"), False)
        )
        if synthetic and not include_synthetic:
            continue
        smoke = bool(
            payload.get("smoke", False)
            or _nested(payload, "experiment.smoke", False)
            or "smoke" in str(path).lower()
            or "smoke" in str(payload.get("run", "")).lower()
            or "smoke" in str(payload.get("processed_dataset", "")).lower()
            or "smoke" in str(payload.get("checkpoint", "")).lower()
        )
        if smoke and not include_smoke:
            continue
        passed = _audit_passed(payload)
        if require_leakage_audit and not passed:
            continue
        experiment = payload.get("experiment") if isinstance(payload.get("experiment"), Mapping) else {}
        train_domain = _canonical_domain(
            _first_present(
                payload.get("train_domain"),
                experiment.get("train_domain"),
                _nested(payload, "protocol.train_domain"),
                payload.get("domain"),
            )
        )
        test_domain = _canonical_domain(
            _first_present(
                payload.get("test_domain"),
                experiment.get("test_domain"),
                _nested(payload, "protocol.test_domain"),
                train_domain,
            )
        )
        if train_domain and test_domain:
            protocol = "within_domain" if train_domain == test_domain else "cross_domain"
        else:
            protocol = str(_first_present(payload.get("protocol"), experiment.get("protocol"), "unspecified"))
        input_transform_audit = payload.get("input_transform_audit_passed") is True
        # Pose-tight Drone boxes enclose the complete target pose.  Artifacts
        # produced before the observed-only input-transform audit are unsafe to
        # pool, even though that same stored box remains valid as metric scale.
        if test_domain == "Drone" and not input_transform_audit:
            continue
        mask_ratio = _as_finite(
            _first_present(payload.get("mask_ratio"), experiment.get("mask_ratio"), _nested(payload, "evaluation.mask_ratio"))
        )
        if mask_ratio is not None and mask_ratio > 1:
            mask_ratio /= 100.0
        data_fraction = next(
            (
                finite
                for value in (
                    payload.get("data_percent"),
                    payload.get("data_fraction"),
                    payload.get("data_efficiency"),
                    experiment.get("data_percent"),
                    experiment.get("data_fraction"),
                )
                if (finite := _as_finite(value)) is not None
            ),
            None,
        )
        if data_fraction is not None and data_fraction <= 1:
            data_percent = 100.0 * data_fraction
        else:
            data_percent = data_fraction
        row: dict[str, Any] = {
            "task": "keypoint_reconstruction",
            "source": str(path.resolve()),
            "run": path.parent.name,
            "metric_source": metric_source,
            "synthetic": synthetic,
            "smoke": smoke,
            "result_scope": (
                "software validation only / NOT research results"
                if synthetic
                else "software smoke run / NOT primary research result"
                if smoke
                else EXPLORATORY_SCOPE
            ),
            "leakage_audit_passed": passed,
            "input_transform_audit_passed": input_transform_audit,
            "grouping_unit": str(
                _first_present(
                    payload.get("grouping_unit"),
                    experiment.get("grouping_unit"),
                    _nested(payload, "leakage_audit.grouping_unit"),
                    "unknown",
                )
            ),
            "protocol": protocol,
            "train_domain": train_domain,
            "test_domain": test_domain,
            "model": str(
                _first_present(payload.get("model"), payload.get("model_name"), experiment.get("model"), "unknown")
            ),
            "representation": _canonical_representation(
                _first_present(payload.get("representation"), experiment.get("representation"), _nested(payload, "config.representation"))
            ),
            "mask_ratio": mask_ratio,
            "mask_percentage": None if mask_ratio is None else 100.0 * mask_ratio,
            "seed": _first_present(payload.get("seed"), experiment.get("seed"), 42),
            "data_percent": data_percent,
            "sequence_length": _first_present(payload.get("sequence_length"), experiment.get("sequence_length")),
            "temporal_reliability_required": _nested(
                payload,
                "temporal_protocol_audit.requires_reliable_temporal_windows",
            ),
            "temporal_sequences_explicitly_reliable": _nested(
                payload,
                "temporal_protocol_audit.selected_sequences_explicitly_reliable",
            ),
            "checkpoint": _first_present(payload.get("checkpoint"), experiment.get("checkpoint")),
            "parameter_count": _first_present(payload.get("parameter_count"), experiment.get("parameter_count")),
            "checkpoint_size_bytes": _first_present(payload.get("checkpoint_size_bytes"), experiment.get("checkpoint_size_bytes")),
            "training_time_seconds": _first_present(payload.get("training_time_seconds"), experiment.get("training_time_seconds")),
            "inference_time_seconds": _first_present(payload.get("inference_time_seconds"), experiment.get("inference_time_seconds")),
            "n_scored_joints": _first_present(metric_values.get("n_scored_joints"), payload.get("n_scored_joints")),
        }
        if row["checkpoint_size_bytes"] is None and row["checkpoint"]:
            checkpoint_path = Path(str(row["checkpoint"]))
            if not checkpoint_path.is_absolute():
                checkpoint_path = path.parent / checkpoint_path
            if checkpoint_path.is_file():
                row["checkpoint_size_bytes"] = checkpoint_path.stat().st_size
        row.update(_normalise_reconstruction_metrics(metric_values))
        if "baseline" in str(row["model"]).lower():
            row["protocol"] = "baseline_domain_stratified"
            row["train_domain"] = "not_applicable"
        elif str(row["train_domain"]).strip().lower() in {"all", "pooled", "mixed", "all_domains"}:
            row["protocol"] = "pooled_domain"
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    numeric = (
        *RECONSTRUCTION_METRICS,
        "mask_ratio",
        "mask_percentage",
        "data_percent",
        "sequence_length",
        "parameter_count",
        "checkpoint_size_bytes",
        "training_time_seconds",
        "inference_time_seconds",
        "n_scored_joints",
    )
    for column in numeric:
        if column in table:
            table[column] = pd.to_numeric(table[column], errors="coerce")
    dedup = [
        column
        for column in (
            "checkpoint",
            "model",
            "representation",
            "train_domain",
            "test_domain",
            "mask_ratio",
            "seed",
            "data_percent",
            "sequence_length",
            "synthetic",
            "smoke",
        )
        if column in table
    ]
    return table.drop_duplicates(subset=dedup, keep="last").reset_index(drop=True)


def discover_experiment_records(
    logs_root: str | Path, *, include_synthetic: bool = False
) -> pd.DataFrame:
    """Compatibility discovery for legacy classification tests and notebooks.

    New scripts call :func:`discover_reconstruction_records` directly.
    """

    reconstructed = discover_reconstruction_records(
        logs_root,
        include_synthetic=include_synthetic,
        include_smoke=False,
        require_leakage_audit=False,
    )
    if not reconstructed.empty:
        return reconstructed
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(logs_root).rglob("metrics.json")):
        payload = _load_json(path)
        if payload is None or bool(payload.get("synthetic", False)) and not include_synthetic:
            continue
        bundle: Mapping[str, Any] | None = None
        for key in ("test", "test_metrics", "metrics"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping) and any(name in candidate for name in SUMMARY_METRICS):
                bundle = candidate
                break
        if bundle is None:
            continue
        row = {
            "source": str(path.resolve()),
            "model": str(payload.get("model", payload.get("model_name", "unknown"))),
            "representation": str(payload.get("representation", "unknown")),
            "synthetic": bool(payload.get("synthetic", False)),
        }
        for metric in SUMMARY_METRICS:
            value = _as_finite(bundle.get(metric))
            if value is not None:
                row[metric] = value
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_present(
    records: pd.DataFrame,
    group_columns: Sequence[str],
    *,
    metrics: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate only finite observed values; no missing metric is imputed."""

    if records.empty or any(column not in records for column in group_columns):
        return pd.DataFrame()
    names = tuple(metrics or (RECONSTRUCTION_METRICS if "nmje" in records else SUMMARY_METRICS))
    rows: list[dict[str, Any]] = []
    grouper: Any = list(group_columns) if len(group_columns) > 1 else group_columns[0]
    for keys, group in records.groupby(grouper, dropna=False, sort=True):
        values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, values, strict=True))
        for metric in names:
            if metric not in group:
                continue
            observed = pd.to_numeric(group[metric], errors="coerce").dropna()
            observed = observed[np.isfinite(observed)]
            if observed.empty:
                continue
            row[f"{metric}_mean"] = float(observed.mean())
            row[f"{metric}_std"] = float(observed.std(ddof=0))
            row[f"{metric}_n"] = int(observed.size)
        if any(key.endswith("_n") for key in row):
            rows.append(row)
    return pd.DataFrame(rows)


def controlled_variation(
    records: pd.DataFrame,
    varied_column: str,
    *,
    controls: Sequence[str],
    minimum_levels: int = 2,
) -> pd.DataFrame:
    """Return the largest fixed-control slice with enough executed levels."""

    required = [varied_column, *controls]
    if records.empty or any(column not in records for column in required):
        return pd.DataFrame(columns=records.columns)
    best = pd.DataFrame(columns=records.columns)
    grouper: Any = list(controls) if len(controls) > 1 else controls[0]
    for _, group in records.groupby(grouper, dropna=False, sort=False):
        levels = group[varied_column].dropna().astype(str).nunique()
        if levels >= minimum_levels and (best.empty or levels > best[varied_column].dropna().astype(str).nunique() or len(group) > len(best)):
            best = group.copy()
    return best.reset_index(drop=True)


def matched_domain_variation(
    records: pd.DataFrame,
    varied_column: str,
    *,
    controls: Sequence[str],
    minimum_levels: int = 2,
    minimum_domains: int = 1,
) -> pd.DataFrame:
    """Select a fixed-control slice whose levels share the same test domains.

    The intersection prevents a model/representation with extra evaluated
    domains from receiving more weight.  Downstream summaries first average
    within each domain and then take an equal-domain macro mean.
    """

    required = [varied_column, "test_domain", *controls]
    if records.empty or any(column not in records for column in required):
        return pd.DataFrame(columns=records.columns)
    best = pd.DataFrame(columns=records.columns)
    best_score: tuple[int, int, int, int] | None = None
    grouper: Any = list(controls) if len(controls) > 1 else controls[0]
    for _, group in records.groupby(grouper, dropna=False, sort=False):
        usable = group.dropna(subset=[varied_column, "test_domain"])
        levels = usable[varied_column].drop_duplicates().tolist()
        if len(levels) < minimum_levels:
            continue
        domain_sets = [
            set(usable.loc[usable[varied_column] == level, "test_domain"].astype(str))
            for level in levels
        ]
        common_domains = set.intersection(*domain_sets) if domain_sets else set()
        if len(common_domains) < minimum_domains:
            continue
        matched = usable[usable["test_domain"].astype(str).isin(common_domains)].copy()
        primary_scope = int(
            not matched.get("synthetic", pd.Series(False, index=matched.index)).astype(bool).any()
            and not matched.get("smoke", pd.Series(False, index=matched.index)).astype(bool).any()
        )
        score = (len(levels), len(common_domains), primary_scope, len(matched))
        if best_score is None or score > best_score:
            best = matched
            best_score = score
    return best.reset_index(drop=True)


def controlled_model_comparison(
    records: pd.DataFrame,
    *,
    minimum_levels: int = 2,
    minimum_domains: int = 1,
) -> pd.DataFrame:
    """Find a matched-domain model slice with one shared temporal protocol."""

    required = {
        "model",
        "representation",
        "test_domain",
        "mask_ratio",
        "data_percent",
        "sequence_length",
        "synthetic",
        "smoke",
    }
    if records.empty or not required.issubset(records.columns):
        return pd.DataFrame(columns=records.columns)
    baseline = records["model"].astype(str).str.contains("baseline", case=False, na=False)
    pooled = records["train_domain"].astype(str).str.lower().isin(
        {"all", "pooled", "mixed", "all_domains"}
    )
    candidates = records[baseline | pooled]
    return matched_domain_variation(
        candidates,
        "model",
        controls=(
            "representation",
            "mask_ratio",
            "data_percent",
            "sequence_length",
            "synthetic",
            "smoke",
        ),
        minimum_levels=minimum_levels,
        minimum_domains=minimum_domains,
    )


def _numeric_match(records: pd.DataFrame, column: str, value: float) -> pd.Series:
    if column not in records:
        return pd.Series(False, index=records.index, dtype=bool)
    numeric = pd.to_numeric(records[column], errors="coerce").to_numpy(dtype=float)
    return pd.Series(np.isclose(numeric, value, rtol=0.0, atol=1e-9), index=records.index)


def _preferred_comparison_slices(records: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Apply the declared primary protocols before any result aggregation."""

    empty = pd.DataFrame(columns=records.columns)
    if records.empty:
        return {
            name: empty.copy()
            for name in (
                "representation",
                "normalization_shift",
                "model",
                "missingness",
                "domain",
                "cross_domain",
                "data_efficiency",
            )
        }
    model = records["model"].astype(str).str.lower()
    representation = records["representation"].astype(str).str.lower()
    train_domain = records["train_domain"].astype(str).str.lower()
    test_domain = records["test_domain"].astype(str).str.lower()
    pooled = train_domain.isin({"all", "pooled", "mixed", "all_domains"})
    seed_42 = _numeric_match(records, "seed", 42.0)
    declared_representations = {
        "raw",
        "bbox_normalized",
        "root_centered",
        "scale_normalized",
        "normalized_angles",
    }
    declared_domains = {domain.lower() for domain in DOMAINS}
    temporal_required = records.get(
        "temporal_reliability_required",
        pd.Series(False, index=records.index, dtype=bool),
    ).eq(True)
    temporal_reliable = records.get(
        "temporal_sequences_explicitly_reliable",
        pd.Series(False, index=records.index, dtype=bool),
    ).eq(True)
    parameter_count = pd.to_numeric(
        records.get("parameter_count", pd.Series(np.nan, index=records.index)),
        errors="coerce",
    )
    primary_mlp_t1 = (
        model.eq("mlp")
        & pooled
        & seed_42
        & _numeric_match(records, "data_percent", 100.0)
        & _numeric_match(records, "sequence_length", 1.0)
    )

    representation_candidates = records[
        primary_mlp_t1
        & _numeric_match(records, "mask_ratio", 0.20)
        & representation.isin(declared_representations)
        & test_domain.isin(declared_domains)
    ]
    representation_slice = matched_domain_variation(
        representation_candidates,
        "representation",
        controls=("synthetic", "smoke"),
        minimum_levels=5,
        minimum_domains=3,
    )

    normalization_shift_candidates = records[
        model.eq("mlp")
        & train_domain.eq("indoor")
        & records["test_domain"].astype(str).str.lower().eq("outdoor")
        & representation.isin({"raw", "bbox_normalized", "scale_normalized"})
        & _numeric_match(records, "mask_ratio", 0.20)
        & _numeric_match(records, "data_percent", 100.0)
        & _numeric_match(records, "sequence_length", 1.0)
        & seed_42
    ]
    normalization_shift_slice = matched_domain_variation(
        normalization_shift_candidates,
        "representation",
        controls=("synthetic", "smoke"),
        minimum_levels=3,
        minimum_domains=1,
    )

    normalized_angles = representation.eq("normalized_angles")
    declared_mask_ratios = (
        _numeric_match(records, "mask_ratio", 0.10)
        | _numeric_match(records, "mask_ratio", 0.20)
        | _numeric_match(records, "mask_ratio", 0.30)
        | _numeric_match(records, "mask_ratio", 0.40)
    )
    missingness_candidates = records[
        primary_mlp_t1
        & normalized_angles
        & declared_mask_ratios
        & test_domain.isin(declared_domains)
    ]
    missingness_slice = matched_domain_variation(
        missingness_candidates,
        "mask_ratio",
        controls=("synthetic", "smoke"),
        minimum_levels=4,
        minimum_domains=3,
    )

    model_candidates = records[
        representation.eq("bbox_normalized")
        & model.isin({"geometric_baseline", "mlp", "tcn", "graph"})
        & _numeric_match(records, "mask_ratio", 0.20)
        & _numeric_match(records, "data_percent", 100.0)
        & _numeric_match(records, "sequence_length", 16.0)
        & seed_42
        & test_domain.isin({"indoor", "outdoor"})
        & (~model.eq("tcn") | (temporal_required & temporal_reliable))
        & (~model.eq("graph") | parameter_count.lt(1_000_000))
    ]
    model_slice = controlled_model_comparison(
        model_candidates,
        minimum_levels=4,
        minimum_domains=2,
    )

    domain_slice = records[
        primary_mlp_t1
        & normalized_angles
        & _numeric_match(records, "mask_ratio", 0.20)
        & test_domain.isin(declared_domains)
    ].copy()

    named_train_domain = train_domain.isin({domain.lower() for domain in DOMAINS})
    cross_candidates = records[
        model.eq("mlp")
        & representation.eq("bbox_normalized")
        & named_train_domain
        & _numeric_match(records, "mask_ratio", 0.20)
        & _numeric_match(records, "data_percent", 100.0)
        & _numeric_match(records, "sequence_length", 1.0)
        & seed_42
        & test_domain.isin(declared_domains)
    ]
    cross_slice = matched_domain_variation(
        cross_candidates,
        "train_domain",
        controls=("synthetic", "smoke"),
        minimum_levels=3,
        minimum_domains=3,
    )

    declared_data_percentages = (
        _numeric_match(records, "data_percent", 25.0)
        | _numeric_match(records, "data_percent", 50.0)
        | _numeric_match(records, "data_percent", 100.0)
    )
    efficiency_candidates = records[
        model.eq("mlp")
        & pooled
        & normalized_angles
        & _numeric_match(records, "mask_ratio", 0.20)
        & _numeric_match(records, "sequence_length", 1.0)
        & seed_42
        & declared_data_percentages
        & test_domain.isin(declared_domains)
    ]
    efficiency_slice = matched_domain_variation(
        efficiency_candidates,
        "data_percent",
        controls=("synthetic", "smoke"),
        minimum_levels=3,
        minimum_domains=3,
    )
    return {
        "representation": representation_slice,
        "normalization_shift": normalization_shift_slice,
        "model": model_slice,
        "missingness": missingness_slice,
        "domain": domain_slice,
        "cross_domain": cross_slice,
        "data_efficiency": efficiency_slice,
    }


def aggregate_equal_domains(
    records: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Macro-average domain means so every matched domain has equal weight."""

    if records.empty or "test_domain" not in records:
        return pd.DataFrame()
    per_domain = aggregate_present(records, (*group_columns, "test_domain"))
    if per_domain.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, source in per_domain.iterrows():
        row = {column: source[column] for column in group_columns}
        row["test_domain"] = source["test_domain"]
        for metric in RECONSTRUCTION_METRICS:
            mean_column = f"{metric}_mean"
            if mean_column in per_domain and pd.notna(source[mean_column]):
                row[metric] = source[mean_column]
        rows.append(row)
    result = aggregate_present(pd.DataFrame(rows), group_columns)
    if not result.empty:
        domain_counts = (
            per_domain.groupby(list(group_columns), dropna=False, as_index=False)[
                "test_domain"
            ]
            .nunique()
            .rename(columns={"test_domain": "domains_averaged"})
        )
        result = result.merge(domain_counts, on=list(group_columns), how="left", validate="one_to_one")
    return result


COMPARISON_POLICIES: dict[str, str] = {
    "representation": (
        "Pooled-domain MLP comparison of all five declared representations at T=1, 20% "
        "masking, 100% data, and seed 42; macro-average the same Indoor, Outdoor, and "
        "Drone rows."
    ),
    "normalization_shift": (
        "Single-seed Indoor-trained to Outdoor-tested MLP normalization hypothesis; "
        "raw, bbox-normalized, and scale-normalized representations at T=1, 20% masking, "
        "100% training data, and seed 42."
    ),
    "model": (
        "T=16 bbox-normalized comparison of the geometric baseline, MLP, reliable-window "
        "TCN, and sub-1M-parameter graph model at 20% masking, 100% data, and seed 42; "
        "macro-average the matched Indoor and Outdoor domains."
    ),
    "missingness": (
        "Pooled-domain normalized+angles MLP at T=1, 100% data, and seed 42; vary only "
        "10%, 20%, 30%, and 40% masking and macro-average all three test domains."
    ),
    "domain": (
        "Pooled-domain normalized+angles MLP at T=1, 20% masking, 100% data, and seed 42; "
        "report each test domain separately."
    ),
    "cross_domain": (
        "Domain-trained bbox-normalized MLP at T=1, 20% masking, 100% data, and seed 42; "
        "require all three train and test domains before reporting cross-domain cells."
    ),
    "data_efficiency": (
        "Pooled-domain normalized+angles MLP at T=1, 20% masking, and seed 42; compare "
        "25%, 50%, and 100% whole-group training data across all three test domains."
    ),
}


def _comparison_audit(slices: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    varied_columns = {
        "representation": "representation",
        "normalization_shift": "representation",
        "model": "model",
        "missingness": "mask_ratio",
        "domain": "test_domain",
        "cross_domain": "train_domain",
        "data_efficiency": "data_percent",
    }
    equal_domain = {
        "representation",
        "normalization_shift",
        "model",
        "missingness",
        "data_efficiency",
    }
    audit: dict[str, Any] = {}
    control_columns = (
        "model",
        "representation",
        "train_domain",
        "mask_ratio",
        "data_percent",
        "sequence_length",
        "seed",
        "synthetic",
        "smoke",
    )
    for name, table in slices.items():
        varied = varied_columns[name]
        entry: dict[str, Any] = {
            "selection_policy": COMPARISON_POLICIES[name],
            "status": "not generated: required matched executed evidence absent"
            if table.empty
            else "generated from executed evidence",
            "domain_aggregation": (
                "equal-domain macro mean after within-domain run averaging"
                if name in equal_domain
                else "per-domain cell; no domain pooling"
            ),
        }
        if not table.empty:
            entry["levels"] = sorted(
                [
                    value.item() if isinstance(value, np.generic) else value
                    for value in table[varied].dropna().unique().tolist()
                ],
                key=str,
            )
            entry["test_domains"] = sorted(table["test_domain"].dropna().astype(str).unique().tolist())
            controls: dict[str, Any] = {}
            for column in control_columns:
                if column == varied or column not in table:
                    continue
                values = [
                    value.item() if isinstance(value, np.generic) else value
                    for value in table[column].dropna().unique().tolist()
                ]
                if values:
                    controls[column] = values[0] if len(values) == 1 else sorted(values, key=str)
            entry["selected_controls"] = controls
            entry["source_artifacts"] = sorted(table["source"].astype(str).unique().tolist())
        audit[name] = entry
    return audit


def _text(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item) for item in values.reshape(-1)],
        dtype=str,
    ).reshape(values.shape)


def _first_array(data: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray | None:
    for name in names:
        if name in data:
            return np.asarray(data[name])
    return None


def _load_processed_arrays(path: str | Path | None) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "dataset.npz"
    if not candidate.is_file():
        return None
    try:
        with np.load(candidate, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError):
        return None


def dataset_tables(data: Mapping[str, np.ndarray] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build observed dataset/subset and original-missingness tables."""

    if not data:
        return pd.DataFrame(), pd.DataFrame()
    coords = _first_array(data, ("coords", "keypoints", "coordinates"))
    valid = _first_array(data, ("valid_mask", "original_valid_mask", "keypoint_mask"))
    domains = _first_array(data, ("domains", "domain", "capture_domain"))
    if coords is None or valid is None or domains is None or coords.ndim != 4 or valid.shape != coords.shape[:-1]:
        return pd.DataFrame(), pd.DataFrame()
    domains = _text(domains).reshape(-1)
    sequence = _first_array(data, ("sequence_mask", "track_valid_mask", "window_valid_mask"))
    if sequence is None:
        sequence = np.ones(coords.shape[:2], dtype=bool)
    else:
        sequence = np.asarray(sequence, dtype=bool)
    video = _first_array(data, ("video_ids", "video_id", "videos"))
    track = _first_array(data, ("track_ids", "track_id", "player_ids"))
    frame_ids = _first_array(data, ("frame_ids", "frame_index", "frames"))
    source_files = _first_array(data, ("source_file", "source_files"))
    source_path_index: dict[str, list[Path]] = {}
    source_root = PROJECT_ROOT / "external" / "TrackID3x3" / "ground_truth"
    if source_files is not None and source_root.is_dir():
        wanted = {Path(str(value)).name for value in np.asarray(source_files).reshape(-1)}
        for candidate in source_root.rglob("*"):
            if candidate.is_file() and candidate.name in wanted:
                source_path_index.setdefault(candidate.name, []).append(candidate)
    video_text = _text(video).reshape(-1) if video is not None else None
    track_text = _text(track).reshape(-1) if track is not None else None
    joint_names = _first_array(data, ("joint_names",))
    names = _text(joint_names).reshape(-1).tolist() if joint_names is not None else [f"joint_{i}" for i in range(coords.shape[2])]
    synthetic_array = _first_array(data, ("synthetic",))
    synthetic = bool(np.asarray(synthetic_array).reshape(-1)[0]) if synthetic_array is not None and synthetic_array.size else False
    subset_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for domain in sorted(set(domains.tolist()), key=lambda item: (DOMAINS.index(item) if item in DOMAINS else 99, item)):
        selected = domains == domain
        frame_mask = sequence[selected]
        joint_eligible = np.broadcast_to(frame_mask[..., None], valid[selected].shape)
        observed = np.asarray(valid[selected], dtype=bool) & joint_eligible
        available_videos: np.ndarray | None = None
        if video_text is not None:
            available_videos = np.asarray(
                [value for value in video_text[selected] if str(value).strip()], dtype=str
            )
        frame_count: int | None = None
        if frame_ids is not None and np.asarray(frame_ids).shape[:2] == coords.shape[:2]:
            scoped_frames: set[str] = set()
            selected_indices = np.flatnonzero(selected)
            for local_index, source_index in enumerate(selected_indices):
                for time_index in np.flatnonzero(frame_mask[local_index]):
                    video_key = (
                        video_text[source_index]
                        if video_text is not None and str(video_text[source_index]).strip()
                        else (
                            str(source_files[source_index, time_index])
                            if source_files is not None and np.asarray(source_files).ndim >= 2
                            else str(track_text[source_index]) if track_text is not None else str(source_index)
                        )
                    )
                    scoped_frames.add(f"{domain}::{video_key}::{np.asarray(frame_ids)[source_index, time_index]}")
            frame_count = len(scoped_frames)
        player_tracks: int | None = None
        if track_text is not None:
            scoped_players = {
                (
                    f"{domain}::{video_text[index]}::{track_text[index]}"
                    if video_text is not None and str(video_text[index]).strip()
                    else f"{domain}::{track_text[index]}"
                )
                for index in np.flatnonzero(selected)
                if str(track_text[index]).strip()
            }
            if scoped_players:
                player_tracks = len(scoped_players)
        storage_bytes = 0
        storage_known = False
        domain_storage = source_root / domain
        if domain_storage.is_dir():
            storage_bytes = sum(
                path.stat().st_size for path in domain_storage.rglob("*") if path.is_file()
            )
            storage_known = True
        elif source_files is not None:
            raw_paths = np.asarray(source_files)[selected].reshape(-1)
            for raw_path in sorted({str(value) for value in raw_paths if str(value).strip()}):
                candidate = Path(raw_path)
                if not candidate.is_absolute():
                    candidate = PROJECT_ROOT / candidate
                if candidate.is_file():
                    storage_bytes += candidate.stat().st_size
                    storage_known = True
                elif Path(raw_path).name in source_path_index:
                    for matched in source_path_index[Path(raw_path).name]:
                        storage_bytes += matched.stat().st_size
                        storage_known = True
        storage = (
            f"{storage_bytes / (1024 * 1024):.2f} MiB" if storage_known else "not available in artifact"
        )
        pose_instances = int((valid[selected].any(axis=-1) & frame_mask).sum())
        row: dict[str, Any] = {
            "Domain": domain,
            "Videos": (
                int(np.unique(available_videos).size)
                if available_videos is not None and available_videos.size
                else "unavailable / not linkable"
            ),
            "Frames": frame_count if frame_count is not None else np.nan,
            "Player instances": pose_instances,
            "Player tracks": player_tracks if player_tracks is not None else np.nan,
            "Pose instances": pose_instances,
            "Storage": storage,
        }
        subset_rows.append(row)
        for joint_index, joint in enumerate(names):
            joint_possible = int(frame_mask.sum())
            joint_valid = int(observed[..., joint_index].sum())
            missing_rows.append(
                {
                    "Domain": domain,
                    "Joint": joint,
                    "Joint index": joint_index,
                    "Valid keypoints": joint_valid,
                    "Possible keypoints": joint_possible,
                    "Missing fraction": float(1.0 - joint_valid / joint_possible) if joint_possible else np.nan,
                    "Synthetic": synthetic,
                }
            )
    return pd.DataFrame(subset_rows), pd.DataFrame(missing_rows)


def _with_scope(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty or "result_scope" in table:
        return table
    result = table.copy()
    if "synthetic" in result:
        result["result_scope"] = np.where(
            result["synthetic"].astype(bool),
            "software validation only / NOT research results",
            EXPLORATORY_SCOPE,
        )
    return result


def _scope_labels(
    table: pd.DataFrame,
    *,
    domain_column: str | None = None,
    other_domain_column: str | None = None,
    includes_drone: bool = False,
) -> np.ndarray:
    """Label empirical rows as single-seed and flag Drone bbox provenance."""

    if table.empty:
        return np.asarray([], dtype=str)
    synthetic = table.get("synthetic", pd.Series(False, index=table.index)).astype(bool)
    smoke = table.get("smoke", pd.Series(False, index=table.index)).astype(bool)
    drone = pd.Series(includes_drone, index=table.index, dtype=bool)
    if domain_column is not None and domain_column in table:
        drone = table[domain_column].astype(str).str.lower().eq("drone")
    if other_domain_column is not None and other_domain_column in table:
        drone |= table[other_domain_column].astype(str).str.lower().eq("drone")
    primary = np.where(drone, DRONE_EXPLORATORY_SCOPE, EXPLORATORY_SCOPE)
    return np.where(
        synthetic,
        "software validation only / NOT research results",
        np.where(smoke, "software smoke run / NOT primary research result", primary),
    )


def _markdown_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "**NOT GENERATED: no executed evidence available.**\n"
    rendered = table.copy()
    for column in rendered:
        if pd.api.types.is_float_dtype(rendered[column]):
            rendered[column] = rendered[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    rendered = rendered.fillna("").astype(str).map(lambda value: value.replace("|", "\\|"))
    header = "| " + " | ".join(rendered.columns) + " |"
    separator = "| " + " | ".join("---" for _ in rendered.columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in rendered.to_numpy().tolist()]
    return "\n".join([header, separator, *rows]) + "\n"


def _write_table(table: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(_markdown_table(table), encoding="utf-8")


def _aggregate_result_tables(records: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if records.empty:
        return {
            "within_domain_metrics": pd.DataFrame(
                columns=["Domain", "NMJE", "PCK@.10", "Runs", "Result scope"]
            ),
            "cross_domain_metrics": pd.DataFrame(
                columns=["Train domain", "Test domain", "NMJE", "PCK", "Runs", "Result scope"]
            ),
        }

    def fixed_configuration(source: pd.DataFrame) -> pd.DataFrame:
        controls = [
            column
            for column in (
                "model",
                "representation",
                "mask_ratio",
                "data_percent",
                "sequence_length",
                "synthetic",
                "smoke",
            )
            if column in source
        ]
        if source.empty or not controls:
            return source
        best = pd.DataFrame(columns=source.columns)
        grouper: Any = controls if len(controls) > 1 else controls[0]
        for _, group in source.groupby(grouper, dropna=False, sort=False):
            if len(group) > len(best):
                best = group.copy()
        return best

    within_source = records[records["protocol"] == "within_domain"]
    within_source = fixed_configuration(within_source)
    within_raw = aggregate_present(within_source, ("test_domain", "synthetic", "smoke"))
    within = pd.DataFrame(
        {
            "Domain": within_raw.get("test_domain", pd.Series(dtype=str)),
            "NMJE": within_raw.get("nmje_mean", pd.Series(dtype=float)),
            "PCK@.10": within_raw.get("pck_at_0_10_mean", pd.Series(dtype=float)),
            "Runs": within_raw.get("nmje_n", pd.Series(dtype=float)),
            "Result scope": _scope_labels(within_raw, domain_column="test_domain"),
        }
    )
    cross_source = fixed_configuration(records[records["protocol"] == "cross_domain"])
    cross_raw = aggregate_present(cross_source, ("train_domain", "test_domain", "synthetic", "smoke"))
    cross = pd.DataFrame(
        {
            "Train domain": cross_raw.get("train_domain", pd.Series(dtype=str)),
            "Test domain": cross_raw.get("test_domain", pd.Series(dtype=str)),
            "NMJE": cross_raw.get("nmje_mean", pd.Series(dtype=float)),
            "PCK": cross_raw.get("pck_at_0_10_mean", pd.Series(dtype=float)),
            "Runs": cross_raw.get("nmje_n", pd.Series(dtype=float)),
            "Result scope": _scope_labels(
                cross_raw,
                domain_column="test_domain",
                other_domain_column="train_domain",
            ),
        }
    )
    return {"within_domain_metrics": within, "cross_domain_metrics": cross}


def _model_table(records: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Model",
        "Parameters",
        "Checkpoint MB",
        "NMJE",
        "PCK@.10",
        "Domains averaged",
        "Result scope",
    ]
    if records.empty:
        return pd.DataFrame(columns=columns)
    summary = aggregate_equal_domains(records, ("model", "synthetic", "smoke"))
    if summary.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for _, metric_row in summary.iterrows():
        model = metric_row["model"]
        synthetic = bool(metric_row["synthetic"])
        smoke = bool(metric_row["smoke"])
        group = records[
            records["model"].eq(model)
            & records["synthetic"].astype(bool).eq(synthetic)
            & records["smoke"].astype(bool).eq(smoke)
        ]
        row: dict[str, Any] = {
            "Model": model,
            "NMJE": metric_row.get("nmje_mean", np.nan),
            "PCK@.10": metric_row.get("pck_at_0_10_mean", np.nan),
            "Domains averaged": metric_row.get("domains_averaged", np.nan),
            "Result scope": _scope_labels(
                pd.DataFrame({"synthetic": [synthetic], "smoke": [smoke]})
            )[0],
        }
        parameters = (
            pd.to_numeric(group["parameter_count"], errors="coerce").dropna()
            if "parameter_count" in group
            else pd.Series(dtype=float)
        )
        sizes = (
            pd.to_numeric(group["checkpoint_size_bytes"], errors="coerce").dropna()
            if "checkpoint_size_bytes" in group
            else pd.Series(dtype=float)
        )
        row["Parameters"] = (
            float(parameters.mean())
            if not parameters.empty
            else 0.0
            if "baseline" in str(model).lower()
            else np.nan
        )
        row["Checkpoint MB"] = (
            float(sizes.mean() / 1_000_000)
            if not sizes.empty
            else 0.0
            if "baseline" in str(model).lower()
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _representation_result_table(records: pd.DataFrame) -> pd.DataFrame:
    columns = ["Representation", "NMJE", "PCK@.10", "Domains averaged", "Result scope"]
    if records.empty:
        return pd.DataFrame(columns=columns)
    aggregate = aggregate_equal_domains(records, ("representation", "synthetic", "smoke"))
    return pd.DataFrame(
        {
            "Representation": aggregate["representation"],
            "NMJE": aggregate["nmje_mean"],
            "PCK@.10": aggregate.get("pck_at_0_10_mean", np.nan),
            "Domains averaged": aggregate["domains_averaged"],
            "Result scope": _scope_labels(
                aggregate,
                includes_drone=records["test_domain"].astype(str).str.lower().eq("drone").any(),
            ),
        }
    )


def _missingness_result_table(records: pd.DataFrame) -> pd.DataFrame:
    columns = ["Missing %", "NMJE", "PCK@.10", "Domains averaged", "Result scope"]
    if records.empty:
        return pd.DataFrame(columns=columns)
    aggregate = aggregate_equal_domains(records, ("mask_percentage", "synthetic", "smoke"))
    return pd.DataFrame(
        {
            "Missing %": aggregate["mask_percentage"],
            "NMJE": aggregate["nmje_mean"],
            "PCK@.10": aggregate.get("pck_at_0_10_mean", np.nan),
            "Domains averaged": aggregate["domains_averaged"],
            "Result scope": _scope_labels(
                aggregate,
                includes_drone=records["test_domain"].astype(str).str.lower().eq("drone").any(),
            ),
        }
    )


def _synthetic_suffix(table: pd.DataFrame) -> str:
    if table.empty or "synthetic" not in table:
        return ""
    if table["synthetic"].astype(bool).all():
        return "\n(synthetic software validation; not research results)"
    if table["synthetic"].astype(bool).any():
        return "\n(includes synthetic software validation; see tables)"
    if "smoke" in table and table["smoke"].astype(bool).all():
        return "\n(software smoke run; not a primary research result)"
    if "smoke" in table and table["smoke"].astype(bool).any():
        return "\n(includes software smoke runs; see tables)"
    return ""


def _remove_stale(path: Path) -> None:
    if path.is_file():
        path.unlink()


def _generate_figures(
    records: pd.DataFrame,
    figures_dir: Path,
    comparison_slices: Mapping[str, pd.DataFrame] | None = None,
) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    slices = dict(comparison_slices or _preferred_comparison_slices(records))

    model_slice = slices["model"]
    if not model_slice.empty:
        plotted = aggregate_equal_domains(model_slice, ("model", "synthetic", "smoke"))
        suffix = _synthetic_suffix(model_slice)
        path = figures_dir / "nmje_by_model.png"
        sequence_length = int(pd.to_numeric(model_slice["sequence_length"]).iloc[0])
        domain_names = ", ".join(sorted(model_slice["test_domain"].astype(str).unique()))
        figure, _ = plot_nmje_by_model(
            plotted,
            title=f"NMJE by model\n(T={sequence_length}; equal-domain mean: {domain_names}; error bars: domain SD)" + suffix,
            output_path=path,
        )
        plt.close(figure)
        if all(column in plotted for column in ("pck_at_0_05_mean", "pck_at_0_10_mean", "pck_at_0_20_mean")):
            figure, _ = plot_pck_curves(
                plotted,
                title=(
                    "PCK by model\n"
                    f"(T={sequence_length}; equal-domain mean: {domain_names})"
                    + suffix
                ),
                output_path=figures_dir / "pck_curves.png",
            )
            plt.close(figure)
        else:
            _remove_stale(figures_dir / "pck_curves.png")
            notes.append("pck_curves: controlled model comparison lacks all requested PCK metrics")
    else:
        for name in ("nmje_by_model.png", "pck_curves.png"):
            _remove_stale(figures_dir / name)
        notes.append("nmje_by_model and pck_curves: require >=2 models at identical representation/domain/mask/data settings")

    mask_slice = slices["missingness"]
    if not mask_slice.empty:
        plotted = aggregate_equal_domains(mask_slice, ("mask_percentage", "synthetic", "smoke"))
        path = figures_dir / "error_vs_missing_percentage.png"
        figure, axis = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
        axis.errorbar(plotted["mask_percentage"], plotted["nmje_mean"], yerr=plotted.get("nmje_std"), marker="o", capsize=3)
        axis.set(xlabel="Experimentally masked keypoints (%)", ylabel="NMJE (bbox diagonal)", title="Error vs missing percentage\n(MLP + normalized angles, T=1; error bars: domain SD)" + _synthetic_suffix(mask_slice))
        axis.grid(alpha=0.25)
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)
    else:
        _remove_stale(figures_dir / "error_vs_missing_percentage.png")
        notes.append("error_vs_missing_percentage: require >=2 mask ratios under fixed controls")

    representation_slice = slices["normalization_shift"]
    if not representation_slice.empty:
        plotted = aggregate_equal_domains(representation_slice, ("representation", "synthetic", "smoke"))
        display_order = {"raw": 0, "bbox_normalized": 1, "scale_normalized": 2}
        plotted = plotted.assign(
            _display_order=plotted["representation"].map(display_order).fillna(99)
        ).sort_values("_display_order")
        path = figures_dir / "normalization_ablation.png"
        figure, axis = plt.subplots(figsize=(7.5, 4.8))
        bars = axis.bar(plotted["representation"].astype(str), plotted["nmje_mean"], yerr=plotted.get("nmje_std"), capsize=3, color="#4477AA")
        axis.set_yscale("log")
        axis.bar_label(
            bars,
            labels=[f"{value:.4f}" for value in plotted["nmje_mean"]],
            padding=3,
            fontsize=8,
        )
        axis.set(ylabel="NMJE (bbox diagonal; log scale)", title="Normalization ablation: Indoor to Outdoor\n(MLP, T=1, 20% mask, single seed 42)" + _synthetic_suffix(representation_slice))
        figure.text(
            0.5,
            0.02,
            "Raw gap also reflects coordinate scale/optimization; not causal proof.",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#374151",
        )
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", which="both", alpha=0.25)
        figure.subplots_adjust(left=0.12, right=0.98, top=0.77, bottom=0.3)
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)
    else:
        _remove_stale(figures_dir / "normalization_ablation.png")
        notes.append("normalization_ablation: require >=2 representations under fixed controls")

    domain_slice = slices["domain"]
    if not domain_slice.empty:
        plotted = aggregate_present(domain_slice, ("test_domain", "synthetic", "smoke"))
        path = figures_dir / "indoor_outdoor_drone_performance.png"
        figure, axis = plt.subplots(figsize=(6.7, 4.4), constrained_layout=True)
        axis.bar(plotted["test_domain"].astype(str), plotted["nmje_mean"], yerr=plotted.get("nmje_std"), capsize=3, color="#228833")
        axis.set(xlabel="Evaluation domain", ylabel="NMJE (bbox diagonal)", title="Indoor vs Outdoor vs Drone performance\n(pooled MLP + normalized angles, T=1, 20% mask)" + _synthetic_suffix(domain_slice))
        axis.grid(axis="y", alpha=0.25)
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)
    else:
        _remove_stale(figures_dir / "indoor_outdoor_drone_performance.png")
        notes.append("indoor_outdoor_drone_performance: require >=2 within-domain results under fixed controls")

    cross_slice = slices["cross_domain"]
    if not cross_slice.empty and cross_slice["test_domain"].dropna().nunique() >= 2:
        plotted = aggregate_present(cross_slice, ("train_domain", "test_domain", "synthetic", "smoke"))
        figure, _ = plot_cross_domain_matrix(
            plotted,
            title="Cross-domain NMJE\n(MLP, bbox normalized, T=1, 20% mask)" + _synthetic_suffix(cross_slice),
            output_path=figures_dir / "cross_domain_matrix.png",
        )
        plt.close(figure)
    else:
        _remove_stale(figures_dir / "cross_domain_matrix.png")
        notes.append("cross_domain_matrix: require >=2 train and >=2 test domains under fixed controls")

    efficiency_slice = slices["data_efficiency"]
    if not efficiency_slice.empty:
        plotted = aggregate_equal_domains(efficiency_slice, ("data_percent", "synthetic", "smoke"))
        figure, _ = plot_reconstruction_data_efficiency(
            plotted,
            title="Data-efficiency curve\n(MLP + normalized angles, T=1, 20% mask; equal-domain mean)" + _synthetic_suffix(efficiency_slice),
            output_path=figures_dir / "data_efficiency_curve.png",
        )
        plt.close(figure)
    else:
        _remove_stale(figures_dir / "data_efficiency_curve.png")
        notes.append("data_efficiency_curve: require >=2 executed group-subset percentages under fixed controls")
    return notes


def run_ablation(
    logs_root: str | Path,
    tables_dir: str | Path,
    figures_dir: str | Path,
    *,
    data_path: str | Path | None = None,
    include_synthetic: bool = False,
    include_smoke: bool = False,
) -> dict[str, Path]:
    """Create evidence-backed reconstruction tables, figures, and an audit index."""

    logs = Path(logs_root)
    tables = Path(tables_dir)
    figures = Path(figures_dir)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    records = discover_reconstruction_records(
        logs, include_synthetic=include_synthetic, include_smoke=include_smoke
    )
    comparison_slices = _preferred_comparison_slices(records)
    processed = _load_processed_arrays(data_path)
    subset, missing = dataset_tables(processed)
    result_tables = _aggregate_result_tables(records)
    representations = pd.DataFrame(REPRESENTATION_DEFINITIONS)
    executed = set(records["representation"].dropna().astype(str)) if not records.empty else set()
    representations["executed_evidence"] = representations["representation"].isin(executed)
    tables_by_name: dict[str, pd.DataFrame] = {
        "dataset_subset_statistics": subset,
        "representation_definitions": representations,
        "representation": _representation_result_table(comparison_slices["representation"]),
        "normalization_shift": _representation_result_table(
            comparison_slices["normalization_shift"]
        ),
        "model_parameters_compute": _model_table(comparison_slices["model"]),
        "missing_keypoints": _missingness_result_table(comparison_slices["missingness"]),
        "natural_missingness_by_joint": missing,
        **result_tables,
    }
    paths: dict[str, Path] = {}
    for name, table in tables_by_name.items():
        stem = tables / name
        _write_table(table, stem)
        paths[f"{name}_csv"] = stem.with_suffix(".csv")
        paths[f"{name}_md"] = stem.with_suffix(".md")
    notes = _generate_figures(records, figures, comparison_slices)
    legacy_count = sum(
        1
        for path in logs.rglob("metrics.json")
        if (payload := _load_json(path)) is not None and _task(payload) not in RECONSTRUCTION_TASKS
    ) if logs.exists() else 0
    pre_audit_drone_count = sum(
        1
        for path in logs.rglob("metrics.json")
        if (payload := _load_json(path)) is not None
        and _task(payload) in RECONSTRUCTION_TASKS
        and _canonical_domain(
            _first_present(
                payload.get("test_domain"),
                _nested(payload, "experiment.test_domain"),
                _nested(payload, "protocol.test_domain"),
            )
        )
        == "Drone"
        and payload.get("input_transform_audit_passed") is not True
    ) if logs.exists() else 0
    index = {
        "artifact_type": "reconstruction_ablation_index",
        "task": "keypoint_reconstruction",
        "title": TITLE,
        "include_synthetic": include_synthetic,
        "include_smoke": include_smoke,
        "synthetic_label": "software validation only / NOT research results",
        "records_included": len(records),
        "sources": records.get("source", pd.Series(dtype=str)).tolist(),
        "legacy_non_reconstruction_artifacts_excluded": legacy_count,
        "pre_observed_transform_drone_artifacts_excluded": pre_audit_drone_count,
        "comparison_protocols": _comparison_audit(comparison_slices),
        "tables": {name: {"rows": len(table), "csv": str((tables / name).with_suffix('.csv')), "markdown": str((tables / name).with_suffix('.md'))} for name, table in tables_by_name.items()},
        "not_generated": notes,
        "rgb_overlay": "NOT GENERATED: no source RGB frames are present in the annotation-only acquisition; remote media size was not assumed.",
    }
    index_path = tables / "ablation_index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    paths["index"] = index_path
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, default=Path("outputs/logs"))
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/trackid3x3/dataset.npz")
    )
    parser.add_argument("--tables-dir", type=Path, default=Path("outputs/tables/reconstruction"))
    parser.add_argument("--figures-dir", type=Path, default=Path("outputs/figures/reconstruction"))
    parser.add_argument("--include-synthetic", action="store_true")
    parser.add_argument("--include-smoke", action="store_true", help="Include real but non-primary smoke runs")
    return parser.parse_args(argv)


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = run_ablation(
        _project_path(args.logs_root),
        _project_path(args.tables_dir),
        _project_path(args.figures_dir),
        data_path=_project_path(args.data),
        include_synthetic=args.include_synthetic,
        include_smoke=args.include_smoke,
    )
    print(f"Wrote reconstruction ablation index: {paths['index']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
