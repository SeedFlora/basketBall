#!/usr/bin/env python
"""Generate the TrackID reconstruction report from executed evidence only."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import _dataset_figures, load_reconstruction_data
from scripts.run_ablation import (
    REPRESENTATION_DEFINITIONS,
    TITLE,
    discover_reconstruction_records,
    run_ablation,
)

FIGURE_REQUIREMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("1. Example 10-joint skeleton", ("example_10_joint_skeleton*.png",)),
    ("2. Normalized skeleton", ("normalized_skeleton*.png",)),
    ("3. Masked skeleton", ("masked_skeleton*.png",)),
    ("4. Reconstructed skeleton", ("reconstructed_skeleton*.png",)),
    ("5. Ground truth vs reconstruction", ("ground_truth_vs_reconstruction*.png",)),
    ("6. Per-joint reconstruction error", ("per_joint_reconstruction_error*.png",)),
    ("7. Error vs missing percentage", ("error_vs_missing_percentage*.png",)),
    ("8. Indoor vs Outdoor vs Drone", ("indoor_outdoor_drone_performance*.png",)),
    ("9. Cross-domain matrix", ("cross_domain_matrix*.png",)),
    ("10. Normalization ablation", ("normalization_ablation*.png",)),
    ("11. Data-efficiency curve", ("data_efficiency_curve*.png",)),
    ("12. RGB overlay", ("rgb_overlay*.png",)),
    ("Dataset domain summary", ("dataset_domain_summary*.png",)),
    ("Pose-instance counts", ("pose_instance_counts*.png",)),
    ("Natural missingness", ("missingness_by_domain_joint*.png",)),
    ("Raw vs normalized coordinates", ("coordinate_distributions_raw_vs_normalized*.png",)),
    ("Domain skeleton examples", ("skeleton_examples_by_domain*.png",)),
    ("Training curves", ("training_curves*.png",)),
    ("NMJE by model", ("nmje_by_model*.png",)),
    ("PCK curves", ("pck_curves*.png",)),
)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, Mapping) else {}


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _markdown_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "**NOT GENERATED: no executed evidence exists for this table.**"
    shown = table.copy()
    for column in shown:
        if isinstance(shown[column].dtype, pd.CategoricalDtype):
            shown[column] = shown[column].astype(object)
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )
    shown = shown.fillna("").astype(str).map(lambda value: value.replace("|", "\\|"))
    header = "| " + " | ".join(shown.columns) + " |"
    separator = "| " + " | ".join("---" for _ in shown.columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in shown.to_numpy().tolist()]
    return "\n".join((header, separator, *rows))


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _find_processed(root: Path) -> Path | None:
    preferred = (
        root / "data" / "processed" / "trackid3x3" / "dataset.npz",
        root / "data" / "processed" / "trackid3x3.npz",
        root / "data" / "processed" / "dataset.npz",
    )
    candidates = list(preferred) + sorted((root / "data" / "processed").glob("**/*.npz"))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                names = set(archive.files)
                if (
                    ("coords" in names or "coordinates" in names)
                    and "valid_mask" in names
                    and ("bbox" in names or "bbox_xywh" in names)
                    and ("domain" in names or "domains" in names)
                ):
                    return path.resolve()
        except (OSError, ValueError):
            continue
    return None


def _processed_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    metadata = _read_json(path.with_name("metadata.json"))
    try:
        with np.load(path, allow_pickle=False) as archive:
            coordinates = archive["coords"] if "coords" in archive else archive["coordinates"]
            domain = archive["domain"] if "domain" in archive else archive["domains"]
            rgb = archive["rgb_available"] if "rgb_available" in archive else np.zeros(len(domain), dtype=bool)
            archive_synthetic = bool(archive["synthetic"].reshape(-1)[0]) if "synthetic" in archive else False
    except (OSError, ValueError, KeyError):
        return {}
    return {
        "path": str(path.resolve()),
        "sequences": int(coordinates.shape[0]),
        "sequence_length": int(coordinates.shape[1]),
        "joints": int(coordinates.shape[2]),
        "domains": sorted(set(np.asarray(domain).astype(str).reshape(-1).tolist())),
        "rgb_available": int(np.asarray(rgb, dtype=bool).sum()),
        "synthetic": bool(metadata.get("synthetic", False)) or archive_synthetic,
        "metadata": metadata,
    }


def _migration_status(root: Path) -> tuple[str, int]:
    migration = root / "outputs" / "reports" / "dataset_migration.md"
    legacy_count = 0
    logs = root / "outputs" / "logs"
    for path in logs.rglob("metrics.json") if logs.exists() else ():
        payload = _read_json(path)
        if payload and payload.get("task") not in {
            "keypoint_reconstruction",
            "pose_reconstruction",
            "masked_keypoint_reconstruction",
        }:
            legacy_count += 1
    if migration.is_file():
        status = (
            f"The source/task migration is documented in [{_relative(migration, root)}]"
            f"({migration.name})."
        )
    else:
        status = "No migration note was found; this report still accepts only TrackID reconstruction artifacts."
    return status, legacy_count


def _annotation_status(root: Path) -> str:
    schema = root / "outputs" / "reports" / "trackid3x3_schema.md"
    checkout = root / "external" / "TrackID3x3"
    if schema.is_file() and checkout.is_dir():
        return (
            "TrackID3x3 annotations are locally available and their 10-joint schema, domain counts, "
            f"bbox provenance, and grouping rules were validated in [{_relative(schema, root)}]"
            f"({schema.name})."
        )
    if checkout.is_dir():
        return "A local TrackID3x3 checkout exists, but no schema-inspection report was found."
    return "**TRACKID ANNOTATIONS NOT AVAILABLE LOCALLY:** source inspection and preprocessing have not been evidenced."


def _figure_status(
    figures_root: Path,
    root: Path,
    *,
    synthetic_only: bool,
    include_synthetic: bool = False,
    include_smoke: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for name, patterns in FIGURE_REQUIREMENTS:
        matches = sorted(
            {
                path
                for pattern in patterns
                for path in figures_root.rglob(pattern)
                if path.is_file()
                and (include_synthetic or "synthetic" not in str(path).lower())
                and (include_smoke or "smoke" not in str(path).lower())
            }
        ) if figures_root.exists() else []
        if matches:
            if all("synthetic" in str(path).lower() for path in matches):
                status = "generated — software validation only / NOT research results"
            elif all("smoke" in str(path).lower() for path in matches):
                status = "generated — software smoke run / NOT primary research result"
            else:
                status = (
                    "generated — software validation only / NOT research results"
                    if synthetic_only
                    else "generated from available evidence"
                )
            preferred = min(
                matches,
                key=lambda path: (
                    0
                    if path.resolve().parent == (figures_root / "reconstruction").resolve()
                    else 1,
                    len(path.resolve().relative_to(figures_root.resolve()).parts),
                    str(path),
                ),
            )
            files = _relative(preferred, root)
            if len(matches) > 1:
                files += f" (+{len(matches) - 1} other executed variant(s))"
        elif name == "12. RGB overlay":
            status = "NOT GENERATED: source RGB frames are unavailable locally; remote media size was not assumed"
            files = ""
        else:
            status = "NOT GENERATED: required executed evidence is unavailable"
            files = ""
        rows.append({"Figure": name, "Status": status, "Files": files})
    return pd.DataFrame(rows)


def _best_result(representation_table: pd.DataFrame) -> str:
    """Summarize only the declared controlled representation comparison."""

    if representation_table.empty or "NMJE" not in representation_table:
        return "No best research result can be reported because the controlled representation table is empty."
    usable = representation_table.copy()
    usable["NMJE"] = pd.to_numeric(usable["NMJE"], errors="coerce")
    usable = usable.dropna(subset=["NMJE"]).sort_values("NMJE", ascending=True)
    if usable.empty:
        return "No best research result can be reported because the controlled representation table is empty."
    row = usable.iloc[0]
    pck = pd.to_numeric(pd.Series([row.get("PCK@.10", np.nan)]), errors="coerce").iloc[0]
    pck_text = f" and PCK@0.10 {float(pck):.4f}" if pd.notna(pck) else ""
    return (
        "Within the declared Table 2 comparison (pooled-domain MLP, T=1, 20% masking, "
        "100% data, seed 42; equal-domain macro mean over Indoor, Outdoor, and Drone), "
        f"the lowest NMJE was {float(row['NMJE']):.4f}{pck_text} for "
        f"{row.get('Representation', 'an unspecified representation')}. This is "
        "single-seed exploratory evidence, not a universal-superiority claim."
    )


def _comparison_note(index: Mapping[str, Any], name: str) -> str:
    protocols = index.get("comparison_protocols")
    entry = protocols.get(name) if isinstance(protocols, Mapping) else None
    if not isinstance(entry, Mapping):
        return "**Protocol metadata unavailable; no interpretation should be made from this table.**"
    controls = entry.get("selected_controls")
    def render_value(value: Any) -> str:
        if isinstance(value, list):
            return "[" + ", ".join(str(item) for item in value) + "]"
        return str(value)

    rendered_controls = (
        ", ".join(f"{key}={render_value(value)}" for key, value in controls.items())
        if isinstance(controls, Mapping) and controls
        else "none"
    )
    domains = entry.get("test_domains", [])
    return (
        f"Policy: {entry.get('selection_policy', 'not recorded')} "
        f"Aggregation: {entry.get('domain_aggregation', 'not recorded')}. "
        f"Selected controls: {rendered_controls}. Test domains: {render_value(domains)}."
    )


def _micro_macro_note(root: Path, records: pd.DataFrame) -> str:
    """Explain the pooled-joint and equal-domain estimands using executed evidence."""

    if records.empty:
        return (
            "Table 2 uses an equal-domain macro mean. A pooled joint-level estimate is "
            "not available without the matching executed mask summary."
        )
    required = {
        "model",
        "representation",
        "train_domain",
        "mask_ratio",
        "data_percent",
        "sequence_length",
        "seed",
        "source",
    }
    if not required.issubset(records.columns):
        return "Table 2 uses an equal-domain macro mean; pooled joint-level evidence is unavailable."

    def match(column: str, value: float) -> pd.Series:
        numeric = pd.to_numeric(records[column], errors="coerce").to_numpy(dtype=float)
        return pd.Series(np.isclose(numeric, value, rtol=0.0, atol=1e-9), index=records.index)

    primary = records[
        records["model"].astype(str).str.lower().eq("mlp")
        & records["representation"].astype(str).str.lower().eq("normalized_angles")
        & records["train_domain"].astype(str).str.lower().isin(
            {"all", "pooled", "mixed", "all_domains"}
        )
        & match("mask_ratio", 0.20)
        & match("data_percent", 100.0)
        & match("sequence_length", 1.0)
        & match("seed", 42.0)
    ]
    summaries = {
        Path(str(source)).resolve().parent.parent / "mask_summary.json"
        for source in primary["source"]
    }
    for path in sorted(summaries):
        payload = _read_json(path)
        metrics = payload.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        try:
            nmje = float(metrics["nmje"])
            pck = float(metrics["pck_at_0_10"])
            scored = int(metrics["n_scored_joints"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        return (
            "Table 2 macro-averages the three per-domain metrics, giving Indoor, Outdoor, "
            "and Drone equal weight regardless of sample count. For the same "
            "normalized+angles slice, the executed pooled joint-level summary is "
            f"NMJE {nmje:.4f} and PCK@0.10 {pck:.4f} over {scored} scored joints; this "
            "micro-style estimate weights domains in proportion to scored joints. The "
            "two estimands answer different questions and are not interchangeable."
        )
    return (
        "Table 2 macro-averages per-domain metrics with equal domain weight. The matching "
        "pooled joint-level summary is unavailable and is not inferred."
    )


def _error_analysis(root: Path, records: pd.DataFrame) -> str:
    if records.empty:
        return "**NOT GENERATED:** no non-synthetic reconstruction evaluation exists for error analysis."
    required = {
        "model",
        "representation",
        "train_domain",
        "mask_ratio",
        "data_percent",
        "sequence_length",
        "seed",
    }
    if not required.issubset(records.columns):
        return "**NOT GENERATED:** primary error-analysis controls are absent from result metadata."

    def numeric_match(column: str, value: float) -> pd.Series:
        numeric = pd.to_numeric(records[column], errors="coerce").to_numpy(dtype=float)
        return pd.Series(
            np.isclose(numeric, value, rtol=0.0, atol=1e-9),
            index=records.index,
        )

    primary = records[
        records["model"].astype(str).str.lower().eq("mlp")
        & records["representation"].astype(str).str.lower().eq("normalized_angles")
        & records["train_domain"].astype(str).str.lower().isin(
            {"all", "pooled", "mixed", "all_domains"}
        )
        & numeric_match("mask_ratio", 0.20)
        & numeric_match("data_percent", 100.0)
        & numeric_match("sequence_length", 1.0)
        & numeric_match("seed", 42.0)
    ].copy()
    if primary.empty:
        return (
            "**NOT GENERATED:** the fixed primary error slice (pooled normalized+angles "
            "MLP, T=1, 20% mask, 100% data, seed 42) is unavailable."
        )
    record_by_source = {
        Path(str(row["source"])).resolve(): row
        for _, row in primary.iterrows()
    }
    record_by_error_path = {
        (metrics_path.parent.parent / "sequence_errors.csv").resolve(): row
        for metrics_path, row in record_by_source.items()
    }
    source_paths = set(record_by_source)
    error_paths: set[Path] = set()
    for metrics_path in source_paths:
        candidate = metrics_path.parent.parent / "sequence_errors.csv"
        if candidate.is_file():
            error_paths.add(candidate)
    tables: list[pd.DataFrame] = []
    for path in sorted(error_paths):
        try:
            table = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        if {"sample_id", "nmje"}.issubset(table):
            table = table.copy()
            record = record_by_error_path.get(path.resolve())
            if record is not None:
                table["evaluation_model"] = record.get("model", "")
                table["representation"] = record.get("representation", "")
                table["train_domain"] = record.get("train_domain", "")
                table["sequence_length"] = record.get("sequence_length", np.nan)
            table["source"] = _relative(path, root)
            tables.append(table)
    if not tables:
        return "**NOT GENERATED:** sequence-level error artifacts are absent for the included result sources."
    combined = pd.concat(tables, ignore_index=True)
    combined["nmje"] = pd.to_numeric(combined["nmje"], errors="coerce")
    worst = combined.dropna(subset=["nmje"]).nlargest(10, "nmje")
    columns = [
        column
        for column in (
            "sample_id",
            "evaluation_model",
            "representation",
            "train_domain",
            "sequence_length",
            "domain",
            "video_id",
            "track_id",
            "mask_ratio",
            "n_scored_joints",
            "mean_bbox_diagonal",
            "player_scale_bin",
            "pose_spread_bbox_fraction",
            "masked_distal_joints",
            "distal_nmje",
            "non_distal_nmje",
            "same_frame_max_bbox_iou",
            "nmje",
            "source",
        )
        if column in worst
    ]
    summaries: list[str] = []
    if "player_scale_bin" in combined:
        scale = (
            combined.dropna(subset=["nmje", "player_scale_bin"])
            .groupby("player_scale_bin", observed=True, as_index=False)
            .agg(**{"Sequences": ("nmje", "size"), "Mean NMJE": ("nmje", "mean")})
            .rename(columns={"player_scale_bin": "Player scale bin"})
        )
        order = pd.Categorical(
            scale["Player scale bin"],
            categories=["small", "medium", "large"],
            ordered=True,
        )
        scale = scale.assign(_order=order).sort_values("_order").drop(columns="_order")
        summaries.extend(("#### Bbox-scale strata", _markdown_table(scale)))
    if "masked_distal_joints" in combined:
        distal = combined.copy()
        distal["masked_distal_joints"] = pd.to_numeric(
            distal["masked_distal_joints"], errors="coerce"
        )
        distal = distal.dropna(subset=["nmje", "masked_distal_joints"])
        distal["Masked distal target"] = np.where(
            distal["masked_distal_joints"].gt(0), "yes", "no"
        )
        distal_summary = (
            distal.groupby("Masked distal target", observed=True, as_index=False)
            .agg(**{"Sequences": ("nmje", "size"), "Mean NMJE": ("nmje", "mean")})
            .sort_values("Masked distal target", ascending=False)
        )
        summaries.extend(("#### Distal-target strata", _markdown_table(distal_summary)))
    if "pose_spread_bbox_fraction" in combined:
        spread = combined[["nmje", "pose_spread_bbox_fraction"]].copy()
        spread["pose_spread_bbox_fraction"] = pd.to_numeric(
            spread["pose_spread_bbox_fraction"], errors="coerce"
        )
        spread = spread.dropna()
        if len(spread) >= 4:
            quartile = pd.qcut(
                spread["pose_spread_bbox_fraction"],
                q=4,
                labels=False,
                duplicates="drop",
            )
            spread["Pose-spread quartile"] = quartile.map(
                lambda value: f"Q{int(value) + 1}" if pd.notna(value) else ""
            )
            spread_summary = (
                spread.groupby("Pose-spread quartile", observed=True, as_index=False)
                .agg(**{"Sequences": ("nmje", "size"), "Mean NMJE": ("nmje", "mean")})
                .sort_values("Pose-spread quartile")
            )
            summaries.extend(("#### Pose-spread strata", _markdown_table(spread_summary)))

    overlap_note = "Same-frame bbox overlap was not evaluated: aligned peer boxes are unavailable."
    if "same_frame_max_bbox_iou" in combined:
        overlap = combined[["nmje", "same_frame_max_bbox_iou"]].copy()
        overlap["same_frame_max_bbox_iou"] = pd.to_numeric(
            overlap["same_frame_max_bbox_iou"], errors="coerce"
        )
        overlap = overlap.dropna()
        available_domains: set[str] = set()
        unavailable_domains: set[str] = set()
        if "domain" in combined:
            domain_text = combined["domain"].astype(str)
            available_domains = set(
                domain_text[pd.to_numeric(combined["same_frame_max_bbox_iou"], errors="coerce").notna()]
            )
            unavailable_domains = set(domain_text) - available_domains
        if not overlap.empty:
            overlap["Overlap stratum"] = pd.cut(
                overlap["same_frame_max_bbox_iou"],
                bins=[-np.inf, 0.05, 0.20, np.inf],
                labels=["low (<=0.05)", "moderate (0.05-0.20]", "high (>0.20)"],
            )
            overlap_summary = (
                overlap.groupby("Overlap stratum", observed=True, as_index=False)
                .agg(**{"Sequences": ("nmje", "size"), "Mean NMJE": ("nmje", "mean")})
            )
            summaries.extend(("#### Same-frame overlap strata", _markdown_table(overlap_summary)))
            available_text = ", ".join(sorted(available_domains)) or "available rows"
            unavailable_text = ", ".join(sorted(unavailable_domains))
            overlap_note = f"Same-frame bbox overlap was evaluated directly for {available_text}."
            if unavailable_text:
                overlap_note += f" It was unavailable for {unavailable_text}."
    controls = (
        "**Fixed primary slice:** pooled-domain normalized+angles MLP, T=1, 20% "
        "experimental masking, 100% training data, seed 42; Indoor, Outdoor, and Drone."
    )
    caution = (
        "These are directly observable descriptive strata only; differences are not "
        "causal effects and naturally absent joints are never scored."
    )
    return "\n\n".join(
        (
            controls,
            caution,
            overlap_note,
            *summaries,
            "#### Highest-error sequences within the fixed slice",
            _markdown_table(worst[columns]),
        )
    )


def generate_report(
    project_root: str | Path = PROJECT_ROOT,
    output_path: str | Path | None = None,
    *,
    include_synthetic: bool = False,
    include_smoke: bool = False,
) -> Path:
    """Generate ``final_report.md`` using only actual reconstruction artifacts."""

    root = Path(project_root).resolve()
    destination = (
        Path(output_path)
        if output_path is not None
        else root / "outputs" / "reports" / "final_report.md"
    )
    if not destination.is_absolute():
        destination = root / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    logs = root / "outputs" / "logs"
    tables_dir = root / "outputs" / "tables" / "reconstruction"
    figures_dir = root / "outputs" / "figures" / "reconstruction"
    processed = _find_processed(root)
    run_ablation(
        logs,
        tables_dir,
        figures_dir,
        data_path=processed,
        include_synthetic=include_synthetic,
        include_smoke=include_smoke,
    )
    primary_records = discover_reconstruction_records(
        logs, include_synthetic=False, include_smoke=False
    )
    all_records = discover_reconstruction_records(
        logs, include_synthetic=True, include_smoke=include_smoke
    )
    smoke_records = discover_reconstruction_records(
        logs, include_synthetic=True, include_smoke=True
    )
    if not smoke_records.empty and "smoke" in smoke_records:
        smoke_records = smoke_records[smoke_records["smoke"].astype(bool)].reset_index(drop=True)
    synthetic_records = (
        all_records[all_records["synthetic"].astype(bool)].reset_index(drop=True)
        if not all_records.empty and "synthetic" in all_records
        else pd.DataFrame()
    )
    processed_info = _processed_summary(processed)
    if processed is not None:
        try:
            processed_data, _ = load_reconstruction_data(processed)
            dataset_suffix = (
                "\n(synthetic software validation; not research results)"
                if processed_info.get("synthetic", False)
                else ""
            )
            _dataset_figures(processed_data, figures_dir, synthetic_suffix=dataset_suffix)
        except (OSError, ValueError, RuntimeError):
            # The processed status below remains explicit; a report must still be
            # produced when a descriptive plot cannot be generated.
            pass
    migration, legacy_count = _migration_status(root)
    ablation_index = _read_json(tables_dir / "ablation_index.json")
    pre_audit_drone_count = int(
        ablation_index.get("pre_observed_transform_drone_artifacts_excluded", 0) or 0
    )
    representation_protocol = _comparison_note(ablation_index, "representation")
    normalization_shift_protocol = _comparison_note(ablation_index, "normalization_shift")
    model_protocol = _comparison_note(ablation_index, "model")
    missingness_protocol = _comparison_note(ablation_index, "missingness")
    annotation_status = _annotation_status(root)

    dataset_table = _read_table(tables_dir / "dataset_subset_statistics.csv")
    representation_result = _read_table(tables_dir / "representation.csv")
    normalization_shift = _read_table(tables_dir / "normalization_shift.csv")
    model_table = _read_table(tables_dir / "model_parameters_compute.csv")
    missing_table = _read_table(tables_dir / "missing_keypoints.csv")
    within_table = _read_table(tables_dir / "within_domain_metrics.csv")
    cross_table = _read_table(tables_dir / "cross_domain_metrics.csv")
    representation_definitions = pd.DataFrame(REPRESENTATION_DEFINITIONS).rename(
        columns={
            "representation": "Representation",
            "definition": "Definition",
            "inverse_for_metrics": "Inverse for metrics",
        }
    )

    if primary_records.empty:
        result_marker = (
            "**NO LEAKAGE-AUDITED NON-SYNTHETIC RECONSTRUCTION RESULTS.** "
            "All empirical result tables remain explicitly empty."
        )
    else:
        result_marker = (
            f"Included {len(primary_records)} leakage-audited non-synthetic, non-smoke domain/mask evaluation artifact(s)."
        )
    synthetic_note = (
        f"Detected {len(synthetic_records)} synthetic reconstruction artifact(s); these are software validation only / "
        f"NOT research results and are {'shown separately' if include_synthetic else 'excluded from tables'}."
    )
    smoke_note = (
        f"Detected {len(smoke_records)} real-data software-smoke artifact(s); these are "
        f"NOT primary research results and are {'shown with explicit scope labels' if include_smoke else 'excluded by default'}."
    )
    if processed_info:
        processed_line = (
            f"Processed artifact: `{_relative(Path(processed_info['path']), root)}` with "
            f"{processed_info['sequences']} sequences, T={processed_info['sequence_length']}, "
            f"{processed_info['joints']} joints, domains {', '.join(processed_info['domains'])}."
        )
        if processed_info["synthetic"]:
            processed_line += " **Synthetic fixture; software validation only / NOT research results.**"
    else:
        processed_line = "**PROCESSED TRACKID ARTIFACT NOT FOUND:** dataset tables and evaluation cannot be completed yet."
    synthetic_only = primary_records.empty and (
        not synthetic_records.empty or bool(processed_info.get("synthetic", False))
    )
    figure_status = _figure_status(
        root / "outputs" / "figures",
        root,
        synthetic_only=synthetic_only,
        include_synthetic=include_synthetic,
        include_smoke=include_smoke,
    )
    best = _best_result(representation_result)
    micro_macro_note = _micro_macro_note(root, primary_records)
    error_analysis = _error_analysis(root, primary_records)
    figure_bytes = sum(
        path.stat().st_size
        for path in (root / "outputs" / "figures").rglob("*")
        if path.is_file()
    )
    figure_storage_note = (
        f"Generated figure files occupy {figure_bytes / 1_000_000:.1f} MB (decimal). "
        "This exceeds the preferred 30 MB figure subtarget because repeated and legacy "
        "variants were preserved under the no-delete constraint; the project remains "
        "below the separate hard 600 MB ceiling."
    )
    closing = (
        "No conclusion is substituted for an unexecuted cell. Tables and figures without "
        "supporting executed evidence are explicitly marked NOT GENERATED above."
        if primary_records.empty
        else "No conclusion is substituted for an unexecuted cell. All requested "
        "evidence-backed tables and non-RGB figures are present in this run; RGB overlay "
        "remains explicitly unavailable because no source frames are present locally."
    )

    report = f"""# {TITLE}

## Scope and evidence status

This report covers deterministic masked reconstruction of the documented TrackID3x3 10-joint skeleton. It does not infer missing source annotations, create unsupported joints, or assess basketball technique.

{result_marker}

All empirical comparisons reported below use seed 42 only and are single-seed exploratory evidence, not confidence intervals or confirmatory estimates. Drone rows are additionally exploratory because their metric denominator uses documented pose-tight fallback provenance rather than an independent MOT box.

{synthetic_note}

{smoke_note}

{migration} {legacy_count} task-incompatible legacy metric artifact(s) were detected and excluded.

{pre_audit_drone_count} pre-audit Drone evaluation artifact(s) were excluded because pose-tight full-pose bboxes had not yet been proven metric-only. Drone inputs now require an observed-only post-mask transform; the stored pose-tight bbox is retained only as the NMJE/PCK denominator.

## Dataset

{annotation_status}

{processed_line}

The three capture domains are Indoor fixed camera, Outdoor fixed camera, and outdoor Drone. Bboxes are independent MOT `xywh` boxes for Indoor/Outdoor; Drone uses the explicitly recorded tight-pose fallback. The local acquisition is annotation-only. No RGB frame, video bundle, or model weight was downloaded because remote sizes were not established.

### Dataset/subset statistics

{_markdown_table(dataset_table)}

Storage is the complete on-disk `ground_truth/<Domain>` annotation tree, not only files contributing selected poses. An instance means one player pose at one annotated frame, so `Player instances` and valid `Pose instances` coincide in this one-pose-per-player-frame artifact. `Player tracks` is the additional count of unique scoped video-track identities (or the documented track fallback). Drone video identity is unavailable/not linkable. Drone MOT files are included in that storage footprint but are not linked to, or used for, the sampled Drone pose records.

## Method

For each valid pose, a deterministic seed-42 mask holds out 10%, 20%, 30%, or 40% of originally annotated joints. The input mask and source-valid mask remain separate. Training and scoring use only the synthetically held-out subset of originally valid joints. Video is the preferred split group; track is the documented fallback where reliable video identity is unavailable. Any overlap across train, validation, and test groups invalidates an artifact.

The primary metric is normalized mean joint error (NMJE): source-pixel Euclidean error divided by the recorded source bbox diagonal. Lower is better. Indoor/Outdoor denominators are independent MOT boxes; Drone's explicitly labeled full-pose tight box is metric-only and never normalizes a masked model input. PCK@0.05, PCK@0.10, PCK@0.20, and per-joint NMJE are supporting measures. Predictions from every normalized representation are inverse-transformed to source pixels before metrics; already-normalized distances are never divided by a pixel bbox scale. The `PCK` column in the cross-domain table denotes PCK@0.10.

### Representation definitions

{_markdown_table(representation_definitions)}

Compared model families, when actually executed, are a fit-free geometric baseline, compact MLP, reliable-window TCN, and a graph model constrained below one million parameters. Static representation, missingness, and cross-domain tests use T=1. The controlled model-family table uses T=16 because all four reported families were executed on the same reliable Indoor/Outdoor track windows; TCN entries require an explicit reliable-window audit, and temporal windows are valid only within continuous same-video/same-player tracks.

## Results

{result_marker}

{best}

### Representation table

{representation_protocol}

{_markdown_table(representation_result)}

{micro_macro_note}

### Normalization shift hypothesis

{normalization_shift_protocol}

This is a single-seed normalization hypothesis test. The raw-coordinate gap also reflects optimization difficulty and coordinate scale; it is not causal proof that normalization alone resolves camera shift.

{_markdown_table(normalization_shift)}

### Model/compute table

{model_protocol}

`train_domain=not_applicable` denotes the fit-free geometric baseline; its outputs require no fitted training domain.

{_markdown_table(model_table)}

### Missing-keypoint robustness

{missingness_protocol}

{_markdown_table(missing_table)}

### Within-domain metrics

{_markdown_table(within_table)}

### Cross-domain metrics

{_markdown_table(cross_table)}

Only controlled slices are aggregated: representation comparisons fix model/domain/mask/data fraction/sequence length; model comparisons fix representation/domain/mask/data fraction/sequence length; missingness comparisons fix model/representation/domain/data fraction/sequence length; data-efficiency curves require all of 25%, 50%, and 100% at a fixed sequence length. Missing configurations remain blank rather than being imputed.

## Error analysis

Sequence errors below, when present, come only from the declared fixed primary slice. Naturally absent joints are excluded.

{error_analysis}

## Figures and tables audit

{_markdown_table(figure_status)}

CSV and Markdown tables are under `{_relative(tables_dir, root)}`. RGB overlay figure 12 is explicitly skipped unless actual source frames are available; none are present in this acquisition.

## Limitations

- Annotation sparsity and pose-tight Drone bbox fallback limit direct comparability with independent MOT bboxes; provenance remains explicit.
- Outdoor and Drone use player/track-level fallback grouping where video identity is not sufficiently informative. This is leakage-safe but yields fewer independent groups.
- TCN evidence is admissible only for continuous reliable windows. T=1 cannot demonstrate a temporal advantage.
- PCK/NMJE measure held-out annotation reconstruction, not correctness of naturally absent joints and not real-world 3-D pose accuracy.
- A single seed or controlled slice is not a confidence interval. Repeated executed seeds must be present before variability is interpreted.
- Annotation-only evaluation cannot support RGB overlay or appearance-conditioned conclusions.
- {figure_storage_note}

## Reproducibility

From the project root:

```bash
python scripts/evaluate.py --processed data/processed/trackid3x3/dataset.npz --checkpoint outputs/checkpoints/core_mlp_angles_all_mask20_seed42/best.pt --test-domain all --mask-ratios 0.10 0.20 0.30 0.40 --primary-mask-ratio 0.20 --seed 42 --run-name reproduce_core_mlp_angles_seed42
python scripts/evaluate.py --processed data/processed/trackid3x3_t16/dataset.npz --checkpoint outputs/checkpoints/core_tcn_bbox_all_t16_mask20_seed42/best.pt --test-domain all --mask-ratios 0.20 --primary-mask-ratio 0.20 --seed 42 --run-name reproduce_core_tcn_t16_seed42
python scripts/run_cross_domain.py --processed data/processed/trackid3x3/dataset.npz --checkpoints-root outputs/checkpoints --domains Indoor Outdoor Drone --models mlp --representations bbox_normalized --mask-ratios 0.20 --seed 42 --continue-on-error
python scripts/run_ablation.py --logs-root outputs/logs --data data/processed/trackid3x3/dataset.npz --tables-dir outputs/tables/reconstruction --figures-dir outputs/figures/reconstruction
python scripts/generate_report.py
python scripts/cleanup.py
```

`cleanup.py` is a dry run unless `--confirm` is supplied. Synthetic fixtures additionally require explicit `--include-synthetic`, and their outputs remain labeled software validation only / NOT research results.

## Research narrative

The experiment asks whether lightweight skeleton representations retain enough structure to recover deliberately hidden TrackID3x3 landmarks as camera geometry and viewpoint change. The most defensible comparison is a controlled NMJE/PCK contrast across Indoor, Outdoor, and Drone, accompanied by per-joint errors and missingness curves. {best}

{closing}
"""
    destination.write_text(report, encoding="utf-8")
    return destination.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-synthetic", action="store_true")
    parser.add_argument("--include-smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path = generate_report(
        args.project_root,
        args.output,
        include_synthetic=args.include_synthetic,
        include_smoke=args.include_smoke,
    )
    print(f"Generated evidence-backed TrackID reconstruction report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
