"""Publication figures for masked 2-D keypoint reconstruction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(figure: plt.Figure, output_path: str | Path | None, dpi: int = 300) -> None:
    if output_path is None:
        return
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")


def _pose(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[-1] != 2:
        raise ValueError(f"{name} must have shape [V,2]")
    return array


def _draw_edges(
    axis: plt.Axes,
    pose: np.ndarray,
    edges: Sequence[tuple[int, int]],
    valid: np.ndarray,
    *,
    color: str,
    alpha: float = 1.0,
    linewidth: float = 1.4,
) -> None:
    for source, target in edges:
        if not (0 <= source < len(pose) and 0 <= target < len(pose)):
            raise ValueError(f"edge {(source, target)} is outside the pose")
        if valid[source] and valid[target] and np.all(np.isfinite(pose[[source, target]])):
            axis.plot(
                pose[[source, target], 0],
                pose[[source, target], 1],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                zorder=1,
            )


def plot_reconstruction_example(
    target: Any,
    masked_input: Any,
    prediction: Any,
    *,
    synthetic_mask: Any,
    original_valid_mask: Any | None = None,
    edges: Sequence[tuple[int, int]] = (),
    bbox: Any | None = None,
    title: str = "Masked-keypoint reconstruction example",
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, np.ndarray]:
    """Show ground truth, the masked input, and reconstructed held-out joints."""

    truth = _pose(target, "target")
    observed_pose = _pose(masked_input, "masked_input")
    reconstructed = _pose(prediction, "prediction")
    if truth.shape != observed_pose.shape or truth.shape != reconstructed.shape:
        raise ValueError("target, masked_input, and prediction shapes must match")
    hidden = np.asarray(synthetic_mask, dtype=bool).reshape(-1)
    valid = (
        np.ones(len(truth), dtype=bool)
        if original_valid_mask is None
        else np.asarray(original_valid_mask, dtype=bool).reshape(-1)
    )
    if hidden.shape != valid.shape or len(hidden) != len(truth):
        raise ValueError("joint masks must have one value per joint")
    hidden &= valid
    observed = valid & ~hidden

    figure, axes = plt.subplots(1, 3, figsize=(11.2, 4.0), sharex=True, sharey=True, constrained_layout=True)
    panels = (
        ("Ground truth", truth, valid),
        ("Masked input", observed_pose, observed),
        ("Reconstruction", truth, observed),
    )
    for axis, (panel_title, pose, panel_valid) in zip(axes, panels, strict=True):
        _draw_edges(axis, pose, edges, panel_valid, color="#475569", alpha=0.8)
        axis.scatter(
            pose[panel_valid, 0],
            pose[panel_valid, 1],
            s=31,
            color="#2563a6",
            edgecolors="white",
            linewidths=0.45,
            zorder=2,
            label="observed/target",
        )
        if panel_title == "Masked input" and np.any(hidden):
            axis.scatter(
                truth[hidden, 0], truth[hidden, 1], marker="x", s=42, color="#b91c1c", label="held out"
            )
        if panel_title == "Reconstruction" and np.any(hidden):
            _draw_edges(axis, reconstructed, edges, valid, color="#d97706", alpha=0.55, linewidth=1.1)
            axis.scatter(
                reconstructed[hidden, 0],
                reconstructed[hidden, 1],
                marker="D",
                s=35,
                color="#d97706",
                edgecolors="white",
                linewidths=0.45,
                zorder=3,
                label="reconstructed",
            )
            axis.scatter(
                truth[hidden, 0],
                truth[hidden, 1],
                marker="x",
                s=38,
                color="#b91c1c",
                zorder=4,
                label="target",
            )
        if bbox is not None:
            box = np.asarray(bbox, dtype=float).reshape(-1)
            if box.size != 4:
                raise ValueError("bbox must contain xywh")
            rectangle = plt.Rectangle(
                (box[0], box[1]), box[2], box[3], fill=False, color="#64748b", linestyle="--", linewidth=0.8
            )
            axis.add_patch(rectangle)
        axis.set(title=panel_title, xlabel="x")
        axis.grid(alpha=0.14, linewidth=0.5)
        axis.set_aspect("equal", adjustable="box")
        if not axis.yaxis_inverted():
            axis.invert_yaxis()
    axes[0].set_ylabel("y")
    for axis in axes[1:]:
        if axis.get_legend_handles_labels()[0]:
            axis.legend(frameon=False, fontsize=7, loc="best")
    figure.suptitle(title, fontsize=12)
    _save(figure, output_path, dpi)
    return figure, axes


def plot_domain_summary(
    domains: Any,
    *,
    title: str = "TrackID3x3 sequence distribution by domain",
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    values = pd.Series(np.asarray(domains).reshape(-1).astype(str))
    if values.empty:
        raise ValueError("domains is empty")
    counts = values.value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(5.8, 4.1), constrained_layout=True)
    bars = axis.bar(counts.index, counts.values, color="#3976af", width=0.62)
    axis.bar_label(bars, padding=3, fontsize=9)
    axis.set(xlabel="Capture domain", ylabel="Sequences", title=title)
    axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    _save(figure, output_path, dpi)
    return figure, axis


def plot_pose_instance_counts(
    domains: Any,
    valid_mask: Any,
    *,
    sequence_mask: Any | None = None,
    title: str = "Annotated pose instances by domain",
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    domain_values = np.asarray(domains).reshape(-1).astype(str)
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 3 or len(valid) != len(domain_values):
        raise ValueError("valid_mask must be [N,T,V] aligned with domains")
    frames = np.any(valid, axis=-1)
    if sequence_mask is not None:
        sequence = np.asarray(sequence_mask, dtype=bool)
        if sequence.shape != frames.shape:
            raise ValueError("sequence_mask must have shape [N,T]")
        frames &= sequence
    counts = pd.DataFrame({"domain": domain_values, "instances": frames.sum(axis=1)}).groupby("domain")["instances"].sum()
    figure, axis = plt.subplots(figsize=(5.8, 4.1), constrained_layout=True)
    bars = axis.bar(counts.index, counts.values, color="#558b6e", width=0.62)
    axis.bar_label(bars, padding=3, fontsize=9)
    axis.set(xlabel="Capture domain", ylabel="Pose instances", title=title)
    axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    _save(figure, output_path, dpi)
    return figure, axis


def missingness_by_domain_joint(
    domains: Any,
    valid_mask: Any,
    *,
    sequence_mask: Any | None = None,
    joint_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    domain_values = np.asarray(domains).reshape(-1).astype(str)
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 3 or len(valid) != len(domain_values):
        raise ValueError("valid_mask must be [N,T,V] aligned with domains")
    frame_mask = np.ones(valid.shape[:2], dtype=bool)
    if sequence_mask is not None:
        frame_mask = np.asarray(sequence_mask, dtype=bool)
        if frame_mask.shape != valid.shape[:2]:
            raise ValueError("sequence_mask must have shape [N,T]")
    names = (
        [f"joint_{index}" for index in range(valid.shape[-1])]
        if joint_names is None
        else [str(value) for value in joint_names]
    )
    if len(names) != valid.shape[-1]:
        raise ValueError("joint_names must have one name per joint")
    rows: list[dict[str, Any]] = []
    for domain in sorted(np.unique(domain_values)):
        selected = domain_values == domain
        eligible_frames = frame_mask[selected]
        denominator = int(eligible_frames.sum())
        for joint_index, name in enumerate(names):
            observed = int((valid[selected, :, joint_index] & eligible_frames).sum())
            rows.append(
                {
                    "domain": domain,
                    "joint": name,
                    "missing_fraction": (1.0 - observed / denominator) if denominator else np.nan,
                    "eligible_frames": denominator,
                }
            )
    return pd.DataFrame(rows)


def plot_missingness_heatmap(
    table: pd.DataFrame,
    *,
    title: str = "Natural keypoint missingness by domain and joint",
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    required = {"domain", "joint", "missing_fraction"}
    if not required.issubset(table.columns):
        raise ValueError(f"table requires {sorted(required)}")
    pivot = table.pivot(index="domain", columns="joint", values="missing_fraction")
    if pivot.empty:
        raise ValueError("missingness table is empty")
    figure, axis = plt.subplots(figsize=(max(7.0, 0.62 * len(pivot.columns)), 3.7), constrained_layout=True)
    image = axis.imshow(pivot.to_numpy(float), cmap="magma_r", vmin=0, vmax=1, aspect="auto")
    figure.colorbar(image, ax=axis, label="Missing fraction", fraction=0.035, pad=0.03)
    axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
    axis.set(xlabel="Joint", ylabel="Domain", title=title)
    _save(figure, output_path, dpi)
    return figure, axis


def plot_coordinate_distributions(
    raw_coordinates: Any,
    normalized_coordinates: Any,
    *,
    valid_mask: Any,
    title: str = "Raw and bbox-normalized coordinate distributions",
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, np.ndarray]:
    raw = np.asarray(raw_coordinates, dtype=float)
    normalized = np.asarray(normalized_coordinates, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    if raw.shape != normalized.shape or raw.shape[:-1] != valid.shape or raw.shape[-1] != 2:
        raise ValueError("coordinates must share [N,T,V,2] and valid_mask [N,T,V]")
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 5.7), constrained_layout=True)
    for row, (values, label) in enumerate(((raw, "Raw (pixels)"), (normalized, "BBox normalized"))):
        for coordinate in range(2):
            selected = values[..., coordinate][valid]
            axes[row, coordinate].hist(selected, bins=40, color="#3976af", alpha=0.85)
            axes[row, coordinate].set(
                xlabel=f"{label} {'x' if coordinate == 0 else 'y'}",
                ylabel="Count",
            )
            axes[row, coordinate].grid(axis="y", alpha=0.16, linewidth=0.5)
    figure.suptitle(title, fontsize=12)
    _save(figure, output_path, dpi)
    return figure, axes


def _metric_column(table: pd.DataFrame, base: str) -> str:
    for candidate in (base, f"{base}_mean"):
        if candidate in table and table[candidate].notna().any():
            return candidate
    raise ValueError(f"table contains no observed {base} values")


def plot_nmje_by_model(
    table: pd.DataFrame,
    *,
    title: str = "BBox-normalized mean joint error by model",
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    metric = _metric_column(table, "nmje")
    if "model" not in table:
        raise ValueError("table requires model")
    data = table.dropna(subset=["model", metric]).copy()
    label_columns = [column for column in ("model", "representation", "test_domain", "mask_ratio") if column in data]
    data["configuration"] = data[label_columns].astype(str).agg(" / ".join, axis=1)
    data = data.sort_values(metric, ascending=False, kind="stable")
    errors = data["nmje_std"].fillna(0) if "nmje_std" in data else None
    figure, axis = plt.subplots(figsize=(7.4, max(3.8, 0.48 * len(data) + 1.5)), constrained_layout=True)
    bars = axis.barh(data["configuration"], data[metric], xerr=errors, color="#3f7f6f", capsize=3)
    axis.bar_label(bars, labels=[f"{value:.4f}" for value in data[metric]], padding=3, fontsize=8)
    axis.set(xlabel="NMJE (bbox diagonal; lower is better)", ylabel="", title=title)
    axis.grid(axis="x", alpha=0.2, linewidth=0.6)
    _save(figure, output_path, dpi)
    return figure, axis


def plot_pck_curves(
    table: pd.DataFrame,
    *,
    title: str = "Percentage of Correct Keypoints",
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    thresholds = (0.05, 0.10, 0.20)
    columns = []
    for threshold in thresholds:
        base = f"pck_at_{threshold:.2f}".replace(".", "_")
        columns.append(_metric_column(table, base))
    if table.empty:
        raise ValueError("PCK table is empty")
    label_columns = [column for column in ("model", "representation", "train_domain", "test_domain", "mask_ratio") if column in table]
    figure, axis = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    for _, row in table.iterrows():
        label = " / ".join(str(row[column]) for column in label_columns) or "executed run"
        values = [float(row[column]) for column in columns]
        axis.plot(thresholds, values, marker="o", linewidth=1.55, label=label)
    axis.set(
        xlabel="BBox-normalized distance threshold",
        ylabel="PCK",
        title=title,
        xticks=thresholds,
        ylim=(0, 1.03),
    )
    axis.grid(alpha=0.2, linewidth=0.6)
    axis.legend(frameon=False, fontsize=7, loc="best")
    _save(figure, output_path, dpi)
    return figure, axis


def plot_per_joint_error(
    per_joint: Mapping[str, Mapping[str, Any]] | pd.DataFrame,
    *,
    title: str = "Per-joint bbox-normalized reconstruction error",
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    if isinstance(per_joint, Mapping):
        rows = [
            {"joint": joint, "nmje": values.get("nmje"), "count": values.get("count")}
            for joint, values in per_joint.items()
        ]
        table = pd.DataFrame(rows)
    else:
        table = per_joint.copy()
    metric = _metric_column(table, "nmje")
    if "joint" not in table:
        raise ValueError("per-joint table requires joint names")
    data = table.dropna(subset=["joint", metric]).sort_values(metric, ascending=False)
    if data.empty:
        raise ValueError("no per-joint errors are available")
    figure, axis = plt.subplots(figsize=(7.2, max(3.8, 0.42 * len(data) + 1.4)), constrained_layout=True)
    bars = axis.barh(data["joint"].astype(str), data[metric].astype(float), color="#b8654a")
    axis.bar_label(bars, labels=[f"{value:.4f}" for value in data[metric]], padding=3, fontsize=8)
    axis.set(xlabel="NMJE (lower is better)", ylabel="Joint", title=title)
    axis.grid(axis="x", alpha=0.2, linewidth=0.6)
    _save(figure, output_path, dpi)
    return figure, axis


def plot_cross_domain_matrix(
    table: pd.DataFrame,
    *,
    metric: str = "nmje_mean",
    title: str = "Cross-domain reconstruction NMJE",
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    if metric not in table and metric.endswith("_mean") and metric[:-5] in table:
        metric = metric[:-5]
    required = {"train_domain", "test_domain", metric}
    if not required.issubset(table.columns):
        raise ValueError(f"table requires {sorted(required)}")
    pivot = table.pivot_table(index="train_domain", columns="test_domain", values=metric, aggfunc="mean")
    if pivot.empty:
        raise ValueError("cross-domain table is empty")
    values = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("viridis_r").with_extremes(bad="#e5e7eb")
    figure, axis = plt.subplots(figsize=(5.8, 4.8), constrained_layout=True)
    image = axis.imshow(masked, cmap=cmap, aspect="auto")
    figure.colorbar(image, ax=axis, label="NMJE (lower is better)", fraction=0.045, pad=0.04)
    axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
    axis.set(xlabel="Test domain", ylabel="Training domain", title=title)
    finite = values[np.isfinite(values)]
    threshold = float(np.median(finite)) if finite.size else 0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(
                column,
                row,
                "—" if not np.isfinite(value) else f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if np.isfinite(value) and value > threshold else "#111827",
                fontsize=9,
            )
    _save(figure, output_path, dpi)
    return figure, axis


def plot_reconstruction_data_efficiency(
    table: pd.DataFrame,
    *,
    title: str = "Pose reconstruction data efficiency",
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    size_column = next(
        (column for column in ("data_fraction", "data_percent", "num_train_sequences") if column in table),
        None,
    )
    if size_column is None:
        raise ValueError("table needs data_fraction, data_percent, or num_train_sequences")
    metric = _metric_column(table, "nmje")
    data = table.dropna(subset=[size_column, metric]).copy()
    if data[size_column].nunique() < 2:
        raise ValueError("a data-efficiency curve requires at least two observed sizes")
    group_columns = [column for column in ("model", "representation", "train_domain", "test_domain", "mask_ratio") if column in data]
    figure, axis = plt.subplots(figsize=(6.3, 4.4), constrained_layout=True)
    grouped: Any = [("executed", data)] if not group_columns else data.groupby(group_columns, sort=True)
    for key, group in grouped:
        group = group.sort_values(size_column)
        label = " / ".join(str(value) for value in key) if isinstance(key, tuple) else str(key)
        error = group["nmje_std"].fillna(0).to_numpy() if "nmje_std" in group else None
        axis.errorbar(group[size_column], group[metric], yerr=error, marker="o", capsize=3, label=label)
    axis.set(xlabel=size_column.replace("_", " ").title(), ylabel="NMJE (lower is better)", title=title)
    axis.grid(alpha=0.2, linewidth=0.6)
    if group_columns:
        axis.legend(frameon=False, fontsize=7)
    _save(figure, output_path, dpi)
    return figure, axis
