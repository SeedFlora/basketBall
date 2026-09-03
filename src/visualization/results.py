"""Compact result plots shared by evaluation and ablation scripts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(figure: plt.Figure, output_path: str | Path | None, dpi: int) -> None:
    if output_path is None:
        return
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")


def plot_per_class_f1(
    per_class: Mapping[str, Mapping[str, Any]],
    *,
    title: str = "Per-class test F1",
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot F1 only for classes actually present in a metrics artifact."""

    names, values = [], []
    for name, metrics in per_class.items():
        value = metrics.get("f1")
        if isinstance(value, (int, float)) and np.isfinite(value):
            names.append(str(name))
            values.append(float(value))
    if not names:
        raise ValueError("per_class contains no finite F1 values")
    figure, axis = plt.subplots(figsize=(5.7, 4.2), constrained_layout=True)
    bars = axis.bar(names, values, color="#3976af", width=0.62)
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9)
    axis.set(ylabel="F1 score", xlabel="Action", title=title, ylim=(0, 1.08))
    axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    _save(figure, output_path, dpi)
    return figure, axis


def plot_class_distribution(
    labels: Any,
    *,
    class_names: Mapping[int, str] | Sequence[str] | None = None,
    title: str = "Observed class distribution",
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot observed class counts without inserting absent sample counts."""

    values = np.asarray(labels).reshape(-1)
    if values.size == 0:
        raise ValueError("labels is empty")
    unique, counts = np.unique(values, return_counts=True)
    if isinstance(class_names, Mapping):
        names = [str(class_names.get(int(value), value)) for value in unique]
    elif class_names is not None:
        supplied = list(class_names)
        names = [str(supplied[int(value)]) if 0 <= int(value) < len(supplied) else str(value) for value in unique]
    else:
        names = [str(value) for value in unique]
    figure, axis = plt.subplots(figsize=(5.7, 4.1), constrained_layout=True)
    bars = axis.bar(names, counts, color="#527db5", width=0.62)
    axis.bar_label(bars, padding=3, fontsize=9)
    axis.set(xlabel="Action", ylabel="Clips", title=title)
    axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    _save(figure, output_path, dpi)
    return figure, axis


def plot_subject_distribution(
    subject_ids: Any,
    *,
    title: str | None = None,
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot clips per observed subject in sorted count order."""

    values = pd.Series(np.asarray(subject_ids).reshape(-1), dtype="object")
    if values.empty:
        raise ValueError("subject_ids is empty")
    counts = values.value_counts(dropna=False).sort_values(kind="stable", ascending=False)
    figure, axis = plt.subplots(figsize=(7.0, 4.1), constrained_layout=True)
    x = np.arange(1, len(counts) + 1)
    axis.bar(x, counts.to_numpy(), color="#6b8e6e", width=0.82)
    axis.set(
        xlabel="Subjects (ordered by clip count)",
        ylabel="Clips",
        title=title or f"Observed subject distribution (n={len(counts)})",
    )
    if len(counts) <= 25:
        axis.set_xticks(x, [str(value) for value in counts.index], rotation=55, ha="right")
    axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    _save(figure, output_path, dpi)
    return figure, axis


def plot_joint_angle_example(
    angle_sequence: Any,
    *,
    angle_names: Sequence[str] | None = None,
    max_angles: int = 8,
    title: str = "Joint-angle sequence example",
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot measured joint-angle trajectories for one clip."""

    values = np.asarray(angle_sequence, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("angle_sequence must have shape [time, angles]")
    count = min(values.shape[1], max(1, int(max_angles)))
    names = list(angle_names or [f"Angle {index + 1}" for index in range(values.shape[1])])
    if len(names) < values.shape[1]:
        raise ValueError("angle_names has fewer entries than angle features")
    figure, axis = plt.subplots(figsize=(7.1, 4.4), constrained_layout=True)
    for index in range(count):
        axis.plot(values[:, index], linewidth=1.45, label=names[index])
    axis.set(xlabel="Standardized frame", ylabel="Angle (radians)", title=title)
    axis.legend(frameon=False, ncol=2, fontsize=8)
    axis.grid(alpha=0.2, linewidth=0.6)
    _save(figure, output_path, dpi)
    return figure, axis


def plot_joint_angle_distributions(
    angles: Any,
    labels: Any,
    *,
    angle_names: Sequence[str] | None = None,
    class_names: Mapping[int, str] | Sequence[str] | None = None,
    angle_index: int = 0,
    title: str | None = None,
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a class-wise distribution of one actually measured angle feature."""

    values = np.asarray(angles, dtype=float)
    labels_array = np.asarray(labels).reshape(-1)
    if values.shape[0] != len(labels_array) or values.ndim not in {2, 3}:
        raise ValueError("angles must have shape [N,A] or [N,T,A] aligned with labels")
    if not 0 <= angle_index < values.shape[-1]:
        raise ValueError("angle_index is outside the available angle features")
    selected = values[..., angle_index]
    if selected.ndim == 2:
        selected = np.nanmean(selected, axis=1)
    observed = sorted(np.unique(labels_array).tolist())
    if isinstance(class_names, Mapping):
        name_map = {int(key): str(value) for key, value in class_names.items()}
    elif class_names is not None:
        name_map = {index: str(value) for index, value in enumerate(class_names)}
    else:
        name_map = {}
    groups = [selected[labels_array == label] for label in observed]
    names = [name_map.get(int(label), str(label)) for label in observed]
    angle_label = (
        str(angle_names[angle_index])
        if angle_names is not None and angle_index < len(angle_names)
        else f"Angle {angle_index + 1}"
    )
    figure, axis = plt.subplots(figsize=(6.2, 4.3), constrained_layout=True)
    axis.violinplot(groups, showmeans=True, showextrema=True)
    axis.set_xticks(np.arange(1, len(names) + 1), names)
    axis.set(
        xlabel="Action",
        ylabel="Mean angle (radians)",
        title=title or f"Observed {angle_label} distribution",
    )
    axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    _save(figure, output_path, dpi)
    return figure, axis


def plot_robustness_curve(
    table: pd.DataFrame,
    *,
    metric: str = "macro_f1",
    title: str = "Test-time missing-keypoint robustness",
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot an observed metric against simulated missing-keypoint rate."""

    required = {"missing_rate", metric}
    if not required.issubset(table.columns):
        raise ValueError(f"robustness table requires columns {sorted(required)}")
    data = table.dropna(subset=["missing_rate", metric]).copy()
    if data.empty:
        raise ValueError("robustness table contains no observed metric values")
    if "strategy" not in data:
        data["strategy"] = "unspecified"
    figure, axis = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    series_columns = [
        column
        for column in ("model", "representation", "augmentation", "strategy")
        if column in data and data[column].notna().any()
    ]
    group_key: str | list[str] = series_columns[0] if len(series_columns) == 1 else series_columns
    for series, group in data.groupby(group_key, sort=True):
        summary = group.groupby("missing_rate", as_index=False)[metric].agg(["mean", "std"]).reset_index()
        x = summary["missing_rate"].to_numpy(dtype=float) * 100.0
        y = summary["mean"].to_numpy(dtype=float)
        series_values = (series,) if len(series_columns) == 1 else tuple(series)
        label = " / ".join(str(value).replace("_", " ") for value in series_values)
        axis.plot(x, y, marker="o", linewidth=1.8, markersize=5, label=label)
        if len(group) > len(summary):
            deviation = summary["std"].fillna(0).to_numpy(dtype=float)
            axis.fill_between(x, y - deviation, y + deviation, alpha=0.12)
    axis.set(
        xlabel="Simulated missing joints (%)",
        ylabel=metric.replace("_", " ").title(),
        title=title,
        ylim=(0, 1.03) if metric.endswith("f1") or metric == "accuracy" else None,
    )
    axis.grid(alpha=0.2, linewidth=0.6)
    axis.legend(title="Configuration / strategy", frameon=False, fontsize=8)
    _save(figure, output_path, dpi)
    return figure, axis


def plot_ablation_comparison(
    table: pd.DataFrame,
    *,
    category: str = "representation",
    metric: str = "macro_f1_mean",
    error: str | None = "macro_f1_std",
    title: str = "Representation ablation",
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot only rows with an actually observed category and metric."""

    if category not in table or metric not in table:
        raise ValueError(f"table requires {category!r} and {metric!r}")
    data = table.dropna(subset=[category, metric]).copy()
    if data.empty:
        raise ValueError("table contains no observed ablation results")
    data = data.sort_values(metric, ascending=True, kind="stable")
    errors = None
    if error and error in data and data[error].notna().any():
        errors = data[error].fillna(0).to_numpy(dtype=float)
    height = max(3.7, 0.52 * len(data) + 1.7)
    figure, axis = plt.subplots(figsize=(7.2, height), constrained_layout=True)
    bars = axis.barh(
        data[category].astype(str),
        data[metric].astype(float),
        xerr=errors,
        color="#3f7f6f",
        alpha=0.92,
        capsize=3,
    )
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in data[metric]], padding=4, fontsize=8)
    axis.set(xlabel=metric.replace("_mean", "").replace("_", " ").title(), ylabel="", title=title)
    if "f1" in metric or "accuracy" in metric:
        axis.set_xlim(0, 1.08)
    axis.grid(axis="x", alpha=0.2, linewidth=0.6)
    _save(figure, output_path, dpi)
    return figure, axis


def plot_sample_size_curve(
    table: pd.DataFrame,
    *,
    size_column: str = "num_subjects",
    metric: str = "macro_f1_mean",
    error: str | None = "macro_f1_std",
    title: str = "Data-efficiency analysis",
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot data efficiency from executed sample-size experiments only."""

    if size_column not in table or metric not in table:
        raise ValueError(f"table requires {size_column!r} and {metric!r}")
    data = table.dropna(subset=[size_column, metric]).sort_values(size_column)
    if data.empty:
        raise ValueError("table contains no observed sample-size results")
    figure, axis = plt.subplots(figsize=(6.2, 4.3), constrained_layout=True)
    group_columns = [column for column in ("model", "representation") if column in data]
    grouped: Any = [("all", data)] if not group_columns else data.groupby(group_columns, sort=True)
    for group_key, group in grouped:
        group = group.sort_values(size_column)
        x = group[size_column].to_numpy(dtype=float)
        y = group[metric].to_numpy(dtype=float)
        yerr = None
        if error and error in group and group[error].notna().any():
            yerr = group[error].fillna(0).to_numpy(dtype=float)
        if isinstance(group_key, tuple):
            label = " / ".join(str(value) for value in group_key)
        else:
            label = str(group_key)
        axis.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.8, capsize=3, label=label)
    axis.set(
        xlabel="Number of sampled subjects",
        ylabel=metric.replace("_mean", "").replace("_", " ").title(),
        title=title,
        ylim=(0, 1.03) if "f1" in metric or "accuracy" in metric else None,
    )
    if group_columns:
        axis.legend(title="Configuration", frameon=False, fontsize=8)
    axis.grid(alpha=0.2, linewidth=0.6)
    _save(figure, output_path, dpi)
    return figure, axis


def plot_training_history(
    history: pd.DataFrame,
    *,
    train_column: str,
    validation_column: str,
    ylabel: str,
    title: str,
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot observed training and validation curves."""

    if train_column not in history or validation_column not in history:
        raise ValueError("requested history columns are unavailable")
    epochs = history["epoch"] if "epoch" in history else np.arange(1, len(history) + 1)
    figure, axis = plt.subplots(figsize=(6.2, 4.3), constrained_layout=True)
    axis.plot(epochs, history[train_column], label="Training", linewidth=1.7)
    axis.plot(epochs, history[validation_column], label="Validation", linewidth=1.7)
    axis.set(xlabel="Epoch", ylabel=ylabel, title=title)
    axis.legend(frameon=False)
    axis.grid(alpha=0.2, linewidth=0.6)
    _save(figure, output_path, dpi)
    return figure, axis
