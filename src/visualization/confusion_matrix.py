"""Publication-readable confusion-matrix figures."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    matrix: Any,
    class_names: Sequence[str],
    *,
    normalize: bool = False,
    title: str | None = None,
    output_path: str | Path | None = None,
    dpi: int = 300,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a square confusion matrix with legible cell annotations."""

    values = np.asarray(matrix)
    names = [str(name) for name in class_names]
    expected = (len(names), len(names))
    if values.shape != expected:
        raise ValueError(f"matrix must have shape {expected}, got {values.shape}")
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("confusion-matrix entries must be finite and non-negative")

    display = values.astype(float)
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(display, row_sums, out=np.zeros_like(display), where=row_sums > 0)

    if ax is None:
        figure, ax = plt.subplots(figsize=(5.4, 4.8), constrained_layout=True)
    else:
        figure = ax.figure
    image = ax.imshow(display, interpolation="nearest", cmap="Blues", vmin=0)
    colorbar = figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.set_ylabel("Row proportion" if normalize else "Samples", rotation=270, labelpad=16)

    ax.set(
        xticks=np.arange(len(names)),
        yticks=np.arange(len(names)),
        xticklabels=names,
        yticklabels=names,
        xlabel="Predicted action",
        ylabel="True action",
        title=title or ("Normalized confusion matrix" if normalize else "Confusion matrix"),
    )
    ax.tick_params(axis="x", rotation=25)
    threshold = float(np.nanmax(display)) / 2.0 if display.size else 0.0
    for row in range(display.shape[0]):
        for column in range(display.shape[1]):
            text = f"{display[row, column]:.2f}" if normalize else f"{int(values[row, column])}"
            ax.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color="white" if display[row, column] > threshold else "#202020",
                fontsize=10,
            )
    ax.set_ylim(len(names) - 0.5, -0.5)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    return figure, ax


def save_confusion_matrix(*args: Any, **kwargs: Any) -> Path:
    """Save a confusion matrix and return the output path."""

    output_path = kwargs.get("output_path")
    if output_path is None:
        raise ValueError("output_path is required")
    figure, _ = plot_confusion_matrix(*args, **kwargs)
    plt.close(figure)
    return Path(output_path).resolve()
