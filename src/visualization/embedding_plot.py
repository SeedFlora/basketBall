"""PCA and t-SNE projections that remain safe for very small test sets."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.evaluation.metrics import DEFAULT_CLASS_NAMES


def _embedding_matrix(embeddings: Any) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    elif matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional sample-by-feature matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("embeddings contain non-finite values")
    return matrix


def pca_projection(embeddings: Any, *, random_state: int = 42) -> np.ndarray:
    """Project embeddings to two PCA axes, padding unavailable axes with zero."""

    matrix = _embedding_matrix(embeddings)
    n_samples, n_features = matrix.shape
    if n_samples == 0:
        return np.empty((0, 2), dtype=float)
    if n_samples == 1 or n_features == 0:
        return np.zeros((n_samples, 2), dtype=float)
    components = min(2, n_samples, n_features)
    projection = PCA(n_components=components, random_state=random_state).fit_transform(matrix)
    if components < 2:
        projection = np.column_stack([projection, np.zeros(n_samples, dtype=float)])
    return projection.astype(float, copy=False)


def tsne_projection(
    embeddings: Any,
    *,
    random_state: int = 42,
    perplexity: float = 30.0,
) -> np.ndarray:
    """Project embeddings with t-SNE using a valid data-dependent perplexity.

    For fewer than three samples, PCA is returned because a t-SNE neighbourhood
    is not meaningful.  Numerical t-SNE failures also fall back to PCA with a
    warning rather than preventing the evaluation report from being generated.
    """

    matrix = _embedding_matrix(embeddings)
    n_samples = matrix.shape[0]
    fallback = pca_projection(matrix, random_state=random_state)
    if n_samples < 3 or matrix.shape[1] == 0:
        return fallback
    effective_perplexity = min(float(perplexity), float(n_samples - 1))
    effective_perplexity = max(1.0, effective_perplexity)
    try:
        return TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            init="random",
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(matrix)
    except (ValueError, FloatingPointError) as error:
        warnings.warn(f"t-SNE failed ({error}); using PCA coordinates", RuntimeWarning, stacklevel=2)
        return fallback


def project_embeddings(
    embeddings: Any,
    *,
    random_state: int = 42,
    perplexity: float = 30.0,
) -> dict[str, np.ndarray]:
    """Return both supported two-dimensional projections."""

    return {
        "pca": pca_projection(embeddings, random_state=random_state),
        "tsne": tsne_projection(
            embeddings, random_state=random_state, perplexity=perplexity
        ),
    }


def plot_embedding(
    projection: Any,
    labels: Any,
    *,
    class_names: Mapping[int, str] | Sequence[str] | None = DEFAULT_CLASS_NAMES,
    title: str = "Embedding projection",
    axis_labels: tuple[str, str] = ("Component 1", "Component 2"),
    subject_ids: Any | None = None,
    output_path: str | Path | None = None,
    dpi: int = 300,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter a two-dimensional embedding, colored only by observed classes."""

    coordinates = np.asarray(projection, dtype=float)
    label_values = np.asarray(labels).reshape(-1)
    if coordinates.shape != (len(label_values), 2):
        raise ValueError("projection must have shape [number of labels, 2]")
    subjects = None if subject_ids is None else np.asarray(subject_ids).reshape(-1)
    if subjects is not None and len(subjects) != len(label_values):
        raise ValueError("subject_ids must have one entry per projection row")

    if isinstance(class_names, Mapping):
        name_map = {int(key): str(value) for key, value in class_names.items()}
    elif class_names is None:
        name_map = DEFAULT_CLASS_NAMES
    else:
        name_map = {index: str(value) for index, value in enumerate(class_names)}
    if ax is None:
        figure, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    else:
        figure = ax.figure

    observed = sorted(np.unique(label_values).tolist())
    colors = plt.get_cmap("tab10")
    for index, label in enumerate(observed):
        selected = label_values == label
        ax.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=34,
            alpha=0.78,
            color=colors(index % 10),
            edgecolors="white",
            linewidths=0.35,
            label=name_map.get(int(label), str(label)),
        )
    if coordinates.shape[0] < 3:
        ax.text(
            0.02,
            0.02,
            "Projection is descriptive only (fewer than 3 samples).",
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
        )
    ax.set(xlabel=axis_labels[0], ylabel=axis_labels[1], title=title)
    if observed:
        ax.legend(title="Action", frameon=False, loc="best")
    ax.grid(alpha=0.18, linewidth=0.6)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    return figure, ax


def save_embedding_projections(
    embeddings: Any,
    labels: Any,
    output_dir: str | Path,
    *,
    class_names: Mapping[int, str] | Sequence[str] | None = DEFAULT_CLASS_NAMES,
    random_state: int = 42,
    perplexity: float = 30.0,
    prefix: str = "test_embedding",
    title_suffix: str = "",
) -> dict[str, Path]:
    """Compute and save PCA and t-SNE figures."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    projections = project_embeddings(
        embeddings, random_state=random_state, perplexity=perplexity
    )
    paths: dict[str, Path] = {}
    for method, projection in projections.items():
        path = destination / f"{prefix}_{method}.png"
        figure, _ = plot_embedding(
            projection,
            labels,
            class_names=class_names,
            title=f"Test-set embedding ({method.upper()}){title_suffix}",
            axis_labels=(f"{method.upper()} 1", f"{method.upper()} 2"),
            output_path=path,
        )
        plt.close(figure)
        paths[method] = path.resolve()
    return paths
