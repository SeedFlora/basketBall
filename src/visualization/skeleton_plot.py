"""Static skeleton and temporal-sequence visualizations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _pose_coordinates(pose: Any) -> np.ndarray:
    coordinates = np.asarray(pose, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("pose must have shape [joints, at least 2 coordinates]")
    return coordinates


def plot_skeleton(
    pose: Any,
    edges: Sequence[tuple[int, int]] = (),
    *,
    joint_names: Sequence[str] | None = None,
    title: str | None = None,
    invert_y: bool = True,
    output_path: str | Path | None = None,
    dpi: int = 300,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one 2-D pose, skipping non-finite joints and edges."""

    coordinates = _pose_coordinates(pose)
    if joint_names is not None and len(joint_names) != len(coordinates):
        raise ValueError("joint_names must have one entry per joint")
    if ax is None:
        figure, ax = plt.subplots(figsize=(4.8, 5.2), constrained_layout=True)
    else:
        figure = ax.figure
    valid = np.all(np.isfinite(coordinates[:, :2]), axis=1)
    for source, target in edges:
        if not 0 <= source < len(coordinates) or not 0 <= target < len(coordinates):
            raise ValueError(f"edge {(source, target)} references a missing joint")
        if valid[source] and valid[target]:
            points = coordinates[[source, target], :2]
            ax.plot(points[:, 0], points[:, 1], color="#334155", linewidth=1.6, zorder=1)
    ax.scatter(
        coordinates[valid, 0],
        coordinates[valid, 1],
        s=28,
        color="#d94841",
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
    )
    if joint_names is not None:
        for index in np.flatnonzero(valid):
            label = str(joint_names[index])
            # Bilateral TrackID labels are long and often spatially close. Put
            # left/right labels on opposite sides of their landmarks so the
            # canonical ten-joint figure remains readable without moving data.
            if label.startswith("left_"):
                offset, alignment = (-4, 3), "right"
            else:
                offset, alignment = (4, 3), "left"
            ax.annotate(
                label,
                coordinates[index, :2],
                xytext=offset,
                textcoords="offset points",
                fontsize=6,
                ha=alignment,
            )
    ax.set(title=title or "Skeleton pose", xlabel="x", ylabel="y")
    # ``box`` remains valid when temporal panels share both axes; ``datalim``
    # raises in Matplotlib for that otherwise useful publication layout.
    ax.set_aspect("equal", adjustable="box")
    # Shared temporal axes propagate inversion to every panel. Guarding avoids
    # toggling the same shared axis once per frame (an even panel count would
    # otherwise end right-side-up).
    if invert_y and not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.grid(alpha=0.14, linewidth=0.5)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    return figure, ax


def plot_skeleton_sequence(
    sequence: Any,
    edges: Sequence[tuple[int, int]] = (),
    *,
    num_frames: int = 6,
    title: str = "Temporal skeleton sequence",
    invert_y: bool = True,
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot uniformly selected frames instead of only the sequence beginning."""

    array = np.asarray(sequence, dtype=float)
    if array.ndim != 3 or array.shape[-1] < 2:
        raise ValueError("sequence must have shape [time, joints, coordinates]")
    if array.shape[0] == 0:
        raise ValueError("sequence must contain at least one frame")
    count = max(1, min(int(num_frames), array.shape[0]))
    indices = np.linspace(0, array.shape[0] - 1, num=count, dtype=int)
    figure, axes = plt.subplots(
        1,
        count,
        figsize=(2.5 * count, 3.2),
        squeeze=False,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    for axis, frame_index in zip(axes[0], indices, strict=True):
        plot_skeleton(
            array[frame_index],
            edges,
            title=f"Frame {frame_index}",
            invert_y=invert_y,
            ax=axis,
        )
    figure.suptitle(title, fontsize=12)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    return figure, axes[0]
