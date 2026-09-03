"""Compact pose models, including the TrackID3x3 reconstruction API."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from torch import Tensor, nn

from .baselines import (
    geometric_neighbor_baseline,
    reconstruction_baseline,
    temporal_interpolation_baseline,
)
from .lstm import BiLSTM, BiLSTMClassifier, LSTMClassifier, LSTMModel
from .mlp import AggregatedMLP, MLPBaseline, MLPClassifier, PoseMLP
from .reconstruction import (
    RECONSTRUCTION_MODEL_NAMES,
    TRACKID3X3_EDGES,
    TRACKID3X3_JOINT_NAMES,
    MaskedMLPReconstructor,
    MaskedTCNReconstructor,
    ReconstructionGraph,
    ReconstructionMLP,
    ReconstructionTCN,
    TrackIDGraphReconstructor,
    build_reconstruction_model,
    trackid3x3_adjacency,
    validate_trackid3x3_anatomy,
)
from .stgcn import (
    COCO_17_EDGES,
    MEDIAPIPE_33_EDGES,
    NTU_25_EDGES,
    STGCN,
    CompactSTGCN,
    STGCNClassifier,
    STGCNModel,
    anatomical_edges,
    normalized_adjacency,
)
from .tcn import TCN, LightweightTCN, TCNClassifier, TemporalConvNet

MODEL_NAMES = ("mlp", "lstm", "bilstm", "tcn", "stgcn")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters, optionally excluding frozen tensors."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )


def _channels(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, int):
        return (value, value, value)
    return tuple(int(item) for item in value)


def build_model(
    name: str,
    *,
    in_features: int,
    num_classes: int = 3,
    num_joints: int | None = None,
    edges: Iterable[tuple[int, int]] | Tensor | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Construct a model while tolerating shared experiment-config fields.

    Only fields relevant to the selected architecture are consumed, which lets a
    single resolved YAML configuration drive all compact baselines.
    """

    model_name = name.lower().replace("-", "").replace("_", "")
    dropout = float(kwargs.get("dropout", 0.25))
    if model_name in {"mlp", "baseline", "aggregatedmlp"}:
        hidden = kwargs.get("hidden_dims")
        if hidden is None and kwargs.get("hidden_dim") is not None:
            width = int(kwargs["hidden_dim"])
            hidden = (max(width * 2, 64), width)
        return AggregatedMLP(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dims=_channels(hidden, (256, 128)),
            dropout=dropout,
        )
    if model_name in {"lstm", "bilstm"}:
        return BiLSTMClassifier(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dim=int(kwargs.get("hidden_dim", 96)),
            num_layers=int(kwargs.get("num_layers", 2)),
            dropout=dropout,
            bidirectional=bool(kwargs.get("bidirectional", True)),
        )
    if model_name in {"tcn", "temporalconvnet", "lightweighttcn"}:
        channels = kwargs.get("channels")
        if channels is None and kwargs.get("hidden_dim") is not None:
            width = int(kwargs["hidden_dim"])
            channels = (max(width // 2, 32), width, width)
        return TCNClassifier(
            in_features=in_features,
            num_classes=num_classes,
            channels=_channels(channels, (64, 96, 128)),
            kernel_size=int(kwargs.get("kernel_size", 3)),
            dropout=dropout,
        )
    if model_name in {"stgcn", "compactstgcn"}:
        if num_joints is None:
            raise ValueError("num_joints is required for ST-GCN")
        channels = kwargs.get("channels")
        return STGCNClassifier(
            in_features=in_features,
            num_joints=num_joints,
            num_classes=num_classes,
            edges=edges,
            channels=_channels(channels, (48, 64, 96, 128)),
            temporal_kernel=int(kwargs.get("temporal_kernel", 5)),
            dropout=dropout,
        )
    raise ValueError(f"Unknown model {name!r}. Choose one of: {', '.join(MODEL_NAMES)}")


def model_metadata(model: nn.Module) -> Mapping[str, Any]:
    """Return compact architecture metadata suitable for JSON/checkpoints."""

    return {
        "class_name": type(model).__name__,
        "parameter_count": count_parameters(model),
        "in_features": getattr(model, "in_features", None),
        "num_joints": getattr(model, "num_joints", None),
        "num_classes": getattr(model, "num_classes", None),
        "embedding_dim": getattr(model, "embedding_dim", None),
    }


__all__ = [
    "COCO_17_EDGES",
    "MEDIAPIPE_33_EDGES",
    "MODEL_NAMES",
    "NTU_25_EDGES",
    "RECONSTRUCTION_MODEL_NAMES",
    "STGCN",
    "TCN",
    "TRACKID3X3_EDGES",
    "TRACKID3X3_JOINT_NAMES",
    "AggregatedMLP",
    "BiLSTM",
    "BiLSTMClassifier",
    "CompactSTGCN",
    "LSTMClassifier",
    "LSTMModel",
    "LightweightTCN",
    "MLPBaseline",
    "MLPClassifier",
    "MaskedMLPReconstructor",
    "MaskedTCNReconstructor",
    "PoseMLP",
    "ReconstructionGraph",
    "ReconstructionMLP",
    "ReconstructionTCN",
    "STGCNClassifier",
    "STGCNModel",
    "TCNClassifier",
    "TemporalConvNet",
    "TrackIDGraphReconstructor",
    "anatomical_edges",
    "build_model",
    "build_reconstruction_model",
    "count_parameters",
    "geometric_neighbor_baseline",
    "model_metadata",
    "normalized_adjacency",
    "reconstruction_baseline",
    "temporal_interpolation_baseline",
    "trackid3x3_adjacency",
    "validate_trackid3x3_anatomy",
]
