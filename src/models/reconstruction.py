"""Lightweight masked-keypoint reconstruction networks.

All models consume coordinates in ``[B, T, V, C]`` order plus an explicit
``[B, T, V]`` observed mask.  They predict coordinates in the same domain and
copy observed coordinates through unchanged.  Originally missing joints may be
predicted, but training/evaluation masks decide which locations are scored.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from src.data.trackid3x3 import TRACKID10_EDGES, TRACKID10_JOINT_NAMES

# Re-export the authoritative data-adapter constants.  Keeping a single
# topology source prevents graph checkpoints from drifting from processed NPZs.
TRACKID3X3_JOINT_NAMES: tuple[str, ...] = tuple(TRACKID10_JOINT_NAMES)
TRACKID3X3_EDGES: tuple[tuple[int, int], ...] = tuple(
    (int(source), int(target)) for source, target in TRACKID10_EDGES.T.tolist()
)

_JOINT_ALIASES = {
    "nose": "head",
    "mid_hip": "center",
    "mid_hips": "center",
    "center_hip": "center",
    "center_hips": "center",
    "pelvis": "center",
}


def _canonical_joint_name(value: str) -> str:
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _JOINT_ALIASES.get(token, token)


def validate_trackid3x3_anatomy(
    num_joints: int,
    joint_names: Sequence[str] | None = None,
    edges: Iterable[tuple[int, int]] | Tensor | None = None,
) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    """Validate the exact ten-landmark TrackID3x3 graph.

    Edge direction is ignored because reconstruction uses an undirected graph.
    Unknown layouts fail instead of silently receiving a chain graph.
    """

    if int(num_joints) != len(TRACKID3X3_JOINT_NAMES):
        raise ValueError(
            "The graph reconstructor supports only the documented 10-joint "
            f"TrackID3x3 layout, got V={num_joints}."
        )
    if joint_names is None:
        names = TRACKID3X3_JOINT_NAMES
    else:
        names = tuple(_canonical_joint_name(name) for name in joint_names)
        if names != TRACKID3X3_JOINT_NAMES:
            raise ValueError(
                "joint_names do not match the supported TrackID3x3 order: "
                f"expected {TRACKID3X3_JOINT_NAMES}, got {names}."
            )
    if edges is None:
        selected_edges = TRACKID3X3_EDGES
    else:
        if isinstance(edges, Tensor):
            edge_values: Any = edges.detach().cpu().tolist()
        else:
            edge_values = list(edges)
        # Processed files may store [2,E] or [E,2].
        if (
            len(edge_values) == 2
            and edge_values
            and len(edge_values[0]) != 2
            and len(edge_values[0]) == len(edge_values[1])
        ):
            edge_values = list(zip(edge_values[0], edge_values[1]))
        selected_edges = tuple((int(edge[0]), int(edge[1])) for edge in edge_values)
        expected = {tuple(sorted(edge)) for edge in TRACKID3X3_EDGES}
        received = {tuple(sorted(edge)) for edge in selected_edges}
        if received != expected or len(selected_edges) != len(expected):
            raise ValueError(
                "edges must exactly match the supported TrackID3x3 anatomy; "
                f"expected {sorted(expected)}, got {sorted(received)}."
            )
    return names, selected_edges


def trackid3x3_adjacency(
    num_joints: int = 10,
    joint_names: Sequence[str] | None = None,
    edges: Iterable[tuple[int, int]] | Tensor | None = None,
) -> Tensor:
    """Return symmetric self-connected normalized TrackID3x3 adjacency."""

    _, selected_edges = validate_trackid3x3_anatomy(num_joints, joint_names, edges)
    adjacency = torch.eye(num_joints, dtype=torch.float32)
    for source, target in selected_edges:
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0
    inverse_sqrt = adjacency.sum(dim=1).clamp_min(1.0).rsqrt()
    return inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]


class _MaskedReconstructor(nn.Module):
    """Shape validation and observed-coordinate passthrough."""

    def __init__(self, in_features: int, num_joints: int, aux_features: int = 0) -> None:
        super().__init__()
        if in_features <= 0 or num_joints <= 0:
            raise ValueError("in_features and num_joints must be positive")
        self.in_features = int(in_features)
        self.num_joints = int(num_joints)
        self.aux_features = int(aux_features)
        if self.aux_features < 0:
            raise ValueError("aux_features must be non-negative")
        self.output_features = self.in_features

    def _prepare(self, coordinates: Tensor, observed_mask: Tensor | None) -> tuple[Tensor, Tensor]:
        if coordinates.ndim != 4:
            raise ValueError(
                f"Reconstruction inputs must be [B,T,V,C], got {tuple(coordinates.shape)}"
            )
        if coordinates.shape[-2:] != (self.num_joints, self.in_features):
            raise ValueError(
                f"Expected V={self.num_joints}, C={self.in_features}; got "
                f"V={coordinates.shape[-2]}, C={coordinates.shape[-1]}"
            )
        if observed_mask is None:
            raise ValueError(
                "observed_mask is required; zero coordinates cannot distinguish missing from valid zero"
            )
        mask = observed_mask.to(device=coordinates.device, dtype=torch.bool)
        if mask.shape != coordinates.shape[:-1]:
            raise ValueError(
                f"observed_mask must be [B,T,V]={tuple(coordinates.shape[:-1])}, "
                f"got {tuple(mask.shape)}"
            )
        clean = torch.nan_to_num(coordinates)
        clean = torch.where(mask.unsqueeze(-1), clean, torch.zeros_like(clean))
        return clean, mask

    def _compose_inputs(
        self,
        clean: Tensor,
        observed_mask: Tensor,
        auxiliary: Tensor | None,
        auxiliary_mask: Tensor | None,
    ) -> Tensor:
        pieces = [clean, observed_mask.unsqueeze(-1).to(clean.dtype)]
        if self.aux_features == 0:
            if (auxiliary is None) != (auxiliary_mask is None):
                raise ValueError("auxiliary and auxiliary_mask must be supplied together")
            if auxiliary is not None:
                expected = (*clean.shape[:-1], 0)
                if tuple(auxiliary.shape) != expected or tuple(auxiliary_mask.shape) != expected:
                    raise ValueError("This model was constructed without auxiliary features")
            return torch.cat(pieces, dim=-1)
        if auxiliary is None or auxiliary_mask is None:
            raise ValueError(
                f"Model expects {self.aux_features} auxiliary features and their explicit mask"
            )
        values = auxiliary.to(device=clean.device, dtype=clean.dtype)
        valid = auxiliary_mask.to(device=clean.device, dtype=torch.bool)
        expected = (*clean.shape[:-1], self.aux_features)
        if values.shape != expected or valid.shape != expected:
            raise ValueError(
                f"auxiliary and auxiliary_mask must be {expected}, got "
                f"{tuple(values.shape)} and {tuple(valid.shape)}"
            )
        values = torch.where(valid, torch.nan_to_num(values), torch.zeros_like(values))
        return torch.cat((pieces[0], pieces[1], values, valid.to(clean.dtype)), dim=-1)

    @staticmethod
    def _complete(observed: Tensor, mask: Tensor, prediction: Tensor) -> Tensor:
        return torch.where(mask.unsqueeze(-1), observed, prediction)


class MaskedMLPReconstructor(_MaskedReconstructor):
    """Per-frame spatial MLP baseline over all ten landmarks and mask bits."""

    def __init__(
        self,
        in_features: int = 2,
        num_joints: int = 10,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.1,
        aux_features: int = 0,
    ) -> None:
        super().__init__(in_features, num_joints, aux_features)
        if not hidden_dims or any(int(width) <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        input_width = self.num_joints * (self.in_features + 1 + 2 * self.aux_features)
        layers: list[nn.Module] = []
        width = input_width
        for hidden in hidden_dims:
            hidden = int(hidden)
            layers.extend(
                [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            )
            width = hidden
        self.encoder = nn.Sequential(*layers)
        self.embedding_dim = width
        self.decoder = nn.Linear(width, self.num_joints * self.in_features)

    def extract_features(
        self,
        coordinates: Tensor,
        observed_mask: Tensor,
        auxiliary: Tensor | None = None,
        auxiliary_mask: Tensor | None = None,
    ) -> Tensor:
        clean, mask = self._prepare(coordinates, observed_mask)
        inputs = self._compose_inputs(clean, mask, auxiliary, auxiliary_mask)
        return self.encoder(inputs.flatten(start_dim=2))

    def forward(
        self,
        coordinates: Tensor,
        observed_mask: Tensor,
        *,
        return_embedding: bool = False,
        track_valid_mask: Tensor | None = None,
        auxiliary: Tensor | None = None,
        auxiliary_mask: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        del track_valid_mask
        clean, mask = self._prepare(coordinates, observed_mask)
        embedding = self.extract_features(clean, mask, auxiliary, auxiliary_mask)
        prediction = self.decoder(embedding).reshape_as(clean)
        reconstruction = self._complete(clean, mask, prediction)
        return (reconstruction, embedding) if return_embedding else reconstruction


class TemporalReconstructionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        padding = dilation * (kernel_size - 1) // 2
        self.network = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(1, out_channels),
        )
        self.residual = (
            nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
        )
        self.activation = nn.GELU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.network(inputs) + self.residual(inputs))


class MaskedTCNReconstructor(_MaskedReconstructor):
    """Dilated temporal reconstructor for continuous, valid track windows only."""

    requires_valid_track_windows = True

    def __init__(
        self,
        in_features: int = 2,
        num_joints: int = 10,
        channels: Sequence[int] = (64, 96, 96),
        kernel_size: int = 3,
        dropout: float = 0.1,
        aux_features: int = 0,
    ) -> None:
        super().__init__(in_features, num_joints, aux_features)
        if not channels or any(int(width) <= 0 for width in channels):
            raise ValueError("channels must contain positive widths")
        blocks: list[nn.Module] = []
        width = self.num_joints * (self.in_features + 1 + 2 * self.aux_features)
        for index, channel in enumerate(channels):
            blocks.append(
                TemporalReconstructionBlock(
                    width,
                    int(channel),
                    dilation=2**index,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )
            width = int(channel)
        self.network = nn.Sequential(*blocks)
        self.embedding_dim = width
        self.decoder = nn.Conv1d(width, self.num_joints * self.in_features, kernel_size=1)

    @staticmethod
    def _check_track_windows(coordinates: Tensor, track_valid_mask: Tensor | None) -> None:
        if track_valid_mask is None:
            return
        valid = track_valid_mask.to(device=coordinates.device, dtype=torch.bool)
        if valid.ndim == 1:
            valid = valid[:, None].expand(-1, coordinates.shape[1])
        if valid.shape != coordinates.shape[:2]:
            raise ValueError(
                f"track_valid_mask must be [B] or [B,T]={tuple(coordinates.shape[:2])}, "
                f"got {tuple(valid.shape)}"
            )
        if not bool(valid.all()):
            raise ValueError("TCN reconstruction is restricted to continuous valid track windows")

    def extract_features(
        self,
        coordinates: Tensor,
        observed_mask: Tensor,
        track_valid_mask: Tensor | None = None,
        auxiliary: Tensor | None = None,
        auxiliary_mask: Tensor | None = None,
    ) -> Tensor:
        clean, mask = self._prepare(coordinates, observed_mask)
        self._check_track_windows(clean, track_valid_mask)
        inputs = self._compose_inputs(clean, mask, auxiliary, auxiliary_mask)
        encoded = self.network(inputs.flatten(start_dim=2).transpose(1, 2))
        return encoded.transpose(1, 2)

    def forward(
        self,
        coordinates: Tensor,
        observed_mask: Tensor,
        *,
        return_embedding: bool = False,
        track_valid_mask: Tensor | None = None,
        auxiliary: Tensor | None = None,
        auxiliary_mask: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        clean, mask = self._prepare(coordinates, observed_mask)
        embedding = self.extract_features(
            clean, mask, track_valid_mask, auxiliary, auxiliary_mask
        )
        prediction = self.decoder(embedding.transpose(1, 2)).transpose(1, 2)
        prediction = prediction.reshape_as(clean)
        reconstruction = self._complete(clean, mask, prediction)
        return (reconstruction, embedding) if return_embedding else reconstruction


class GraphTemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.spatial = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.temporal = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)
        )
        self.norm = nn.GroupNorm(1, out_channels)
        self.residual = (
            nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor, adjacency: Tensor) -> Tensor:
        messages = torch.einsum("bctv,vw->bctw", inputs, adjacency)
        output = self.norm(self.temporal(self.spatial(messages)))
        return self.dropout(self.activation(output + self.residual(inputs)))


class TrackIDGraphReconstructor(_MaskedReconstructor):
    """Compact graph-temporal model restricted to TrackID3x3's ten landmarks."""

    def __init__(
        self,
        in_features: int = 2,
        num_joints: int = 10,
        joint_names: Sequence[str] | None = None,
        edges: Iterable[tuple[int, int]] | Tensor | None = None,
        channels: Sequence[int] = (32, 64, 96),
        dropout: float = 0.1,
        aux_features: int = 0,
    ) -> None:
        super().__init__(in_features, num_joints, aux_features)
        names, selected_edges = validate_trackid3x3_anatomy(num_joints, joint_names, edges)
        self.joint_names = names
        self.edges = tuple((int(a), int(b)) for a, b in selected_edges)
        self.register_buffer(
            "adjacency", trackid3x3_adjacency(num_joints, self.joint_names, self.edges)
        )
        if not channels or any(int(width) <= 0 for width in channels):
            raise ValueError("channels must contain positive widths")
        blocks: list[nn.Module] = []
        width = self.in_features + 1 + 2 * self.aux_features
        for channel in channels:
            blocks.append(GraphTemporalBlock(width, int(channel), dropout))
            width = int(channel)
        self.blocks = nn.ModuleList(blocks)
        self.embedding_dim = width
        self.decoder = nn.Conv2d(width, self.in_features, kernel_size=1)

    def extract_features(
        self,
        coordinates: Tensor,
        observed_mask: Tensor,
        auxiliary: Tensor | None = None,
        auxiliary_mask: Tensor | None = None,
    ) -> Tensor:
        clean, mask = self._prepare(coordinates, observed_mask)
        features = self._compose_inputs(clean, mask, auxiliary, auxiliary_mask)
        output = features.permute(0, 3, 1, 2).contiguous()
        for block in self.blocks:
            output = block(output, self.adjacency)
        return output.permute(0, 2, 3, 1).contiguous()

    def forward(
        self,
        coordinates: Tensor,
        observed_mask: Tensor,
        *,
        return_embedding: bool = False,
        track_valid_mask: Tensor | None = None,
        auxiliary: Tensor | None = None,
        auxiliary_mask: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        del track_valid_mask
        clean, mask = self._prepare(coordinates, observed_mask)
        embedding = self.extract_features(clean, mask, auxiliary, auxiliary_mask)
        prediction = self.decoder(embedding.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        reconstruction = self._complete(clean, mask, prediction)
        return (reconstruction, embedding) if return_embedding else reconstruction


RECONSTRUCTION_MODEL_NAMES = ("mlp", "tcn", "graph")


def build_reconstruction_model(
    name: str,
    *,
    in_features: int = 2,
    num_joints: int = 10,
    joint_names: Sequence[str] | None = None,
    edges: Iterable[tuple[int, int]] | Tensor | None = None,
    aux_features: int = 0,
    **kwargs: Any,
) -> _MaskedReconstructor:
    """Build one of the masked coordinate reconstructors."""

    token = name.strip().lower().replace("-", "").replace("_", "")
    dropout = float(kwargs.get("dropout", 0.1))
    if token in {"mlp", "maskedmlp", "maskedmlpreconstructor"}:
        hidden = kwargs.get("hidden_dims", (256, 128))
        if isinstance(hidden, int):
            hidden = (hidden, max(hidden // 2, 16))
        return MaskedMLPReconstructor(
            in_features=in_features,
            num_joints=num_joints,
            hidden_dims=tuple(int(item) for item in hidden),
            dropout=dropout,
            aux_features=aux_features,
        )
    if token in {"tcn", "maskedtcn", "maskedtcnreconstructor"}:
        channels = kwargs.get("channels", (64, 96, 96))
        if isinstance(channels, int):
            channels = (channels, channels, channels)
        return MaskedTCNReconstructor(
            in_features=in_features,
            num_joints=num_joints,
            channels=tuple(int(item) for item in channels),
            kernel_size=int(kwargs.get("kernel_size", 3)),
            dropout=dropout,
            aux_features=aux_features,
        )
    if token in {"graph", "gcn", "trackidgraph", "trackidgraphreconstructor"}:
        channels = kwargs.get("channels", (32, 64, 96))
        if isinstance(channels, int):
            channels = (channels, channels, channels)
        model = TrackIDGraphReconstructor(
            in_features=in_features,
            num_joints=num_joints,
            joint_names=joint_names,
            edges=edges,
            channels=tuple(int(item) for item in channels),
            dropout=dropout,
            aux_features=aux_features,
        )
        parameters = sum(parameter.numel() for parameter in model.parameters())
        if parameters >= 1_000_000:
            raise ValueError(
                f"Graph reconstructor has {parameters:,} parameters; TrackID3x3 models must be <1M"
            )
        return model
    raise ValueError(
        f"Unknown reconstruction model {name!r}; choose {', '.join(RECONSTRUCTION_MODEL_NAMES)}"
    )


# Concise public aliases.
ReconstructionMLP = MaskedMLPReconstructor
ReconstructionTCN = MaskedTCNReconstructor
ReconstructionGraph = TrackIDGraphReconstructor


__all__ = [
    "RECONSTRUCTION_MODEL_NAMES",
    "TRACKID3X3_EDGES",
    "TRACKID3X3_JOINT_NAMES",
    "MaskedMLPReconstructor",
    "MaskedTCNReconstructor",
    "ReconstructionGraph",
    "ReconstructionMLP",
    "ReconstructionTCN",
    "TrackIDGraphReconstructor",
    "build_reconstruction_model",
    "trackid3x3_adjacency",
    "validate_trackid3x3_anatomy",
]
