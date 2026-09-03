"""A compact spatial-temporal graph convolutional network."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import Tensor, nn

COCO_17_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
)

MEDIAPIPE_33_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19),
    (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20),
    (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
    (25, 27), (27, 29), (29, 31), (27, 31), (24, 26), (26, 28),
    (28, 30), (30, 32), (28, 32),
)

# NTU RGB+D 25-joint layout, converted from the common one-indexed definition.
NTU_25_EDGES: tuple[tuple[int, int], ...] = tuple(
    (a - 1, b - 1)
    for a, b in (
        (1, 2), (2, 21), (21, 3), (3, 4), (21, 5), (5, 6), (6, 7),
        (7, 8), (8, 22), (8, 23), (21, 9), (9, 10), (10, 11), (11, 12),
        (12, 24), (12, 25), (1, 13), (13, 14), (14, 15), (15, 16),
        (1, 17), (17, 18), (18, 19), (19, 20),
    )
)


def anatomical_edges(num_joints: int) -> tuple[tuple[int, int], ...]:
    """Return standard anatomical edges, with a deterministic fallback."""

    if num_joints == 17:
        return COCO_17_EDGES
    if num_joints == 25:
        return NTU_25_EDGES
    if num_joints == 33:
        return MEDIAPIPE_33_EDGES
    if num_joints <= 0:
        raise ValueError("num_joints must be positive")
    # Unknown layouts must normally provide dataset edges. The chain keeps small
    # synthetic fixtures usable without pretending to know their joint semantics.
    return tuple((index, index + 1) for index in range(num_joints - 1))


def normalized_adjacency(
    num_joints: int,
    edges: Iterable[tuple[int, int]] | Tensor | None = None,
) -> Tensor:
    """Build symmetric, self-connected ``D^-1/2 A D^-1/2`` adjacency."""

    if num_joints <= 0:
        raise ValueError("num_joints must be positive")
    selected = anatomical_edges(num_joints) if edges is None else edges
    if isinstance(selected, Tensor):
        selected = selected.detach().cpu().tolist()
    adjacency = torch.eye(num_joints, dtype=torch.float32)
    for edge in selected:
        if len(edge) != 2:
            raise ValueError(f"Every edge must contain two indices, got {edge!r}")
        source, target = int(edge[0]), int(edge[1])
        if not (0 <= source < num_joints and 0 <= target < num_joints):
            raise ValueError(f"Edge {(source, target)} is outside 0..{num_joints - 1}")
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


class STGCNBlock(nn.Module):
    """Spatial message passing followed by temporal convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel: int = 5,
        stride: int = 1,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be odd")
        self.spatial_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.temporal = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(temporal_kernel, 1),
            stride=(stride, 1),
            padding=(temporal_kernel // 2, 0),
        )
        self.norm = nn.BatchNorm2d(out_channels)
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        spatial = torch.einsum("bctv,vw->bctw", x, adjacency)
        output = self.norm(self.temporal(self.spatial_projection(spatial)))
        return self.dropout(self.activation(output + self.residual(x)))


class STGCNClassifier(nn.Module):
    """Small ST-GCN accepting skeleton tensors in ``[B, C, T, V]`` order."""

    def __init__(
        self,
        in_features: int,
        num_joints: int,
        num_classes: int = 3,
        edges: Iterable[tuple[int, int]] | Tensor | None = None,
        channels: Sequence[int] = (48, 64, 96, 128),
        temporal_kernel: int = 5,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if min(in_features, num_joints, num_classes) <= 0 or not channels:
            raise ValueError("in_features, num_joints, num_classes and channels must be positive")
        self.in_features = int(in_features)
        self.num_joints = int(num_joints)
        self.num_classes = int(num_classes)
        self.register_buffer("adjacency", normalized_adjacency(self.num_joints, edges))
        self.input_norm = nn.BatchNorm2d(self.in_features)
        blocks: list[nn.Module] = []
        width = self.in_features
        for index, channel in enumerate(channels):
            # One stride-2 block is enough for T=32 and keeps the model efficient.
            stride = 2 if index == 2 else 1
            blocks.append(
                STGCNBlock(
                    width,
                    int(channel),
                    temporal_kernel=temporal_kernel,
                    stride=stride,
                    dropout=dropout,
                )
            )
            width = int(channel)
        self.blocks = nn.ModuleList(blocks)
        self.embedding_dim = width
        self.classifier = nn.Linear(width, self.num_classes)

    def _node_mask(self, valid_mask: Tensor | None, x: Tensor) -> Tensor:
        batch, _, frames, joints = x.shape
        if valid_mask is None:
            return torch.ones((batch, frames, joints), dtype=torch.bool, device=x.device)
        mask = valid_mask.to(device=x.device, dtype=torch.bool)
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1).expand(-1, -1, joints)
        elif mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        elif mask.ndim == 4 and mask.shape[:3] == (batch, frames, joints):
            mask = mask.any(dim=-1)
        if mask.shape != (batch, frames, joints):
            raise ValueError(
                f"valid_mask must be [B,T,V]={batch, frames, joints}, got {tuple(mask.shape)}"
            )
        return mask

    def extract_features(self, x: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"STGCNClassifier expects [B,C,T,V], got {tuple(x.shape)}")
        if x.shape[1] != self.in_features or x.shape[3] != self.num_joints:
            raise ValueError(
                f"Expected C={self.in_features}, V={self.num_joints}; got C={x.shape[1]}, V={x.shape[3]}"
            )
        mask = self._node_mask(valid_mask, x)
        output = self.input_norm(x * mask.unsqueeze(1).to(dtype=x.dtype))
        for block in self.blocks:
            output = block(output, self.adjacency)
        # Temporal downsampling changes the mask length; nearest preserves validity.
        if output.shape[2] != mask.shape[1]:
            resized = torch.nn.functional.interpolate(
                mask.permute(0, 2, 1).to(dtype=output.dtype),
                size=output.shape[2],
                mode="nearest",
            )
            mask = resized.permute(0, 2, 1).to(dtype=torch.bool)
        weights = mask.unsqueeze(1).to(dtype=output.dtype)
        pooled = (output * weights).sum(dim=(2, 3)) / weights.sum(dim=(2, 3)).clamp_min(1.0)
        return pooled

    def forward(
        self,
        x: Tensor,
        valid_mask: Tensor | None = None,
        return_embedding: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        embedding = self.extract_features(x, valid_mask)
        logits = self.classifier(embedding)
        return (logits, embedding) if return_embedding else logits


STGCN = STGCNClassifier
CompactSTGCN = STGCNClassifier
STGCNModel = STGCNClassifier
