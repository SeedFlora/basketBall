"""Compact recurrent baselines for temporal pose features."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _frame_mask(valid_mask: Tensor | None, x: Tensor) -> Tensor:
    if valid_mask is None:
        return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    mask = valid_mask.to(device=x.device, dtype=torch.bool)
    if mask.ndim > 2:
        mask = mask.flatten(2).any(dim=-1)
    if mask.shape != x.shape[:2]:
        raise ValueError(f"valid_mask has shape {tuple(valid_mask.shape)}, expected [B,T,...]")
    return mask


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM with mask-aware temporal pooling."""

    def __init__(
        self,
        in_features: int,
        num_classes: int = 3,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.25,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        if min(in_features, num_classes, hidden_dim, num_layers) <= 0:
            raise ValueError("model dimensions and num_layers must be positive")
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        directions = 2 if self.bidirectional else 1
        recurrent_dropout = dropout if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.in_features,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=self.bidirectional,
        )
        pooled_dim = directions * self.hidden_dim * 2
        self.embedding_dim = directions * self.hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(pooled_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes)

    def extract_features(self, x: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"BiLSTMClassifier expects [B,T,F], got {tuple(x.shape)}")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected F={self.in_features}, got F={x.shape[-1]}")
        mask = _frame_mask(valid_mask, x)
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        sequence, _ = self.lstm(x)
        weights = mask.unsqueeze(-1).to(dtype=sequence.dtype)
        mean = (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        neg_inf = torch.tensor(float("-inf"), dtype=sequence.dtype, device=sequence.device)
        maximum = sequence.masked_fill(~mask.unsqueeze(-1), neg_inf).amax(dim=1)
        maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        return self.projection(torch.cat((mean, maximum), dim=-1))

    def forward(
        self,
        x: Tensor,
        valid_mask: Tensor | None = None,
        return_embedding: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        embedding = self.extract_features(x, valid_mask)
        logits = self.classifier(embedding)
        return (logits, embedding) if return_embedding else logits


BiLSTM = BiLSTMClassifier
LSTMClassifier = BiLSTMClassifier
LSTMModel = BiLSTMClassifier
