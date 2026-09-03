"""Archived compact-classifier trainer retained for pre-pivot compatibility.

Public TrackID3x3 experiments use :mod:`src.training.reconstruction_trainer`.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.models import build_model, count_parameters
from src.utils import environment_info, storage_report
from src.utils import set_seed as _set_seed

from .losses import build_cross_entropy, class_counts
from .metrics import classification_metrics


@dataclass(slots=True)
class TrainerConfig:
    """Core optimization settings, mirroring ``config/default.yaml``."""

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 8
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-6
    mixed_precision: bool = True
    class_weighting: str | bool | Sequence[float] | None = "auto"
    num_workers: int = 2
    gradient_clip_norm: float | None = 5.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> TrainerConfig:
        source = dict(values or {})
        aliases = {
            "lr": "learning_rate",
            "patience": "early_stopping_patience",
            "amp": "mixed_precision",
        }
        for old, new in aliases.items():
            if old in source and new not in source:
                source[new] = source[old]
        valid = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in source.items() if key in valid})

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.early_stopping_patience < 1 or self.scheduler_patience < 0:
            raise ValueError("early stopping patience must be positive")
        if not 0 < self.scheduler_factor < 1:
            raise ValueError("scheduler_factor must be between zero and one")


@dataclass(slots=True)
class TrainingResult:
    metrics: dict[str, Any]
    history: list[dict[str, Any]]
    checkpoint_path: Path
    run_dir: Path


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Public reproducibility helper for Python, NumPy, PyTorch, and CUDA."""

    _set_seed(seed, deterministic=deterministic)


def seed_worker(worker_id: int) -> None:
    """Seed NumPy/Python workers from PyTorch's deterministic worker seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(data), handle, indent=2, sort_keys=True)


def _write_yaml(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_json_safe(data), handle, sort_keys=False)


def _write_history(history: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    fields = list(history[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def _unpack_batch(batch: Any) -> tuple[Tensor, Tensor, Tensor | None]:
    """Accept conventional tuple batches or named dictionary batches."""

    if isinstance(batch, Mapping):
        inputs = next(
            (batch[key] for key in ("inputs", "x", "features", "skeleton", "keypoints") if key in batch),
            None,
        )
        targets = next(
            (batch[key] for key in ("labels", "label", "targets", "target", "y") if key in batch),
            None,
        )
        mask = next(
            (batch[key] for key in ("valid_mask", "mask", "masks") if key in batch),
            None,
        )
        if inputs is None or targets is None:
            raise KeyError("Batch dictionaries need inputs/features and labels/targets")
    elif isinstance(batch, (tuple, list)) and len(batch) in {2, 3}:
        inputs, targets = batch[:2]
        mask = batch[2] if len(batch) == 3 else None
    else:
        raise TypeError("Batches must be (inputs, labels[, mask]) tuples or dictionaries")
    return torch.as_tensor(inputs), torch.as_tensor(targets), None if mask is None else torch.as_tensor(mask)


def _dataset_labels(loader: DataLoader) -> Tensor:
    dataset = loader.dataset
    for attribute in ("labels", "targets", "y"):
        if hasattr(dataset, attribute):
            return torch.as_tensor(getattr(dataset, attribute), dtype=torch.long).flatten()
    labels: list[Tensor] = []
    for batch in loader:
        _, targets, _ = _unpack_batch(batch)
        labels.append(targets.to(dtype=torch.long).flatten().cpu())
    if not labels:
        raise ValueError("Training loader is empty")
    return torch.cat(labels)


def _make_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # PyTorch 2.2 compatibility
        return torch.cuda.amp.GradScaler(enabled=enabled)


class Trainer:
    """Train and retain exactly one checkpoint: the best validation macro-F1."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig | Mapping[str, Any] | None = None,
        *,
        device: str | torch.device | None = None,
        num_classes: int = 3,
        class_names: Sequence[str] = ("dribble", "lay-up", "shoot"),
    ) -> None:
        self.model = model
        self.config = config if isinstance(config, TrainerConfig) else TrainerConfig.from_mapping(config)
        self.config.validate()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.num_classes = int(num_classes)
        self.class_names = list(class_names)
        if len(self.class_names) != self.num_classes:
            raise ValueError("class_names length must equal num_classes")
        self.model.to(self.device)
        self.amp_enabled = bool(self.config.mixed_precision and self.device.type == "cuda")

    def _epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: AdamW | None,
        scaler: Any | None,
    ) -> tuple[float, dict[str, object]]:
        training = optimizer is not None
        self.model.train(training)
        total_loss = 0.0
        sample_count = 0
        targets_all: list[Tensor] = []
        predictions_all: list[Tensor] = []
        for batch in loader:
            inputs, targets, mask = _unpack_batch(batch)
            inputs = inputs.to(self.device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(self.device, dtype=torch.long, non_blocking=True).flatten()
            mask = None if mask is None else mask.to(self.device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.amp_enabled,
                    dtype=torch.float16,
                ):
                    output = self.model(inputs, valid_mask=mask)
                    logits = output[0] if isinstance(output, tuple) else output
                    loss = criterion(logits, targets)
                if training:
                    assert scaler is not None
                    scaler.scale(loss).backward()
                    if self.config.gradient_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
            batch_size = int(targets.numel())
            total_loss += float(loss.detach()) * batch_size
            sample_count += batch_size
            targets_all.append(targets.detach().cpu())
            predictions_all.append(logits.detach().argmax(dim=-1).cpu())
        if sample_count == 0:
            raise ValueError("A training or validation loader is empty")
        metrics = classification_metrics(
            torch.cat(targets_all),
            torch.cat(predictions_all),
            num_classes=self.num_classes,
            class_names=self.class_names,
        )
        return total_loss / sample_count, metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        run_dir: str | Path,
        checkpoint_path: str | Path | None = None,
        seed: int = 42,
        train_labels: Sequence[int] | np.ndarray | Tensor | None = None,
        model_name: str | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        resolved_config: Mapping[str, Any] | None = None,
        synthetic: bool = False,
        project_root: str | Path = ".",
    ) -> TrainingResult:
        seed_everything(seed)
        output_dir = Path(run_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        best_path = Path(checkpoint_path) if checkpoint_path else output_dir / "best.pt"
        best_path.parent.mkdir(parents=True, exist_ok=True)

        labels = (
            torch.as_tensor(train_labels, dtype=torch.long).flatten()
            if train_labels is not None
            else _dataset_labels(train_loader)
        )
        criterion, weights = build_cross_entropy(
            labels,
            self.num_classes,
            self.config.class_weighting,
            device=self.device,
        )
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            min_lr=self.config.min_learning_rate,
        )
        scaler = _make_grad_scaler(self.amp_enabled)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        config_payload = dict(resolved_config or {})
        if not config_payload:
            config_payload = {"project": {"seed": seed}, "training": asdict(self.config)}
        _write_yaml(config_payload, output_dir / "config.yaml")
        _write_json(environment_info(project_root), output_dir / "environment.json")

        history: list[dict[str, Any]] = []
        best_score = float("-inf")
        best_epoch = 0
        stale_epochs = 0
        started = time.perf_counter()
        for epoch in range(1, self.config.epochs + 1):
            set_epoch = getattr(train_loader.dataset, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch - 1)
            train_loss, train_metrics = self._epoch(train_loader, criterion, optimizer, scaler)
            with torch.inference_mode():
                val_loss, val_metrics = self._epoch(val_loader, criterion, None, None)
            score = float(val_metrics["macro_f1"])
            scheduler.step(score)
            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "train_loss": train_loss,
                    "train_accuracy": train_metrics["accuracy"],
                    "train_macro_f1": train_metrics["macro_f1"],
                    "val_loss": val_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                }
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                stale_epochs = 0
                checkpoint = {
                    "format_version": 1,
                    "model_name": model_name or type(self.model).__name__,
                    "model_kwargs": _json_safe(dict(model_kwargs or {})),
                    "model_state_dict": self.model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "class_names": self.class_names,
                    "config": _json_safe(config_payload),
                    "synthetic": bool(synthetic),
                }
                temporary = best_path.with_suffix(best_path.suffix + ".tmp")
                torch.save(checkpoint, temporary)
                os.replace(temporary, best_path)
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.early_stopping_patience:
                    break

        training_seconds = time.perf_counter() - started
        peak_memory = (
            int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else None
        )
        loaded = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(loaded["model_state_dict"])
        with torch.inference_mode():
            best_val_loss, best_val_metrics = self._epoch(val_loader, criterion, None, None)

        counts = class_counts(labels, self.num_classes)
        final_metrics: dict[str, Any] = {
            "synthetic": bool(synthetic),
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "early_stopped": len(history) < self.config.epochs,
            "best_validation_loss": best_val_loss,
            "best_validation": best_val_metrics,
            "parameter_count": count_parameters(self.model),
            "training_time_seconds": training_seconds,
            "peak_gpu_memory_bytes": peak_memory,
            "checkpoint_size_bytes": best_path.stat().st_size,
            "device": str(self.device),
            "mixed_precision_used": self.amp_enabled,
            "training_class_counts": counts.tolist(),
            "training_class_weights": None if weights is None else weights.tolist(),
            "checkpoint": str(best_path),
            "storage": storage_report(project_root),
        }
        _write_history(history, output_dir / "history.csv")
        _write_json(final_metrics, output_dir / "metrics.json")
        return TrainingResult(final_metrics, history, best_path, output_dir)


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device | None = None,
    *,
    return_embeddings: bool = False,
) -> dict[str, Tensor]:
    """Run inference and return CPU logits, labels, predictions, and embeddings."""

    selected = torch.device(device or next(model.parameters()).device)
    model.to(selected).eval()
    logits_all: list[Tensor] = []
    labels_all: list[Tensor] = []
    embeddings_all: list[Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            inputs, labels, mask = _unpack_batch(batch)
            inputs = inputs.to(selected, dtype=torch.float32)
            mask = None if mask is None else mask.to(selected)
            output = model(inputs, valid_mask=mask, return_embedding=return_embeddings)
            if return_embeddings:
                logits, embeddings = output
                embeddings_all.append(embeddings.cpu())
            else:
                logits = output
            logits_all.append(logits.cpu())
            labels_all.append(labels.to(dtype=torch.long).flatten().cpu())
    if not logits_all:
        raise ValueError("Inference loader is empty")
    logits_tensor = torch.cat(logits_all)
    result = {
        "logits": logits_tensor,
        "labels": torch.cat(labels_all),
        "predictions": logits_tensor.argmax(dim=-1),
    }
    if return_embeddings:
        result["embeddings"] = torch.cat(embeddings_all)
    return result


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device | None = None,
    *,
    num_classes: int = 3,
    class_names: Sequence[str] = ("dribble", "lay-up", "shoot"),
) -> dict[str, object]:
    outputs = predict(model, loader, device)
    return classification_metrics(
        outputs["labels"], outputs["predictions"], num_classes, class_names
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a best checkpoint and reconstruct the model when metadata permits."""

    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"{path} is not a supported basketball-pose checkpoint")
    if checkpoint.get("task") == "keypoint_reconstruction":
        from src.models import build_reconstruction_model

        if model is None:
            model = build_reconstruction_model(
                checkpoint["model_name"], **checkpoint["model_kwargs"]
            )
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        model.to(map_location)
        return model, checkpoint
    if model is None:
        name = checkpoint.get("model_name")
        kwargs = checkpoint.get("model_kwargs") or {}
        if not name or not kwargs:
            raise ValueError("Checkpoint lacks model_name/model_kwargs; pass an initialized model")
        model = build_model(name, **kwargs)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model.to(map_location)
    return model, checkpoint


build_model_from_checkpoint = load_checkpoint
set_seed = seed_everything


__all__ = [
    "Trainer",
    "TrainerConfig",
    "TrainingResult",
    "build_model_from_checkpoint",
    "evaluate_model",
    "load_checkpoint",
    "predict",
    "seed_everything",
    "seed_worker",
    "set_seed",
]
