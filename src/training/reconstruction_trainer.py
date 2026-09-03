"""Resource-aware trainer for masked keypoint coordinate reconstruction."""

from __future__ import annotations

import csv
import json
import os
import time
from collections.abc import Mapping
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

from src.data.storage import current_storage_budget, directory_bytes
from src.models import build_reconstruction_model, count_parameters
from src.utils import environment_info, set_seed

from .losses import MaskedReconstructionLoss
from .metrics import reconstruction_metrics
from .representations import inverse_reconstruction_coordinates


def reconstruction_storage_report(project_root: str | Path = ".") -> dict[str, int | float]:
    """Return non-overlapping TrackID-aware byte counts and budget limits."""

    root = Path(project_root).resolve()
    budget = current_storage_budget(root)
    checkpoint_bytes = directory_bytes(
        root / "outputs", suffixes={".pt", ".pth", ".ckpt"}
    )
    return {
        "raw_dataset_bytes": round(budget["current_raw_data_mb"] * 1_000_000),
        "processed_dataset_bytes": round(
            budget["current_processed_data_mb"] * 1_000_000
        ),
        "model_checkpoint_bytes": int(checkpoint_bytes),
        "total_project_artifact_bytes": round(budget["total_mb"] * 1_000_000),
        "output_data_bytes": round(budget["current_output_data_mb"] * 1_000_000),
        "source_overhead_bytes": round(budget["source_overhead_mb"] * 1_000_000),
        "temporary_data_bytes": round(budget["temporary_data_mb"] * 1_000_000),
        "remaining_budget_mb": float(budget["remaining_budget_mb"]),
        "max_storage_mb": float(budget["max_storage_mb"]),
    }


@dataclass(slots=True)
class ReconstructionTrainerConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 8
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-6
    mixed_precision: bool = True
    num_workers: int = 2
    gradient_clip_norm: float | None = 5.0
    loss: str = "smooth_l1"
    huber_beta: float = 0.05
    selection_metric: str = "nmje"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> ReconstructionTrainerConfig:
        source = dict(values or {})
        aliases = {"lr": "learning_rate", "patience": "early_stopping_patience", "amp": "mixed_precision"}
        for old, new in aliases.items():
            if old in source and new not in source:
                source[new] = source[old]
        fields = cls.__dataclass_fields__.keys()
        result = cls(**{key: value for key, value in source.items() if key in fields})
        result.validate()
        return result

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.early_stopping_patience < 1 or self.scheduler_patience < 0:
            raise ValueError("early stopping patience must be positive")
        if not 0 < self.scheduler_factor < 1:
            raise ValueError("scheduler_factor must be between zero and one")
        if self.huber_beta <= 0:
            raise ValueError("huber_beta must be positive")
        if self.selection_metric not in {"mpjpe", "rmse", "mae", "nmje"}:
            raise ValueError("selection_metric must be mpjpe, rmse, mae, or nmje")


@dataclass(slots=True)
class ReconstructionTrainingResult:
    metrics: dict[str, Any]
    history: list[dict[str, Any]]
    checkpoint_path: Path
    run_dir: Path


@dataclass(slots=True)
class ReconstructionBatch:
    inputs: Tensor
    targets: Tensor
    observed_mask: Tensor
    original_valid_mask: Tensor
    synthetic_mask: Tensor
    pixel_targets: Tensor | None = None
    inverse_offset: Tensor | None = None
    inverse_scale: Tensor | None = None
    auxiliary: Tensor | None = None
    auxiliary_mask: Tensor | None = None
    bbox_diagonal: Tensor | None = None
    track_valid_mask: Tensor | None = None


def _first(batch: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in batch:
            return batch[name]
    return None


def unpack_reconstruction_batch(batch: Any) -> ReconstructionBatch:
    """Adapt explicit reconstruction dictionaries or a seven-item tuple."""

    if isinstance(batch, Mapping):
        inputs = _first(batch, ("inputs", "masked_keypoints", "masked_coordinates"))
        targets = _first(batch, ("targets", "target_keypoints", "coordinates", "keypoints"))
        original = _first(batch, ("original_valid_mask", "valid_mask"))
        synthetic = _first(
            batch,
            ("synthetic_mask", "masked_target_mask", "target_mask", "reconstruction_mask"),
        )
        observed = _first(batch, ("observed_mask", "input_mask"))
        pixel_targets = _first(batch, ("pixel_targets", "pixel_coordinates", "pixel_keypoints"))
        inverse_offset = _first(batch, ("inverse_offset", "coordinate_offset"))
        inverse_scale = _first(batch, ("inverse_scale", "coordinate_scale"))
        auxiliary = _first(batch, ("auxiliary", "auxiliary_features", "angles"))
        auxiliary_mask = _first(batch, ("auxiliary_mask", "angle_mask"))
        track_valid = _first(batch, ("track_valid_mask", "window_valid_mask"))
        bbox_diagonal = _first(batch, ("bbox_diagonal", "bbox_diag", "bbox_scale"))
        bbox_xywh = _first(batch, ("bbox_xywh", "bboxes_xywh"))
        if bbox_diagonal is None and bbox_xywh is not None:
            boxes = torch.as_tensor(bbox_xywh)
            if boxes.shape[-1] != 4:
                raise ValueError("bbox_xywh must end in four values")
            bbox_diagonal = torch.linalg.vector_norm(boxes[..., 2:4], dim=-1)
    elif isinstance(batch, (tuple, list)) and 5 <= len(batch) <= 7:
        inputs, targets, observed, original, synthetic = batch[:5]
        pixel_targets = None
        inverse_offset = None
        inverse_scale = None
        auxiliary = None
        auxiliary_mask = None
        bbox_diagonal = batch[5] if len(batch) >= 6 else None
        track_valid = batch[6] if len(batch) == 7 else None
    else:
        raise TypeError(
            "Reconstruction batches must be mappings with explicit masks or "
            "(inputs, targets, observed, original, synthetic[, bbox, track_valid])"
        )
    if any(value is None for value in (inputs, targets, original, synthetic)):
        raise KeyError(
            "Reconstruction batches require inputs, targets, original_valid_mask, and synthetic_mask"
        )
    original_tensor = torch.as_tensor(original, dtype=torch.bool)
    synthetic_tensor = torch.as_tensor(synthetic, dtype=torch.bool)
    observed_tensor = (
        original_tensor & ~synthetic_tensor
        if observed is None
        else torch.as_tensor(observed, dtype=torch.bool)
    )
    if bool((synthetic_tensor & ~original_tensor).any()):
        raise ValueError("synthetic_mask selects originally missing keypoints")
    if not torch.equal(observed_tensor, original_tensor & ~synthetic_tensor):
        raise ValueError("observed_mask must equal original_valid_mask & ~synthetic_mask")
    if (inverse_offset is None) != (inverse_scale is None):
        raise ValueError("inverse_offset and inverse_scale must be supplied together")
    if (auxiliary is None) != (auxiliary_mask is None):
        raise ValueError("auxiliary and auxiliary_mask must be supplied together")
    return ReconstructionBatch(
        inputs=torch.as_tensor(inputs),
        targets=torch.as_tensor(targets),
        observed_mask=observed_tensor,
        original_valid_mask=original_tensor,
        synthetic_mask=synthetic_tensor,
        pixel_targets=None if pixel_targets is None else torch.as_tensor(pixel_targets),
        inverse_offset=None if inverse_offset is None else torch.as_tensor(inverse_offset),
        inverse_scale=None if inverse_scale is None else torch.as_tensor(inverse_scale),
        auxiliary=None if auxiliary is None else torch.as_tensor(auxiliary),
        auxiliary_mask=None if auxiliary_mask is None else torch.as_tensor(auxiliary_mask),
        bbox_diagonal=None if bbox_diagonal is None else torch.as_tensor(bbox_diagonal),
        track_valid_mask=None if track_valid is None else torch.as_tensor(track_valid),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (Tensor, np.ndarray)):
        return value.detach().cpu().tolist() if isinstance(value, Tensor) else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_history(history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(data), handle, indent=2, sort_keys=True)


def _write_yaml(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_json_safe(data), handle, sort_keys=False)


def _grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class ReconstructionTrainer:
    """Train against synthetic masks and retain one best compact checkpoint."""

    def __init__(
        self,
        model: nn.Module,
        config: ReconstructionTrainerConfig | Mapping[str, Any] | None = None,
        *,
        device: str | torch.device | None = None,
        joint_names: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.model = model
        self.config = (
            config
            if isinstance(config, ReconstructionTrainerConfig)
            else ReconstructionTrainerConfig.from_mapping(config)
        )
        self.config.validate()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.model.to(self.device)
        self.amp_enabled = bool(self.config.mixed_precision and self.device.type == "cuda")
        self.criterion = MaskedReconstructionLoss(self.config.loss, self.config.huber_beta)
        self.joint_names = None if joint_names is None else tuple(str(name) for name in joint_names)

    def _epoch(
        self,
        loader: DataLoader,
        optimizer: AdamW | None,
        scaler: Any | None,
    ) -> tuple[float, dict[str, object]]:
        training = optimizer is not None
        self.model.train(training)
        total_loss = 0.0
        scored_coordinates = 0
        predictions: list[Tensor] = []
        targets: list[Tensor] = []
        originals: list[Tensor] = []
        synthetics: list[Tensor] = []
        bbox_scales: list[Tensor] = []
        all_have_scale = True
        for raw_batch in loader:
            batch = unpack_reconstruction_batch(raw_batch)
            inputs = batch.inputs.to(self.device, dtype=torch.float32, non_blocking=True)
            target = batch.targets.to(self.device, dtype=torch.float32, non_blocking=True)
            observed = batch.observed_mask.to(self.device, non_blocking=True)
            original = batch.original_valid_mask.to(self.device, non_blocking=True)
            synthetic = batch.synthetic_mask.to(self.device, non_blocking=True)
            track_valid = (
                None
                if batch.track_valid_mask is None
                else batch.track_valid_mask.to(self.device, non_blocking=True)
            )
            auxiliary = (
                None
                if batch.auxiliary is None
                else batch.auxiliary.to(self.device, dtype=torch.float32, non_blocking=True)
            )
            auxiliary_mask = (
                None
                if batch.auxiliary_mask is None
                else batch.auxiliary_mask.to(self.device, non_blocking=True)
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.amp_enabled,
                ):
                    output = self.model(
                        inputs,
                        observed,
                        track_valid_mask=track_valid,
                        auxiliary=auxiliary,
                        auxiliary_mask=auxiliary_mask,
                    )
                    reconstructed = output[0] if isinstance(output, tuple) else output
                    loss = self.criterion(reconstructed, target, original, synthetic)
                if training:
                    assert scaler is not None
                    scaler.scale(loss).backward()
                    if self.config.gradient_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clip_norm
                        )
                    scaler.step(optimizer)
                    scaler.update()
            coordinate_count = int(synthetic.sum()) * int(target.shape[-1])
            total_loss += float(loss.detach()) * coordinate_count
            scored_coordinates += coordinate_count
            if batch.inverse_offset is not None:
                pixel_prediction = inverse_reconstruction_coordinates(
                    reconstructed,
                    batch.inverse_offset.to(self.device, dtype=torch.float32, non_blocking=True),
                    batch.inverse_scale.to(self.device, dtype=torch.float32, non_blocking=True),
                )
                pixel_target = (
                    inverse_reconstruction_coordinates(
                        target,
                        batch.inverse_offset.to(self.device, dtype=torch.float32, non_blocking=True),
                        batch.inverse_scale.to(self.device, dtype=torch.float32, non_blocking=True),
                    )
                    if batch.pixel_targets is None
                    else batch.pixel_targets.to(
                        self.device, dtype=torch.float32, non_blocking=True
                    )
                )
            else:
                pixel_prediction = reconstructed
                pixel_target = (
                    target
                    if batch.pixel_targets is None
                    else batch.pixel_targets.to(
                        self.device, dtype=torch.float32, non_blocking=True
                    )
                )
            predictions.append(pixel_prediction.detach().cpu())
            targets.append(pixel_target.detach().cpu())
            originals.append(original.detach().cpu())
            synthetics.append(synthetic.detach().cpu())
            if batch.bbox_diagonal is None:
                all_have_scale = False
            else:
                bbox_scales.append(batch.bbox_diagonal.detach().cpu())
        if scored_coordinates == 0:
            raise ValueError("Loader contains no synthetically masked valid coordinates")
        metrics = reconstruction_metrics(
            torch.cat(predictions),
            torch.cat(targets),
            torch.cat(originals),
            torch.cat(synthetics),
            bbox_diagonal=torch.cat(bbox_scales) if all_have_scale and bbox_scales else None,
            joint_names=self.joint_names,
        )
        return total_loss / scored_coordinates, metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        run_dir: str | Path,
        checkpoint_path: str | Path | None = None,
        seed: int = 42,
        model_name: str,
        model_kwargs: Mapping[str, Any],
        resolved_config: Mapping[str, Any] | None = None,
        experiment_metadata: Mapping[str, Any] | None = None,
        synthetic: bool = False,
        project_root: str | Path = ".",
    ) -> ReconstructionTrainingResult:
        set_seed(seed)
        output_dir = Path(run_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        best_path = Path(checkpoint_path) if checkpoint_path else output_dir / "best.pt"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(resolved_config or {})
        if not payload:
            payload = {"project": {"seed": seed}, "training": asdict(self.config)}
        _write_yaml(payload, output_dir / "config.yaml")
        _write_json(environment_info(project_root), output_dir / "environment.json")

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            min_lr=self.config.min_learning_rate,
        )
        scaler = _grad_scaler(self.amp_enabled)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        history: list[dict[str, Any]] = []
        best_score = float("inf")
        best_epoch = 0
        stale_epochs = 0
        started = time.perf_counter()
        for epoch in range(1, self.config.epochs + 1):
            setter = getattr(train_loader.dataset, "set_epoch", None)
            if callable(setter):
                setter(epoch - 1)
            train_loss, train_metrics = self._epoch(train_loader, optimizer, scaler)
            with torch.inference_mode():
                val_loss, val_metrics = self._epoch(val_loader, None, None)
            selection = val_metrics.get(self.config.selection_metric)
            if selection is None:
                raise ValueError(
                    f"selection_metric={self.config.selection_metric!r} is unavailable; "
                    "bbox diagonal is required for nmje"
                )
            score = float(selection)
            if not np.isfinite(score):
                raise ValueError(
                    f"Validation {self.config.selection_metric} is non-finite at epoch {epoch}"
                )
            scheduler.step(score)
            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "train_loss": train_loss,
                    "train_mae": train_metrics["mae"],
                    "train_rmse": train_metrics["rmse"],
                    "train_mpjpe": train_metrics["mpjpe"],
                    "train_nmje": train_metrics["nmje"],
                    "train_pck@0.10": train_metrics["pck@0.10"],
                    "val_loss": val_loss,
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                    "val_mpjpe": val_metrics["mpjpe"],
                    "val_nmje": val_metrics["nmje"],
                    "val_pck@0.10": val_metrics["pck@0.10"],
                }
            )
            if score < best_score:
                best_score = score
                best_epoch = epoch
                stale_epochs = 0
                experiment = dict(experiment_metadata or {})
                checkpoint = {
                    "format_version": 2,
                    "task": "keypoint_reconstruction",
                    "model_name": model_name,
                    "model_kwargs": _json_safe(dict(model_kwargs)),
                    "model_state_dict": self.model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "selection_metric": self.config.selection_metric,
                    "config": _json_safe(payload),
                    "experiment": _json_safe(experiment),
                    "representation": experiment.get("representation"),
                    "mask_ratio": experiment.get("mask_ratio"),
                    "domain": experiment.get("domain"),
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
        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        with torch.inference_mode():
            best_val_loss, best_val_metrics = self._epoch(val_loader, None, None)
        metrics: dict[str, Any] = {
            "task": "keypoint_reconstruction",
            "synthetic": bool(synthetic),
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "early_stopped": len(history) < self.config.epochs,
            "selection_metric": self.config.selection_metric,
            "best_validation_loss": best_val_loss,
            "best_validation": best_val_metrics,
            "parameter_count": count_parameters(self.model),
            "training_time_seconds": training_seconds,
            "peak_gpu_memory_bytes": peak_memory,
            "checkpoint_size_bytes": best_path.stat().st_size,
            "checkpoint": str(best_path),
            "device": str(self.device),
            "mixed_precision_used": self.amp_enabled,
            "experiment": dict(experiment_metadata or {}),
            "storage": reconstruction_storage_report(project_root),
        }
        _write_history(history, output_dir / "history.csv")
        _write_json(metrics, output_dir / "metrics.json")
        return ReconstructionTrainingResult(metrics, history, best_path, output_dir)


def predict_reconstruction(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device | None = None,
) -> dict[str, Tensor | None]:
    """Return reconstruction tensors while retaining both missingness masks."""

    selected_device = torch.device(device or next(model.parameters()).device)
    model.to(selected_device).eval()
    collected: dict[str, list[Tensor]] = {
        "predictions": [],
        "targets": [],
        "represented_predictions": [],
        "represented_targets": [],
        "original_valid_mask": [],
        "synthetic_mask": [],
        "observed_mask": [],
        "bbox_diagonal": [],
    }
    all_have_scale = True
    with torch.inference_mode():
        for raw_batch in loader:
            batch = unpack_reconstruction_batch(raw_batch)
            output = model(
                batch.inputs.to(selected_device, dtype=torch.float32),
                batch.observed_mask.to(selected_device),
                track_valid_mask=(
                    None
                    if batch.track_valid_mask is None
                    else batch.track_valid_mask.to(selected_device)
                ),
                auxiliary=(
                    None
                    if batch.auxiliary is None
                    else batch.auxiliary.to(selected_device, dtype=torch.float32)
                ),
                auxiliary_mask=(
                    None
                    if batch.auxiliary_mask is None
                    else batch.auxiliary_mask.to(selected_device)
                ),
            )
            prediction = output[0] if isinstance(output, tuple) else output
            represented_target = batch.targets.to(selected_device, dtype=torch.float32)
            if batch.inverse_offset is not None:
                pixel_prediction = inverse_reconstruction_coordinates(
                    prediction,
                    batch.inverse_offset.to(selected_device, dtype=torch.float32),
                    batch.inverse_scale.to(selected_device, dtype=torch.float32),
                )
                pixel_target = (
                    inverse_reconstruction_coordinates(
                        represented_target,
                        batch.inverse_offset.to(selected_device, dtype=torch.float32),
                        batch.inverse_scale.to(selected_device, dtype=torch.float32),
                    )
                    if batch.pixel_targets is None
                    else batch.pixel_targets.to(selected_device, dtype=torch.float32)
                )
            else:
                pixel_prediction = prediction
                pixel_target = (
                    represented_target
                    if batch.pixel_targets is None
                    else batch.pixel_targets.to(selected_device, dtype=torch.float32)
                )
            collected["predictions"].append(pixel_prediction.cpu())
            collected["targets"].append(pixel_target.cpu())
            collected["represented_predictions"].append(prediction.cpu())
            collected["represented_targets"].append(represented_target.cpu())
            collected["original_valid_mask"].append(batch.original_valid_mask.cpu())
            collected["synthetic_mask"].append(batch.synthetic_mask.cpu())
            collected["observed_mask"].append(batch.observed_mask.cpu())
            if batch.bbox_diagonal is None:
                all_have_scale = False
            else:
                collected["bbox_diagonal"].append(batch.bbox_diagonal.cpu())
    if not collected["predictions"]:
        raise ValueError("Inference loader is empty")
    result: dict[str, Tensor | None] = {
        key: torch.cat(value) for key, value in collected.items() if key != "bbox_diagonal"
    }
    result["bbox_diagonal"] = (
        torch.cat(collected["bbox_diagonal"])
        if all_have_scale and collected["bbox_diagonal"]
        else None
    )
    return result


def evaluate_reconstruction_model(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device | None = None,
    *,
    joint_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    result = predict_reconstruction(model, loader, device)
    return reconstruction_metrics(
        result["predictions"],
        result["targets"],
        result["original_valid_mask"],
        result["synthetic_mask"],
        bbox_diagonal=result["bbox_diagonal"],
        joint_names=joint_names,
    )


def load_reconstruction_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if checkpoint.get("task") != "keypoint_reconstruction":
        raise ValueError(f"{path} is not a keypoint-reconstruction checkpoint")
    if model is None:
        model = build_reconstruction_model(
            checkpoint["model_name"], **checkpoint["model_kwargs"]
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model.to(map_location)
    return model, checkpoint


__all__ = [
    "ReconstructionBatch",
    "ReconstructionTrainer",
    "ReconstructionTrainerConfig",
    "ReconstructionTrainingResult",
    "evaluate_reconstruction_model",
    "load_reconstruction_checkpoint",
    "predict_reconstruction",
    "reconstruction_storage_report",
    "unpack_reconstruction_batch",
]
