#!/usr/bin/env python
"""Train compact TrackID3x3 masked-keypoint reconstruction models.

Targets are hidden in source-pixel space before any representation is built.
All validation predictions are inverted to pixels for bbox-diagonal NMJE/PCK.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    TRACKID3X3_DOMAINS,
    TrackIDProcessedError,
    load_trackid_processed,
)
from src.models import (
    RECONSTRUCTION_MODEL_NAMES,
    build_reconstruction_model,
    count_parameters,
)
from src.training import (
    RECONSTRUCTION_REPRESENTATIONS,
    SUPPORTED_MASK_RATIOS,
    ReconstructionTrainer,
    ReconstructionTrainerConfig,
    canonical_representation_name,
    deterministic_reconstruction_mask,
    prepare_reconstruction_representation,
    reconstruction_storage_report,
    validate_mask_ratio,
)
from src.utils import (
    human_bytes,
    load_config,
    set_seed,
    write_json,
)


class ReconstructionDataError(ValueError):
    """Raised when selected data cannot support a scientific run."""


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _resolve_processed(config: Mapping[str, Any], argument: str | None) -> Path:
    configured = config.get("data", {}).get("processed_dir", "data/processed")
    path = _project_path(argument or configured)
    if path.suffix:
        return path
    direct = path / "dataset.npz"
    nested = path / "trackid3x3" / "dataset.npz"
    return nested if not direct.is_file() and nested.is_file() else direct


def _metadata_for(dataset_path: Path) -> dict[str, Any]:
    path = dataset_path.with_name("metadata.json")
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReconstructionDataError(f"Could not parse {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ReconstructionDataError(f"Metadata root must be an object: {path}")
    return dict(value)


def canonical_domain_name(value: str) -> str:
    token = str(value).strip().casefold()
    if token == "all":
        return "all"
    for domain in TRACKID3X3_DOMAINS:
        if token == domain.casefold():
            return domain
    raise argparse.ArgumentTypeError("domain must be indoor, outdoor, drone, or all")


def _indices_for(data: Mapping[str, np.ndarray], split: str, domain: str) -> np.ndarray:
    selector = np.asarray(data["split"]).astype(str) == split
    if domain != "all":
        selector &= np.asarray(data["domain"]).astype(str) == domain
    indices = np.flatnonzero(selector)
    if indices.size == 0:
        raise ReconstructionDataError(f"No {split!r} samples exist for domain={domain}")
    return indices


def validate_train_fraction(value: float) -> float:
    fraction = float(value)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("train_fraction must satisfy 0 < fraction <= 1")
    return fraction


def _fractional_train_indices(
    data: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select whole train split-groups deterministically within each domain."""

    requested = validate_train_fraction(fraction)
    domains = np.asarray(data["domain"]).astype(str)
    groups = np.asarray(data["split_group"]).astype(str)
    chosen_groups: set[tuple[str, str]] = set()
    by_domain: dict[str, dict[str, Any]] = {}
    for domain in TRACKID3X3_DOMAINS:
        domain_indices = indices[domains[indices] == domain]
        if domain_indices.size == 0:
            continue
        available = sorted(set(groups[domain_indices].tolist()))
        count = (
            len(available)
            if requested == 1.0
            else max(1, min(len(available), int(np.floor(len(available) * requested + 0.5))))
        )
        digest = hashlib.blake2b(
            f"{seed}\0train-fraction\0{domain}".encode(), digest_size=8
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        selected = sorted(available[index] for index in rng.permutation(len(available))[:count])
        chosen_groups.update((domain, group) for group in selected)
        selected_rows = domain_indices[np.isin(groups[domain_indices], selected)]
        group_digest = hashlib.sha256("\n".join(selected).encode()).hexdigest()
        by_domain[domain] = {
            "available_groups": len(available),
            "selected_groups": len(selected),
            "available_samples": int(domain_indices.size),
            "selected_samples": int(selected_rows.size),
            "selected_group_sha256": group_digest,
        }
    selected_selector = np.fromiter(
        ((domains[index], groups[index]) in chosen_groups for index in indices),
        dtype=bool,
        count=indices.size,
    )
    selected_indices = indices[selected_selector]
    if selected_indices.size == 0:
        raise ReconstructionDataError("Training fraction selected no complete split group")
    available_groups = len(
        set(zip(domains[indices].tolist(), groups[indices].tolist(), strict=True))
    )
    selected_group_count = len(chosen_groups)
    report = {
        "requested_fraction": requested,
        "selection_unit": "split_group",
        "selection_algorithm": "seeded-domain-stratified-whole-groups-v1",
        "seed": int(seed),
        "available_samples": int(indices.size),
        "selected_samples": int(selected_indices.size),
        "actual_sample_fraction": float(selected_indices.size / indices.size),
        "available_groups": available_groups,
        "selected_groups": selected_group_count,
        "actual_group_fraction": float(selected_group_count / available_groups),
        "by_domain": by_domain,
        "validation_unchanged": True,
    }
    return selected_indices, report


def _validate_bbox_support(data: Mapping[str, np.ndarray], indices: np.ndarray) -> None:
    sequence = np.asarray(data["sequence_mask"], dtype=bool)[indices]
    valid = np.asarray(data["valid_mask"], dtype=bool)[indices] & sequence[..., None]
    bbox_valid = np.asarray(data["bbox_valid"], dtype=bool)[indices]
    if np.any(valid.sum(axis=(1, 2)) == 0):
        raise ReconstructionDataError("Every selected sample must contain a valid source joint")
    if np.any(sequence & ~bbox_valid):
        raise ReconstructionDataError(
            "Every real frame requires a source/fallback bbox for primary NMJE/PCK"
        )


def _validate_tcn_windows(data: Mapping[str, np.ndarray], indices: np.ndarray) -> None:
    if "temporal_reliable" not in data:
        raise ReconstructionDataError("TCN requires temporal_reliable provenance")
    reliable = np.asarray(data["temporal_reliable"], dtype=bool)[indices]
    if reliable.shape != (indices.size,) or not bool(reliable.all()):
        raise ReconstructionDataError(
            "TCN is restricted to reliable same-video/same-player track windows"
        )
    sequence = np.asarray(data["sequence_mask"], dtype=bool)[indices]
    if not bool(sequence.all()):
        raise ReconstructionDataError("TCN cannot consume padded/incomplete track windows")
    for key in ("video_id", "track_id"):
        identifiers = np.asarray(data[key]).astype(str)[indices]
        if any(not value.strip() or "unavailable" in value.casefold() for value in identifiers):
            raise ReconstructionDataError(f"TCN requires a reliable non-empty {key}")
    frames = np.asarray(data["frame_index"], dtype=np.int64)[indices]
    if frames.shape[1] > 1:
        differences = np.diff(frames, axis=1)
        regular = (differences > 0).all(axis=1) & (
            differences == differences[:, :1]
        ).all(axis=1)
        if not bool(regular.all()):
            raise ReconstructionDataError(
                "TCN frame indices must increase at one constant source stride per window"
            )


class ReconstructionDataset(Dataset[dict[str, Any]]):
    """Apply a fixed per-sample mask, then build observed-only inputs."""

    def __init__(
        self,
        data: Mapping[str, np.ndarray],
        indices: np.ndarray,
        *,
        representation: str,
        mask_ratio: float,
        seed: int,
    ) -> None:
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.representation = canonical_representation_name(representation)
        self.mask_ratio = validate_mask_ratio(mask_ratio)
        self.seed = int(seed)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = int(self.indices[index])
        pixels = torch.from_numpy(
            np.asarray(self.data["coords"][source_index], dtype=np.float32).copy()
        )
        original = torch.from_numpy(
            np.asarray(self.data["valid_mask"][source_index], dtype=bool).copy()
        )
        sequence = torch.from_numpy(
            np.asarray(self.data["sequence_mask"][source_index], dtype=bool).copy()
        )
        original &= sequence.unsqueeze(-1)
        sample_id = str(self.data["sample_id"][source_index])
        synthetic = deterministic_reconstruction_mask(
            original,
            self.mask_ratio,
            seed=self.seed,
            instance_ids=sample_id,
        )
        prepared = prepare_reconstruction_representation(
            pixels,
            original,
            synthetic,
            representation=self.representation,
            bbox_xywh=torch.from_numpy(
                np.asarray(self.data["bbox"][source_index], dtype=np.float32).copy()
            ),
            bbox_valid=torch.from_numpy(
                np.asarray(self.data["bbox_valid"][source_index], dtype=bool).copy()
            ),
            bbox_source=np.asarray(self.data["bbox_source"][source_index]).astype(str),
            sequence_mask=sequence,
        )
        tensors = {
            key: value for key, value in prepared.items() if isinstance(value, Tensor)
        }
        tensors.update(
            {
                "sample_id": sample_id,
                "domain": str(self.data["domain"][source_index]),
            }
        )
        return tensors


def _worker_seed(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _architecture_kwargs(model_name: str, config: Mapping[str, Any]) -> dict[str, Any]:
    model_config = dict(config.get("model", {}))
    reconstruction = config.get("reconstruction", {})
    if isinstance(reconstruction, Mapping):
        overrides = reconstruction.get("model_kwargs")
        if isinstance(overrides, Mapping):
            model_config.update(overrides)
    allowed = {
        "mlp": {"hidden_dims", "dropout"},
        "tcn": {"channels", "kernel_size", "dropout"},
        "graph": {"channels", "dropout"},
    }[model_name]
    selected = {key: value for key, value in model_config.items() if key in allowed}
    width = model_config.get("hidden_dim")
    if width is not None and model_name == "mlp" and "hidden_dims" not in selected:
        width = int(width)
        selected["hidden_dims"] = (max(2 * width, 64), width)
    if width is not None and model_name in {"tcn", "graph"} and "channels" not in selected:
        width = int(width)
        selected["channels"] = (max(width // 2, 32), width, width)
    return selected


def _experiment_name(
    value: str | None,
    model: str,
    representation: str,
    domain: str,
    mask_ratio: float,
    train_fraction: float,
    seed: int,
) -> str:
    proposed = value or (
        f"reconstruction_{model}_{representation}_{domain.lower()}_"
        f"mask{round(100 * mask_ratio):02d}_data{round(100 * train_fraction):03d}_seed{seed}"
    )
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", proposed).strip("._")
    if not cleaned:
        raise ValueError("experiment-name must contain at least one letter or number")
    return cleaned


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=RECONSTRUCTION_MODEL_NAMES, default=None)
    parser.add_argument(
        "--representation",
        choices=(*RECONSTRUCTION_REPRESENTATIONS, "normalized"),
        default=None,
    )
    parser.add_argument(
        "--features",
        choices=(*RECONSTRUCTION_REPRESENTATIONS, "normalized"),
        default=None,
        help="Backward-compatible alias for --representation",
    )
    parser.add_argument("--mask-ratio", type=float, choices=SUPPORTED_MASK_RATIOS, default=None)
    parser.add_argument(
        "--domain",
        type=canonical_domain_name,
        default=None,
        help="indoor, outdoor, drone, or all (case-insensitive)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=None,
        help="Deterministic whole-group fraction of the selected training split (0 < f <= 1)",
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--processed", default=None, help="dataset.npz or its parent directory")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser.parse_args(argv)


def _configured(config: Mapping[str, Any], section: str, key: str) -> Any:
    values = config.get(section, {})
    return values.get(key) if isinstance(values, Mapping) else None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(_project_path(args.config))
    legacy_workflow = config.get("legacy_workflow")
    if isinstance(legacy_workflow, Mapping) and str(
        legacy_workflow.get("status", "")
    ).casefold() == "disabled":
        raise ReconstructionDataError(
            "legacy_henu_classification_disabled: this archived configuration cannot "
            "be used by the TrackID3x3 reconstruction training CLI; use "
            "config/trackid3x3.yaml instead"
        )
    seed = int(args.seed if args.seed is not None else _configured(config, "project", "seed") or 42)
    set_seed(seed)

    configured_model = _configured(config, "reconstruction", "model")
    if configured_model is None:
        configured_model = _configured(config, "model", "name")
    model_name = str(args.model or configured_model or "mlp").lower()
    if model_name not in RECONSTRUCTION_MODEL_NAMES:
        model_name = "mlp" if args.model is None else model_name
    if model_name not in RECONSTRUCTION_MODEL_NAMES:
        raise ValueError(f"model must be one of {RECONSTRUCTION_MODEL_NAMES}")

    if args.representation and args.features and (
        canonical_representation_name(args.representation)
        != canonical_representation_name(args.features)
    ):
        raise ValueError("--representation and --features disagree")
    configured_representation = _configured(config, "reconstruction", "representation")
    if configured_representation is None:
        configured_representation = _configured(config, "features", "representation")
    representation = canonical_representation_name(
        args.representation
        or args.features
        or configured_representation
        or "bbox_normalized"
    )
    configured_ratio = _configured(config, "reconstruction", "mask_ratio")
    if configured_ratio is None:
        configured_ratio = _configured(config, "reconstruction", "default_mask_rate")
    mask_ratio = validate_mask_ratio(
        args.mask_ratio if args.mask_ratio is not None else configured_ratio or 0.2
    )
    configured_domain = _configured(config, "reconstruction", "domain")
    if configured_domain is None:
        configured_domain = _configured(config, "data", "domain")
    domain = args.domain or canonical_domain_name(str(configured_domain or "all"))
    configured_fraction = _configured(config, "reconstruction", "train_fraction")
    if configured_fraction is None:
        configured_fraction = _configured(config, "data", "train_fraction")
    train_fraction = validate_train_fraction(
        args.train_fraction if args.train_fraction is not None else configured_fraction or 1.0
    )

    training_values = dict(config.get("training", {}))
    if args.epochs is not None:
        training_values["epochs"] = args.epochs
    if args.batch_size is not None:
        training_values["batch_size"] = args.batch_size
    if args.num_workers is not None:
        training_values["num_workers"] = args.num_workers
    training_values.setdefault("selection_metric", "nmje")
    trainer_config = ReconstructionTrainerConfig.from_mapping(training_values)

    processed_path = _resolve_processed(config, args.processed)
    data = load_trackid_processed(processed_path, aliases=True)
    metadata = _metadata_for(processed_path)
    full_train_indices = _indices_for(data, "train", domain)
    val_indices = _indices_for(data, "val", domain)
    train_indices, efficiency = _fractional_train_indices(
        data, full_train_indices, fraction=train_fraction, seed=seed
    )
    _validate_bbox_support(data, train_indices)
    _validate_bbox_support(data, val_indices)
    if model_name == "tcn":
        _validate_tcn_windows(data, train_indices)
        _validate_tcn_windows(data, val_indices)

    train_dataset = ReconstructionDataset(
        data,
        train_indices,
        representation=representation,
        mask_ratio=mask_ratio,
        seed=seed,
    )
    val_dataset = ReconstructionDataset(
        data,
        val_indices,
        representation=representation,
        mask_ratio=mask_ratio,
        seed=seed,
    )
    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": trainer_config.batch_size,
        "num_workers": trainer_config.num_workers,
        "pin_memory": bool(torch.cuda.is_available() and args.device != "cpu"),
        "worker_init_fn": _worker_seed,
        "persistent_workers": trainer_config.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model_kwargs: dict[str, Any] = {
        "in_features": 2,
        "num_joints": 10,
        "aux_features": 1 if representation == "normalized_angles" else 0,
        **_architecture_kwargs(model_name, config),
    }
    if model_name == "graph":
        model_kwargs["joint_names"] = np.asarray(data["joint_names"]).astype(str).tolist()
        model_kwargs["edges"] = np.asarray(data["edges"], dtype=np.int64).tolist()
    model = build_reconstruction_model(model_name, **model_kwargs)
    parameter_count = count_parameters(model)
    if parameter_count >= 5_000_000:
        raise ValueError(f"Configured {model_name} has {parameter_count:,} parameters; expected <5M")
    if model_name == "graph" and parameter_count >= 1_000_000:
        raise ValueError(f"Configured graph has {parameter_count:,} parameters; expected <1M")

    experiment_name = _experiment_name(
        args.experiment_name,
        model_name,
        representation,
        domain,
        mask_ratio,
        train_fraction,
        seed,
    )
    output_root = _project_path(_configured(config, "project", "output_dir") or "outputs")
    run_dir = output_root / "logs" / experiment_name
    checkpoint_dir = output_root / "checkpoints" / experiment_name
    extras = [path for path in checkpoint_dir.glob("*.pt") if path.name != "best.pt"]
    if extras:
        raise RuntimeError(f"{checkpoint_dir} contains extra checkpoints; use a unique run name")

    selected_domains = sorted(
        set(np.asarray(data["domain"])[np.r_[train_indices, val_indices]].astype(str).tolist())
    )
    synthetic = metadata.get("synthetic") is True
    selected_for_provenance = np.r_[train_indices, val_indices]
    bbox_source_values = np.asarray(data["bbox_source"])[selected_for_provenance].astype(str)
    bbox_source_counts = {
        source: int((bbox_source_values == source).sum())
        for source in sorted(set(bbox_source_values.reshape(-1).tolist()))
    }
    experiment = {
        "name": experiment_name,
        "task": "keypoint_reconstruction",
        "model": model_name,
        "representation": representation,
        "mask_ratio": mask_ratio,
        "domain": domain,
        "train_domain": domain,
        "selected_domains": selected_domains,
        "train_fraction": train_fraction,
        "data_fraction": train_fraction,
        "data_percent": 100.0 * train_fraction,
        "data_efficiency": efficiency,
        "sequence_length": int(np.asarray(data["coords"]).shape[1]),
        "mask_policy": "fixed deterministic per sample id; source-missing joints excluded",
        "transform_policy": "mask pixels first; observed-only center, scale, and angles",
        "bbox_transform_policy": (
            "MOT xywh is independent; pose_tight_fallback is recomputed from observed joints "
            "and retained from source only as the metric denominator"
        ),
        "bbox_source_frame_counts": bbox_source_counts,
        "metric_space": "source pixels divided by source bbox diagonal",
        "grouping_unit": "split_group",
        "leakage_audit_passed": True,
        "synthetic": synthetic,
    }
    resolved_config = dict(config)
    resolved_config["project"] = {
        **dict(config.get("project", {})),
        "seed": seed,
        "output_dir": str(output_root),
    }
    resolved_config["data"] = {
        **dict(config.get("data", {})),
        "processed_dataset": str(processed_path),
        "domain": domain,
    }
    resolved_config["reconstruction"] = {
        "model": model_name,
        "representation": representation,
        "mask_ratio": mask_ratio,
        "train_fraction": train_fraction,
        "model_kwargs": model_kwargs,
    }
    resolved_config["training"] = asdict(trainer_config)
    resolved_config["run"] = experiment

    trainer = ReconstructionTrainer(
        model,
        trainer_config,
        device=args.device,
        joint_names=np.asarray(data["joint_names"]).astype(str).tolist(),
    )
    result = trainer.fit(
        train_loader,
        val_loader,
        run_dir=run_dir,
        checkpoint_path=checkpoint_dir / "best.pt",
        seed=seed,
        model_name=model_name,
        model_kwargs=model_kwargs,
        resolved_config=resolved_config,
        experiment_metadata=experiment,
        synthetic=synthetic,
        project_root=PROJECT_ROOT,
    )
    if sorted(path.name for path in checkpoint_dir.glob("*.pt")) != ["best.pt"]:
        raise RuntimeError(f"Expected exactly one best checkpoint in {checkpoint_dir}")

    result.metrics["dataset"] = {
        "path": str(processed_path),
        "train_samples": int(train_indices.size),
        "available_train_samples": int(full_train_indices.size),
        "validation_samples": int(val_indices.size),
        "sequence_length": int(np.asarray(data["coords"]).shape[1]),
        "domains": selected_domains,
        "synthetic": synthetic,
    }
    result.metrics["data_fraction"] = train_fraction
    result.metrics["data_percent"] = 100.0 * train_fraction
    result.metrics["data_efficiency"] = efficiency
    result.metrics["storage"] = reconstruction_storage_report(PROJECT_ROOT)
    write_json(result.metrics, run_dir / "metrics.json")

    validation = result.metrics["best_validation"]
    print(f"Experiment: {experiment_name}")
    print(f"Synthetic fixture: {synthetic}")
    print(f"Parameter count: {result.metrics['parameter_count']:,}")
    print(f"Training time: {result.metrics['training_time_seconds']:.3f} seconds")
    print(f"Peak GPU memory: {result.metrics['peak_gpu_memory_bytes']}")
    print(f"Best checkpoint size: {result.metrics['checkpoint_size_bytes']} bytes")
    print(f"Best validation NMJE: {validation['nmje']:.6f}")
    print(
        "Validation PCK: "
        f"0.05={validation['pck@0.05']:.6f}, "
        f"0.10={validation['pck@0.10']:.6f}, "
        f"0.20={validation['pck@0.20']:.6f}"
    )
    storage = result.metrics["storage"]
    print(
        f"Raw TrackID data size: {human_bytes(storage['raw_dataset_bytes'])} "
        f"({storage['raw_dataset_bytes']} bytes)"
    )
    print(
        f"Processed dataset size: {human_bytes(storage['processed_dataset_bytes'])} "
        f"({storage['processed_dataset_bytes']} bytes)"
    )
    print(
        f"Model checkpoint size: {human_bytes(storage['model_checkpoint_bytes'])} "
        f"({storage['model_checkpoint_bytes']} bytes)"
    )
    print(
        f"Total TrackID artifact size: {human_bytes(storage['total_project_artifact_bytes'])} "
        f"({storage['total_project_artifact_bytes']} bytes)"
    )
    print(f"Run artifacts: {run_dir}")
    print(f"Best checkpoint: {result.checkpoint_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        argparse.ArgumentTypeError,
        FileNotFoundError,
        OSError,
        ReconstructionDataError,
        RuntimeError,
        TrackIDProcessedError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
