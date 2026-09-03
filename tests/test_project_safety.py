"""Cross-cutting safety and reproducibility checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from src.utils import deep_merge, load_config, storage_report

ROOT = Path(__file__).resolve().parents[1]


def _load_download_module():
    spec = importlib.util.spec_from_file_location(
        "download_dataset", ROOT / "scripts" / "download_dataset.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trackid_downloader_is_annotation_only_and_bounded() -> None:
    module = _load_download_module()
    assert module.REPOSITORY == "open-starlab/TrackID3x3"
    assert module.CLONE_URL == "https://github.com/open-starlab/TrackID3x3.git"
    assert set(module.SPARSE_PATHS) == {"ground_truth", "court_images", "script"}
    assert "videos" not in module.SPARSE_PATHS
    assert module.TARGET_DATASET_MB == 500.0
    assert module.MAX_DATASET_MB == 600.0


def test_no_forbidden_archive_present() -> None:
    forbidden = {"Subjects1000.rar", "MultiSubjects_train_val_test.rar"}
    present = {path.name for path in ROOT.rglob("*") if path.is_file()}
    assert forbidden.isdisjoint(present)


def test_config_inheritance_and_deep_merge() -> None:
    merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}})
    assert merged == {"a": {"b": 3, "c": 2}}
    config = load_config(ROOT / "config" / "smoke_synthetic.yaml")
    assert config["data"]["sequence_length"] == 16
    assert config["training"]["learning_rate"] == 0.001
    assert config["training"]["epochs"] == 2


def test_storage_report_has_required_categories() -> None:
    report = storage_report(ROOT)
    assert set(report) == {
        "raw_dataset_bytes",
        "processed_dataset_bytes",
        "model_checkpoint_bytes",
        "total_project_artifact_bytes",
    }
    assert all(isinstance(value, int) and value >= 0 for value in report.values())
