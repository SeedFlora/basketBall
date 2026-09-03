from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from scripts.cleanup import discover_cleanup_candidates, validate_target
from scripts.evaluate import (
    ReconstructionEvaluationError,
    audit_group_leakage,
)
from scripts.evaluate import (
    evaluate as run_evaluation,
)
from scripts.evaluate import (
    parse_args as parse_evaluate_args,
)
from scripts.generate_report import generate_report
from scripts.run_ablation import (
    aggregate_equal_domains,
    aggregate_present,
    controlled_model_comparison,
    dataset_tables,
    discover_reconstruction_records,
    run_ablation,
)
from src.evaluation.error_analysis import (
    build_error_analysis_table,
    build_reconstruction_error_table,
)
from src.evaluation.metrics import (
    aggregate_metric_runs,
    classification_metrics,
    reconstruction_metrics,
)
from src.evaluation.robustness import (
    apply_missing_joints,
    generate_missing_joint_mask,
    temporal_interpolate_missing,
)
from src.models import temporal_interpolation_baseline
from src.training import prepare_reconstruction_representation
from src.visualization.confusion_matrix import plot_confusion_matrix
from src.visualization.embedding_plot import (
    pca_projection,
    plot_embedding,
    tsne_projection,
)
from src.visualization.reconstruction import (
    plot_cross_domain_matrix,
    plot_per_joint_error,
)
from src.visualization.skeleton_plot import plot_skeleton_sequence


def test_complete_classification_metrics() -> None:
    truth = np.array([0, 0, 1, 1, 2, 2])
    prediction = np.array([0, 1, 1, 1, 2, 0])
    metrics = classification_metrics(truth, prediction)

    assert metrics["primary_metric"] == "macro_f1"
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["macro_precision"] == pytest.approx((0.5 + 2 / 3 + 1.0) / 3)
    assert metrics["macro_recall"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert metrics["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert metrics["weighted_f1"] == pytest.approx(metrics["macro_f1"])
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert set(metrics["per_class"]) == {"dribble", "lay-up", "shoot"}
    assert metrics["per_class"]["lay-up"]["support"] == 2


def test_metrics_accept_logits_and_aggregate_only_present_values() -> None:
    logits = np.array([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    metrics = classification_metrics([0, 1], logits)
    assert metrics["accuracy"] == 1.0
    aggregate = aggregate_metric_runs(
        [{"macro_f1": 0.5}, {"macro_f1": 0.7, "accuracy": 0.8}]
    )
    assert aggregate["macro_f1"] == {"mean": pytest.approx(0.6), "std": pytest.approx(0.1), "n": 2}
    assert aggregate["accuracy"]["n"] == 1
    assert "macro_precision" not in aggregate


def test_bbox_normalized_reconstruction_metrics_and_per_joint_support() -> None:
    target = np.zeros((1, 1, 10, 2), dtype=np.float32)
    prediction = target.copy()
    prediction[0, 0, 0, 0] = 0.5
    prediction[0, 0, 1, 0] = 1.0
    prediction[0, 0, 2, 0] = 2.0
    evaluation_mask = np.zeros((1, 1, 10), dtype=bool)
    evaluation_mask[0, 0, :3] = True
    original_valid = np.ones_like(evaluation_mask)
    bboxes = np.array([[[0.0, 0.0, 6.0, 8.0]]], dtype=np.float32)
    names = [f"joint_{index}" for index in range(10)]

    metrics = reconstruction_metrics(
        prediction,
        target,
        bboxes=bboxes,
        evaluation_mask=evaluation_mask,
        original_valid_mask=original_valid,
        joint_names=names,
    )

    assert metrics["primary_metric"] == "nmje"
    assert metrics["primary_metric_direction"] == "lower_is_better"
    assert metrics["nmje"] == pytest.approx((0.05 + 0.10 + 0.20) / 3)
    assert metrics["pck_at_0_05"] == pytest.approx(1 / 3)
    assert metrics["pck_at_0_10"] == pytest.approx(2 / 3)
    assert metrics["pck_at_0_20"] == pytest.approx(1.0)
    assert metrics["n_scored_joints"] == 3
    assert metrics["per_joint"]["joint_0"]["nmje"] == pytest.approx(0.05)
    assert metrics["per_joint"]["joint_9"]["count"] == 0
    assert metrics["per_joint"]["joint_9"]["nmje"] is None


def test_reconstruction_metrics_never_score_naturally_missing_joint() -> None:
    target = np.zeros((1, 1, 10, 2), dtype=np.float32)
    prediction = target.copy()
    prediction[0, 0, 0] = [1000.0, 1000.0]
    prediction[0, 0, 1] = [1.0, 0.0]
    held_out = np.zeros((1, 1, 10), dtype=bool)
    held_out[0, 0, :2] = True
    original = np.ones_like(held_out)
    original[0, 0, 0] = False
    metrics = reconstruction_metrics(
        prediction,
        target,
        bboxes=np.array([[[0.0, 0.0, 6.0, 8.0]]]),
        evaluation_mask=held_out,
        original_valid_mask=original,
    )
    assert metrics["n_scored_joints"] == 1
    assert metrics["nmje"] == pytest.approx(0.1)


def test_drone_pose_tight_bbox_is_metric_only_after_masking() -> None:
    coords = np.zeros((1, 1, 10, 2), dtype=np.float32)
    coords[0, 0, :, 0] = np.arange(10, dtype=np.float32)
    coords[0, 0, :, 1] = np.arange(10, dtype=np.float32) * 2
    valid = np.ones((1, 1, 10), dtype=bool)
    hidden = np.zeros_like(valid)
    hidden[0, 0, 9] = True
    source = np.array([["pose_tight_fallback"]])
    first = prepare_reconstruction_representation(
        torch.from_numpy(coords),
        torch.from_numpy(valid),
        torch.from_numpy(hidden),
        representation="bbox_normalized",
        bbox_xywh=torch.tensor([[[0.0, 0.0, 9.0, 18.0]]]),
        bbox_valid=torch.ones((1, 1), dtype=torch.bool),
        bbox_source=source,
    )
    changed = coords.copy()
    changed[0, 0, 9] = [900.0, 1800.0]
    second = prepare_reconstruction_representation(
        torch.from_numpy(changed),
        torch.from_numpy(valid),
        torch.from_numpy(hidden),
        representation="bbox_normalized",
        bbox_xywh=torch.tensor([[[0.0, 0.0, 900.0, 1800.0]]]),
        bbox_valid=torch.ones((1, 1), dtype=torch.bool),
        bbox_source=source,
    )
    assert torch.equal(first["inputs"], second["inputs"])
    assert torch.equal(first["inverse_offset"], second["inverse_offset"])
    assert torch.equal(first["inverse_scale"], second["inverse_scale"])
    assert torch.equal(first["transform_bbox_xywh"], second["transform_bbox_xywh"])
    assert bool(first["pose_tight_metric_only"].all())
    assert not torch.equal(first["bbox_diagonal"], second["bbox_diagonal"])


def test_missing_joint_mask_is_exact_and_deterministic() -> None:
    sequences = np.ones((3, 10, 5, 2), dtype=np.float32)
    first = generate_missing_joint_mask(sequences, 0.2, seed=123)
    second = generate_missing_joint_mask(sequences, 0.2, seed=123)
    different = generate_missing_joint_mask(sequences, 0.2, seed=124)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert first.shape == (3, 10, 5)
    assert np.all(first.sum(axis=(1, 2)) == 10)

    perturbed, returned_mask = apply_missing_joints(
        sequences, 0.4, strategy="joint_masking", seed=9, return_mask=True
    )
    assert np.all(perturbed[returned_mask] == 0)
    # A selected joint is masked for every frame.
    assert np.all(np.logical_or(returned_mask.all(axis=1), ~returned_mask.any(axis=1)))


def test_temporal_interpolation_handles_interior_edges_and_all_missing() -> None:
    sequence = np.array(
        [
            [[0.0], [5.0]],
            [[99.0], [6.0]],
            [[4.0], [7.0]],
            [[6.0], [8.0]],
        ]
    )
    mask = np.array(
        [
            [False, True],
            [True, True],
            [False, True],
            [False, True],
        ]
    )
    interpolated = temporal_interpolate_missing(sequence, mask)
    assert interpolated[1, 0, 0] == pytest.approx(2.0)
    assert np.all(interpolated[:, 1, 0] == 0.0)
    edge_mask = np.zeros((4, 2), dtype=bool)
    edge_mask[0, 0] = True
    edge_result = temporal_interpolate_missing(sequence, edge_mask)
    assert edge_result[0, 0, 0] == pytest.approx(99.0)


def test_reconstruction_temporal_baseline_interpolates_only_from_observed_frames() -> None:
    coordinates = np.zeros((1, 3, 2, 2), dtype=np.float32)
    coordinates[0, 0, 0] = [0.0, 2.0]
    coordinates[0, 2, 0] = [4.0, 6.0]
    observed = np.zeros((1, 3, 2), dtype=bool)
    observed[0, [0, 2], 0] = True
    prediction = temporal_interpolation_baseline(
        torch.as_tensor(coordinates), torch.as_tensor(observed)
    ).numpy()
    assert prediction[0, 1, 0].tolist() == pytest.approx([2.0, 4.0])
    assert np.allclose(prediction[0, :, 1], 0.0)


def test_error_table_adds_only_evidence_backed_optional_columns() -> None:
    basic = build_error_analysis_table([0, 1], [1, 1], sample_ids=["a", "b"])
    assert list(basic["sample_id"]) == ["a"]
    assert "duration_frames" not in basic
    assert "mean_motion_magnitude" not in basic
    assert "missing_joint_fraction" not in basic
    assert "mean_confidence" not in basic

    sequence = np.arange(2 * 3 * 2 * 2, dtype=float).reshape(2, 3, 2, 2)
    valid = np.ones((2, 3, 2), dtype=bool)
    valid[0, 1, 0] = False
    confidence = np.full((2, 3, 2), 0.8)
    enriched = build_error_analysis_table(
        [0, 1],
        [1, 1],
        sequences=sequence,
        sequence_lengths=[7, 8],
        valid_mask=valid,
        confidence=confidence,
        include_correct=True,
    )
    assert {
        "duration_frames",
        "mean_motion_magnitude",
        "missing_joint_fraction",
        "mean_confidence",
    }.issubset(enriched.columns)
    assert enriched.loc[0, "missing_joint_fraction"] == pytest.approx(1 / 6)


def test_reconstruction_error_table_uses_observable_scale_pose_and_distal_fields() -> None:
    target = np.zeros((2, 1, 10, 2), dtype=np.float32)
    target[:, 0, :, 0] = np.arange(10)
    prediction = target.copy()
    prediction[:, 0, 5, 0] += 1.0
    hidden = np.zeros((2, 1, 10), dtype=bool)
    hidden[:, 0, 5] = True
    table = build_reconstruction_error_table(
        prediction,
        target,
        bboxes=np.array([[[0.0, 0.0, 6.0, 8.0]], [[0.0, 0.0, 12.0, 16.0]]]),
        evaluation_mask=hidden,
        original_valid_mask=np.ones_like(hidden),
        sample_ids=["small", "large"],
        domains=["Indoor", "Outdoor"],
    )
    assert {
        "mean_bbox_diagonal",
        "bbox_diagonal_percentile",
        "player_scale_bin",
        "pose_spread_bbox_fraction",
        "masked_distal_joints",
        "distal_nmje",
        "non_distal_nmje",
    }.issubset(table.columns)
    assert table.loc[0, "mean_bbox_diagonal"] == pytest.approx(10.0)
    assert table.loc[0, "masked_distal_joints"] == 1
    assert pd.isna(table.loc[0, "non_distal_nmje"])


def test_tiny_embedding_and_publication_plots_do_not_crash(tmp_path: Path) -> None:
    assert pca_projection(np.empty((0, 3))).shape == (0, 2)
    assert np.array_equal(pca_projection([[4.0, 5.0]]), np.zeros((1, 2)))
    tiny_tsne = tsne_projection([[0.0, 1.0], [1.0, 0.0]])
    assert tiny_tsne.shape == (2, 2)

    embedding_path = tmp_path / "embedding.png"
    figure, _ = plot_embedding(tiny_tsne, [0, 1], output_path=embedding_path)
    plt.close(figure)
    assert embedding_path.stat().st_size > 0

    confusion_path = tmp_path / "confusion.png"
    figure, _ = plot_confusion_matrix(
        [[2, 0, 0], [0, 1, 1], [0, 0, 2]],
        ["dribble", "lay-up", "shoot"],
        output_path=confusion_path,
    )
    plt.close(figure)
    assert confusion_path.stat().st_size > 0

    skeleton_path = tmp_path / "sequence.png"
    skeleton = np.zeros((4, 3, 2), dtype=float)
    skeleton[:, :, 0] = np.arange(3)
    figure, _ = plot_skeleton_sequence(
        skeleton, [(0, 1), (1, 2)], output_path=skeleton_path
    )
    assert all(axis.yaxis_inverted() for axis in figure.axes)
    plt.close(figure)
    assert skeleton_path.stat().st_size > 0


def test_group_audit_prefers_stored_video_track_groups_and_detects_leakage() -> None:
    data = {
        "domains": np.array(["Indoor", "Indoor", "Drone", "Drone"]),
        "splits": np.array(["train", "val", "train", "test"]),
        "video_ids": np.array(["v1", "v2", "", ""]),
        "track_ids": np.array(["p1", "p1", "d1", "d2"]),
        "split_group": np.array(
            ["Indoor/video/v1", "Indoor/video/v2", "Drone/player/d1", "Drone/player/d2"]
        ),
        "grouping_level": np.array(["video", "video", "player", "player"]),
    }
    audit = audit_group_leakage(data)
    assert audit["passed"] is True
    assert "video" in audit["grouping_unit"]
    assert audit["track_fallback_used"] is False

    leaked = dict(data)
    leaked["split_group"] = np.array(
        ["Indoor/video/shared", "Indoor/video/shared", "Drone/player/d1", "Drone/player/d2"]
    )
    failed = audit_group_leakage(leaked)
    assert failed["passed"] is False
    assert failed["overlap_counts"]["train/val"] == 1


def test_temporal_evaluation_rejects_single_frame_artifact(tmp_path: Path) -> None:
    coordinates = np.zeros((3, 1, 10, 2), dtype=np.float32)
    valid = np.ones((3, 1, 10), dtype=bool)
    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset,
        coords=coordinates,
        valid_mask=valid,
        bbox=np.tile(np.array([[[0.0, 0.0, 10.0, 10.0]]], dtype=np.float32), (3, 1, 1)),
        bbox_valid=np.ones((3, 1), dtype=bool),
        bbox_source=np.full((3, 1), "mot_xywh"),
        domain=np.array(["Indoor", "Indoor", "Indoor"]),
        sample_id=np.array(["train", "val", "test"]),
        split=np.array(["train", "val", "test"]),
        video_id=np.array(["v_train", "v_val", "v_test"]),
        track_id=np.array(["p_train", "p_val", "p_test"]),
        sequence_mask=np.ones((3, 1), dtype=bool),
        temporal_reliable=np.ones(3, dtype=bool),
        joint_names=np.array([f"joint_{index}" for index in range(10)]),
        edges=np.array([[index, index + 1] for index in range(9)], dtype=np.int64).T,
    )
    args = parse_evaluate_args(
        [
            "--processed",
            str(dataset),
            "--baseline",
            "temporal",
            "--mask-ratios",
            "0.20",
            "--output-dir",
            str(tmp_path / "logs"),
            "--figures-dir",
            str(tmp_path / "figures"),
        ]
    )
    with pytest.raises(ReconstructionEvaluationError, match="T>1"):
        run_evaluation(args)


def test_model_comparison_never_mixes_sequence_lengths() -> None:
    records = pd.DataFrame(
        {
            "model": ["mlp", "geometric_baseline", "tcn"],
            "representation": ["bbox_normalized"] * 3,
            "train_domain": ["all", "not_applicable", "all"],
            "test_domain": ["Indoor"] * 3,
            "mask_ratio": [0.2] * 3,
            "data_percent": [100.0] * 3,
            "sequence_length": [1, 1, 16],
            "synthetic": [False] * 3,
            "smoke": [False] * 3,
        }
    )
    comparison = controlled_model_comparison(records)
    assert set(comparison["model"]) == {"mlp", "geometric_baseline"}
    assert comparison["sequence_length"].nunique() == 1
    assert int(comparison["sequence_length"].iloc[0]) == 1


def test_equal_domain_aggregation_does_not_weight_duplicate_runs() -> None:
    records = pd.DataFrame(
        {
            "representation": ["bbox_normalized"] * 3,
            "test_domain": ["Indoor", "Indoor", "Outdoor"],
            "synthetic": [False] * 3,
            "smoke": [False] * 3,
            "nmje": [0.0, 0.0, 1.0],
            "pck_at_0_10": [1.0, 1.0, 0.0],
        }
    )
    aggregate = aggregate_equal_domains(
        records, ("representation", "synthetic", "smoke")
    )
    assert aggregate.loc[0, "nmje_mean"] == pytest.approx(0.5)
    assert aggregate.loc[0, "pck_at_0_10_mean"] == pytest.approx(0.5)
    assert aggregate.loc[0, "domains_averaged"] == 2


def test_dataset_table_counts_unique_scoped_players_and_exact_fields() -> None:
    coords = np.zeros((3, 1, 10, 2), dtype=np.float32)
    valid = np.ones((3, 1, 10), dtype=bool)
    subset, _ = dataset_tables(
        {
            "coords": coords,
            "valid_mask": valid,
            "domain": np.array(["Indoor", "Indoor", "Indoor"]),
            "sequence_mask": np.ones((3, 1), dtype=bool),
            "video_id": np.array(["v1", "v1", "v2"]),
            "track_id": np.array(["p1", "p1", "p1"]),
            "frame_index": np.array([[1], [2], [1]]),
            "joint_names": np.array([f"joint_{index}" for index in range(10)]),
            "synthetic": np.zeros(3, dtype=bool),
        }
    )
    assert list(subset.columns) == [
        "Domain",
        "Videos",
        "Frames",
        "Player instances",
        "Player tracks",
        "Pose instances",
        "Storage",
    ]
    assert subset.loc[0, "Videos"] == 2
    assert subset.loc[0, "Frames"] == 3
    assert subset.loc[0, "Player instances"] == 3
    assert subset.loc[0, "Player tracks"] == 2
    assert subset.loc[0, "Pose instances"] == 3


def test_reconstruction_plots_retain_missing_cross_domain_cells(tmp_path: Path) -> None:
    cross = pd.DataFrame(
        {
            "train_domain": ["Indoor", "Outdoor", "Outdoor"],
            "test_domain": ["Indoor", "Indoor", "Drone"],
            "nmje": [0.1, 0.2, 0.3],
        }
    )
    path = tmp_path / "cross.png"
    figure, axis = plot_cross_domain_matrix(cross, metric="nmje", output_path=path)
    labels = [text.get_text() for text in axis.texts]
    assert "—" in labels
    assert "0.000" not in labels
    plt.close(figure)
    assert path.stat().st_size > 0

    joint_path = tmp_path / "joints.png"
    figure, _ = plot_per_joint_error(
        pd.DataFrame({"joint": ["head", "center"], "nmje": [0.1, 0.2]}),
        output_path=joint_path,
    )
    plt.close(figure)
    assert joint_path.stat().st_size > 0


def test_ablation_uses_only_audited_non_smoke_reconstruction_metrics(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    real = logs / "real_run"
    synthetic = logs / "synthetic_run"
    smoke = logs / "smoke_real_run"
    legacy = logs / "legacy_run"
    real.mkdir(parents=True)
    synthetic.mkdir(parents=True)
    smoke.mkdir(parents=True)
    legacy.mkdir(parents=True)
    (real / "metrics.json").write_text(
        json.dumps(
            {
                "artifact_type": "reconstruction_evaluation",
                "task": "keypoint_reconstruction",
                "synthetic": False,
                "model": "mlp",
                "representation": "bbox_normalized",
                "train_domain": "Indoor",
                "test_domain": "Indoor",
                "mask_ratio": 0.2,
                "leakage_audit_passed": True,
                "metrics": {"nmje": 0.061, "pck_at_0_10": 0.70},
            }
        ),
        encoding="utf-8",
    )
    (synthetic / "metrics.json").write_text(
        json.dumps(
            {
                "task": "keypoint_reconstruction",
                "synthetic": True,
                "model": "graph",
                "representation": "normalized_angles",
                "train_domain": "Indoor",
                "test_domain": "Indoor",
                "mask_ratio": 0.2,
                "leakage_audit_passed": True,
                "metrics": {"nmje": 0.001, "pck_at_0_10": 1.0},
            }
        ),
        encoding="utf-8",
    )
    (smoke / "metrics.json").write_text(
        json.dumps(
            {
                "task": "keypoint_reconstruction",
                "synthetic": False,
                "model": "mlp",
                "representation": "raw",
                "train_domain": "Indoor",
                "test_domain": "Indoor",
                "mask_ratio": 0.2,
                "leakage_audit_passed": True,
                "metrics": {"nmje": 0.0001},
            }
        ),
        encoding="utf-8",
    )
    (legacy / "metrics.json").write_text(
        json.dumps({"test": {"macro_f1": 0.99, "accuracy": 1.0}}),
        encoding="utf-8",
    )
    records = discover_reconstruction_records(logs)
    assert records["representation"].tolist() == ["bbox_normalized"]
    table = aggregate_present(records, ("model", "representation"))
    assert len(table) == 1
    assert table.loc[0, "nmje_mean"] == pytest.approx(0.061)
    assert "macro_f1_mean" not in table


def test_pre_observed_transform_drone_metrics_are_rejected(tmp_path: Path) -> None:
    run = tmp_path / "logs" / "drone"
    run.mkdir(parents=True)
    payload = {
        "task": "keypoint_reconstruction",
        "model": "mlp",
        "representation": "bbox_normalized",
        "train_domain": "Indoor",
        "test_domain": "Drone",
        "mask_ratio": 0.2,
        "data_percent": 100,
        "sequence_length": 1,
        "leakage_audit_passed": True,
        "metrics": {"nmje": 0.1, "pck_at_0_10": 0.5},
    }
    path = run / "metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert discover_reconstruction_records(tmp_path / "logs").empty
    payload["input_transform_audit_passed"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    records = discover_reconstruction_records(tmp_path / "logs")
    assert len(records) == 1
    assert records.loc[0, "data_percent"] == pytest.approx(100.0)


def test_ablation_gates_uncontrolled_figures_and_never_invents_rows(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    run = logs / "executed"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "task": "keypoint_reconstruction",
                "model": "mlp",
                "representation": "bbox_normalized",
                "train_domain": "Indoor",
                "test_domain": "Indoor",
                "mask_ratio": 0.2,
                "data_percent": 100,
                "leakage_audit_passed": True,
                "metrics": {
                    "nmje": 0.1,
                    "pck_at_0_05": 0.2,
                    "pck_at_0_10": 0.5,
                    "pck_at_0_20": 0.8,
                },
            }
        ),
        encoding="utf-8",
    )
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"
    run_ablation(logs, tables, figures)
    assert pd.read_csv(tables / "model_parameters_compute.csv").empty
    assert pd.read_csv(tables / "representation.csv").empty
    assert pd.read_csv(tables / "missing_keypoints.csv").empty
    assert not (figures / "nmje_by_model.png").exists()
    assert not (figures / "normalization_ablation.png").exists()
    index = json.loads((tables / "ablation_index.json").read_text(encoding="utf-8"))
    assert any("require >=2 models" in reason for reason in index["not_generated"])


def test_report_marks_missing_results_instead_of_fabricating(tmp_path: Path) -> None:
    (tmp_path / "outputs" / "reports").mkdir(parents=True)
    (tmp_path / "outputs" / "logs").mkdir(parents=True)
    path = generate_report(tmp_path)
    report = path.read_text(encoding="utf-8")
    assert "NO LEAKAGE-AUDITED NON-SYNTHETIC RECONSTRUCTION RESULTS" in report
    assert "No best research result can be reported" in report
    assert "TRACKID ANNOTATIONS NOT AVAILABLE LOCALLY" in report
    assert "PCK@0.10" in report
    assert "Macro F1" not in report
    assert "0.0000" not in report


def test_cleanup_is_allowlisted_and_path_safe(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    archive = raw / "MultiSubjects_SA.zip"
    archive.write_bytes(b"archive")
    temporary = tmp_path / "outputs" / "tmp"
    temporary.mkdir(parents=True)
    (temporary / "scratch.bin").write_bytes(b"temporary")
    candidates = discover_cleanup_candidates(tmp_path)
    by_category = {candidate.category: candidate for candidate in candidates}
    assert by_category["archive"].protected is True
    assert by_category["temporary"].protected is False
    with pytest.raises(ValueError):
        validate_target(tmp_path, tmp_path)
    with pytest.raises(ValueError):
        validate_target(tmp_path, tmp_path.parent / "outside")
