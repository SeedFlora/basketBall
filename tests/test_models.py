"""Forward-contract and compactness tests for every model family."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.train import _fractional_train_indices
from src.data import TRACKID10_EDGES
from src.models import (
    RECONSTRUCTION_MODEL_NAMES,
    TRACKID3X3_EDGES,
    TRACKID3X3_JOINT_NAMES,
    AggregatedMLP,
    BiLSTMClassifier,
    STGCNClassifier,
    TCNClassifier,
    build_model,
    build_reconstruction_model,
    count_parameters,
    normalized_adjacency,
    temporal_interpolation_baseline,
)
from src.training import (
    ReconstructionMasker,
    ReconstructionTrainer,
    ReconstructionTrainerConfig,
    deterministic_reconstruction_mask,
    inverse_reconstruction_coordinates,
    load_reconstruction_checkpoint,
    masked_reconstruction_loss,
    prepare_reconstruction_representation,
    reconstruction_metrics,
)


@pytest.mark.parametrize(
    ("model", "inputs", "mask", "embedding_dim"),
    [
        (AggregatedMLP(34, hidden_dims=(64, 32)), torch.randn(3, 12, 34), torch.ones(3, 12), 32),
        (BiLSTMClassifier(34, hidden_dim=24), torch.randn(3, 12, 34), torch.ones(3, 12), 48),
        (TCNClassifier(34, channels=(24, 32)), torch.randn(3, 12, 34), torch.ones(3, 12), 32),
        (
            STGCNClassifier(2, num_joints=17, channels=(16, 24, 32)),
            torch.randn(3, 2, 12, 17),
            torch.ones(3, 12, 17),
            32,
        ),
    ],
)
def test_forward_and_embedding_contract(model, inputs, mask, embedding_dim):
    model.eval()
    logits, embedding = model(inputs, valid_mask=mask, return_embedding=True)
    assert logits.shape == (3, 3)
    assert embedding.shape == (3, embedding_dim)
    assert torch.allclose(model.extract_features(inputs, mask), embedding)
    logits.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("mlp", {"in_features": 99}),
        ("lstm", {"in_features": 99}),
        ("tcn", {"in_features": 99}),
        ("stgcn", {"in_features": 3, "num_joints": 33}),
    ],
)
def test_default_models_are_below_five_million_parameters(name, kwargs):
    model = build_model(name, **kwargs)
    assert 0 < count_parameters(model) < 5_000_000


def test_stgcn_adjacency_is_symmetric_normalized_and_self_connected():
    adjacency = normalized_adjacency(4, [(0, 1), (1, 2), (1, 3)])
    assert adjacency.shape == (4, 4)
    assert torch.allclose(adjacency, adjacency.T)
    assert torch.all(adjacency.diag() > 0)
    assert torch.isfinite(adjacency).all()


def test_masks_make_padded_mlp_frames_invariant():
    torch.manual_seed(7)
    model = AggregatedMLP(5, hidden_dims=(12, 8), dropout=0).eval()
    sequence = torch.randn(2, 6, 5)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    changed = sequence.clone()
    changed[0, 3:] = 10_000
    assert torch.allclose(model(sequence, mask)[0], model(changed, mask)[0], atol=1e-6)


def test_clear_shape_errors():
    with pytest.raises(ValueError, match="expects"):
        TCNClassifier(8)(torch.randn(2, 8))
    with pytest.raises(ValueError, match="num_joints"):
        build_model("stgcn", in_features=2)


@pytest.mark.parametrize("name", RECONSTRUCTION_MODEL_NAMES)
def test_reconstructors_predict_coordinates_and_preserve_observations(name):
    torch.manual_seed(11)
    coordinates = torch.randn(2, 7, 10, 2)
    observed = torch.ones(2, 7, 10, dtype=torch.bool)
    observed[:, 2, [3, 6]] = False
    inputs = coordinates.masked_fill(~observed.unsqueeze(-1), 0)
    model = build_reconstruction_model(name, in_features=2, num_joints=10).eval()
    prediction, embedding = model(inputs, observed, return_embedding=True)
    assert prediction.shape == coordinates.shape
    assert embedding.shape[:2] == coordinates.shape[:2]
    assert torch.equal(prediction[observed], coordinates[observed])
    prediction.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_graph_reconstructor_is_strict_and_below_one_million_parameters():
    graph = build_reconstruction_model(
        "graph",
        in_features=2,
        num_joints=10,
        joint_names=TRACKID3X3_JOINT_NAMES,
        edges=TRACKID3X3_EDGES,
    )
    assert count_parameters(graph) < 1_000_000
    canonical_npz_graph = build_reconstruction_model(
        "graph",
        in_features=2,
        num_joints=10,
        joint_names=TRACKID3X3_JOINT_NAMES,
        edges=TRACKID10_EDGES,
    )
    assert count_parameters(canonical_npz_graph) == count_parameters(graph)
    with pytest.raises(ValueError, match="10-joint"):
        build_reconstruction_model("graph", in_features=2, num_joints=17)
    wrong_names = list(TRACKID3X3_JOINT_NAMES)
    wrong_names[-1] = "right_knee"
    with pytest.raises(ValueError, match="joint_names"):
        build_reconstruction_model(
            "graph", in_features=2, num_joints=10, joint_names=wrong_names
        )


def test_tcn_rejects_noncontinuous_track_windows():
    model = build_reconstruction_model("tcn", in_features=2, num_joints=10)
    coordinates = torch.randn(2, 5, 10, 2)
    observed = torch.ones(2, 5, 10, dtype=torch.bool)
    track_valid = torch.ones(2, 5, dtype=torch.bool)
    track_valid[0, 2] = False
    with pytest.raises(ValueError, match="continuous valid track windows"):
        model(coordinates, observed, track_valid_mask=track_valid)


@pytest.mark.parametrize("ratio", (0.1, 0.2, 0.3, 0.4))
def test_reconstruction_masks_are_exact_per_instance_and_deterministic(ratio):
    original = torch.ones(2, 5, 10, dtype=torch.bool)
    original[0, 0, 0] = False
    first = deterministic_reconstruction_mask(
        original, ratio, seed=123, instance_ids=("track-a", "track-b")
    )
    second = deterministic_reconstruction_mask(
        original.flip(0), ratio, seed=123, instance_ids=("track-b", "track-a")
    ).flip(0)
    expected = [max(1, int(count * ratio + 0.5)) for count in (49, 50)]
    assert first.sum(dim=(1, 2)).tolist() == expected
    assert torch.equal(first, second)
    assert not bool((first & ~original).any())


def test_masked_loss_and_metrics_never_score_originally_missing_points():
    target = torch.zeros(1, 2, 2, 2)
    original = torch.tensor([[[False, True], [True, True]]])
    synthetic = torch.tensor([[[False, True], [False, False]]])
    prediction = target.clone()
    prediction[0, 0, 0] = 10_000  # Originally absent: must not affect result.
    prediction[0, 0, 1] = torch.tensor([3.0, 4.0])
    loss = masked_reconstruction_loss(
        prediction, target, original, synthetic, loss="mse"
    )
    metrics = reconstruction_metrics(
        prediction, target, original, synthetic, bbox_diagonal=torch.tensor([[10.0, 10.0]])
    )
    assert loss.item() == pytest.approx(12.5)
    assert metrics["mpjpe"] == pytest.approx(5.0)
    assert metrics["nmje"] == pytest.approx(0.5)
    assert metrics["pck@0.20"] == pytest.approx(0.0)
    assert metrics["per_joint_normalized_error"]["1"] == pytest.approx(0.5)
    assert metrics["scored_joint_count"] == 1


def test_masker_retains_all_three_mask_states():
    coordinates = torch.arange(24, dtype=torch.float32).reshape(1, 3, 4, 2)
    original = torch.ones(1, 3, 4, dtype=torch.bool)
    original[0, 0, 0] = False
    example = ReconstructionMasker(0.2, seed=4)(
        coordinates, original, instance_ids=("sample",)
    )
    assert torch.equal(
        example["observed_mask"], original & ~example["synthetic_mask"]
    )
    assert not bool((example["synthetic_mask"] & ~original).any())
    assert torch.all(example["inputs"][~example["observed_mask"]] == 0)


def test_temporal_interpolation_baseline_uses_nearest_edges_and_linear_middle():
    coordinates = torch.tensor(
        [[[[0.0, 0.0]], [[1.0, 2.0]], [[2.0, 4.0]], [[3.0, 6.0]]]]
    ).reshape(1, 4, 1, 2)
    observed = torch.tensor([[[False], [True], [False], [True]]])
    prediction = temporal_interpolation_baseline(coordinates, observed)
    assert torch.allclose(prediction[0, 0, 0], coordinates[0, 1, 0])
    assert torch.allclose(prediction[0, 2, 0], torch.tensor([2.0, 4.0]))


def test_reconstruction_representation_masks_before_observed_only_context():
    pixels = torch.arange(20, dtype=torch.float32).reshape(1, 10, 2) + 10
    original = torch.ones(1, 10, dtype=torch.bool)
    synthetic = torch.zeros_like(original)
    synthetic[:, 7] = True  # Hide the preferred centre anchor.
    changed_hidden_target = pixels.clone()
    changed_hidden_target[:, 7] = torch.tensor([99_999.0, -99_999.0])
    first = prepare_reconstruction_representation(
        pixels, original, synthetic, representation="normalized_angles"
    )
    second = prepare_reconstruction_representation(
        changed_hidden_target, original, synthetic, representation="normalized_angles"
    )
    assert torch.equal(first["inverse_offset"], second["inverse_offset"])
    assert torch.equal(first["inverse_scale"], second["inverse_scale"])
    assert torch.equal(first["inputs"], second["inputs"])
    assert torch.equal(first["auxiliary"], second["auxiliary"])
    assert torch.equal(first["auxiliary_mask"], second["auxiliary_mask"])


def test_bbox_representation_round_trips_to_source_pixels():
    pixels = torch.randn(2, 10, 2) * 5 + 50
    original = torch.ones(2, 10, dtype=torch.bool)
    synthetic = torch.zeros_like(original)
    synthetic[:, [3, 6]] = True
    boxes = torch.tensor([[10.0, 20.0, 80.0, 100.0], [8.0, 18.0, 90.0, 120.0]])
    prepared = prepare_reconstruction_representation(
        pixels,
        original,
        synthetic,
        representation="bbox_normalized",
        bbox_xywh=boxes,
    )
    restored = inverse_reconstruction_coordinates(
        prepared["targets"], prepared["inverse_offset"], prepared["inverse_scale"]
    )
    assert torch.allclose(restored[original], pixels[original], atol=1e-6)


@pytest.mark.parametrize(
    "representation",
    ("bbox_normalized", "root_centered", "scale_normalized", "normalized_angles"),
)
def test_pose_tight_bbox_is_metric_only_and_cannot_leak_hidden_drone_target(
    representation,
):
    pixels = torch.arange(20, dtype=torch.float32).reshape(1, 10, 2) + 5
    changed = pixels.clone()
    changed[:, 9] = torch.tensor([100_000.0, -100_000.0])
    original = torch.ones(1, 10, dtype=torch.bool)
    synthetic = torch.zeros_like(original)
    synthetic[:, 9] = True
    first = prepare_reconstruction_representation(
        pixels,
        original,
        synthetic,
        representation=representation,
        bbox_xywh=torch.tensor([[5.0, 6.0, 18.0, 18.0]]),
        bbox_valid=torch.tensor([True]),
        bbox_source=np.asarray(["pose_tight_fallback"]),
    )
    second = prepare_reconstruction_representation(
        changed,
        original,
        synthetic,
        representation=representation,
        bbox_xywh=torch.tensor([[5.0, -100_000.0, 100_000.0, 200_000.0]]),
        bbox_valid=torch.tensor([True]),
        bbox_source=np.asarray(["pose_tight_fallback"]),
    )
    for key in (
        "inputs",
        "auxiliary",
        "auxiliary_mask",
        "inverse_offset",
        "inverse_scale",
        "transform_bbox_xywh",
    ):
        if first[key].dtype == torch.bool:
            assert torch.equal(first[key], second[key])
        else:
            assert torch.allclose(first[key], second[key], equal_nan=True)
    assert bool(first["pose_tight_metric_only"].all())
    if representation == "bbox_normalized":
        assert first["transform_center_source"] == ["observed_pose_tight"]
    assert not torch.equal(first["bbox_diagonal"], second["bbox_diagonal"])


def test_scale_representation_uses_observed_torso_length():
    pixels = torch.zeros(1, 10, 2)
    pixels[:, 1] = torch.tensor([0.0, 0.0])
    pixels[:, 2] = torch.tensor([2.0, 0.0])
    pixels[:, 7] = torch.tensor([1.0, 4.0])
    original = torch.ones(1, 10, dtype=torch.bool)
    synthetic = torch.zeros_like(original)
    synthetic[:, 5] = True
    prepared = prepare_reconstruction_representation(
        pixels, original, synthetic, representation="scale_normalized"
    )
    assert torch.allclose(prepared["inverse_scale"], torch.full((1, 2), 4.0))


def test_normalized_angles_use_explicit_auxiliary_mask():
    pixels = torch.randn(1, 3, 10, 2)
    original = torch.ones(1, 3, 10, dtype=torch.bool)
    synthetic = torch.zeros_like(original)
    synthetic[:, :, 7] = True
    prepared = prepare_reconstruction_representation(
        pixels, original, synthetic, representation="normalized_angles"
    )
    assert prepared["auxiliary"].shape == (1, 3, 10, 1)
    assert not bool(prepared["auxiliary_mask"][:, :, [1, 2]].any())
    for name in RECONSTRUCTION_MODEL_NAMES:
        model = build_reconstruction_model(name, aux_features=1)
        output = model(
            prepared["inputs"],
            prepared["observed_mask"],
            auxiliary=prepared["auxiliary"],
            auxiliary_mask=prepared["auxiliary_mask"],
        )
        assert output.shape == pixels.shape


def test_reconstruction_trainer_writes_one_reloadable_best_checkpoint(tmp_path):
    items = []
    for index in range(4):
        pixels = torch.arange(20, dtype=torch.float32).reshape(1, 10, 2) + index
        original = torch.ones(1, 10, dtype=torch.bool)
        synthetic = deterministic_reconstruction_mask(
            original, 0.2, seed=9, instance_ids=f"sample-{index}"
        )
        prepared = prepare_reconstruction_representation(
            pixels,
            original,
            synthetic,
            representation="bbox_normalized",
            bbox_xywh=torch.tensor([[0.0, 0.0, 100.0, 100.0]]),
        )
        items.append({key: value for key, value in prepared.items() if torch.is_tensor(value)})
    loader = DataLoader(items, batch_size=2, shuffle=False)
    model_kwargs = {
        "in_features": 2,
        "num_joints": 10,
        "hidden_dims": (16, 8),
        "dropout": 0.0,
    }
    model = build_reconstruction_model("mlp", **model_kwargs)
    trainer = ReconstructionTrainer(
        model,
        ReconstructionTrainerConfig(
            epochs=1,
            batch_size=2,
            num_workers=0,
            mixed_precision=False,
            selection_metric="nmje",
        ),
        device="cpu",
        joint_names=list(TRACKID3X3_JOINT_NAMES),
    )
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    result = trainer.fit(
        loader,
        loader,
        run_dir=tmp_path / "logs",
        checkpoint_path=checkpoint,
        model_name="mlp",
        model_kwargs=model_kwargs,
        seed=9,
        resolved_config={"test": True},
        synthetic=True,
        project_root=tmp_path,
    )
    assert result.checkpoint_path == checkpoint
    assert [path.name for path in checkpoint.parent.glob("*.pt")] == ["best.pt"]
    assert {"config.yaml", "environment.json", "history.csv", "metrics.json"} <= {
        path.name for path in result.run_dir.iterdir()
    }
    restored, payload = load_reconstruction_checkpoint(checkpoint)
    assert payload["task"] == "keypoint_reconstruction"
    assert count_parameters(restored) == count_parameters(model)


def test_train_fraction_is_deterministic_and_never_splits_groups():
    data = {
        "domain": np.asarray(["Indoor"] * 4 + ["Outdoor"] * 4),
        "split_group": np.asarray(
            ["shared-a", "shared-a", "shared-b", "shared-b"]
            + ["shared-a", "shared-a", "shared-b", "shared-b"]
        ),
    }
    indices = np.arange(8)
    first, report = _fractional_train_indices(data, indices, fraction=0.5, seed=42)
    reordered, _ = _fractional_train_indices(
        data, indices[::-1], fraction=0.5, seed=42
    )
    assert set(first.tolist()) == set(reordered.tolist())
    selected_groups = set(
        zip(data["domain"][first], data["split_group"][first], strict=True)
    )
    for domain, group in selected_groups:
        group_rows = (data["domain"] == domain) & (data["split_group"] == group)
        assert set(np.flatnonzero(group_rows)) <= set(first.tolist())
    assert report["selected_groups"] == 2
    assert report["selected_samples"] == 4
    assert report["validation_unchanged"] is True
