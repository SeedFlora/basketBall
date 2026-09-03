from __future__ import annotations

import csv
import json
import pickle
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.archive import (
    ArchiveSafetyError,
    discover_allowed_archive,
    safe_extract_allowed_archive,
)
from src.data.augmentation import PoseAugmenter
from src.data.loader import (
    LABEL_TO_INDEX,
    PoseSample,
    PoseSchemaError,
    label_to_index,
    parse_sample_name,
    samples_from_object,
)
from src.data.normalization import normalize_pose
from src.data.preprocessing import preprocess_samples
from src.data.processed import PoseDataset, load_processed
from src.data.splitting import (
    assert_no_subject_leakage,
    exact_split_counts,
    select_subjects,
    split_subjects,
)
from src.data.storage import (
    MAX_STORAGE_MB,
    TARGET_STORAGE_MB,
    StorageBudgetError,
    build_storage_manifest,
    estimate_trackid_processed_bytes,
    storage_preflight,
    write_storage_manifest,
)
from src.data.trackid3x3 import (
    TRACKID10_JOINT_NAMES,
    TRACKID10_TO_COCO17,
    PoseInstance,
    load_trackid3x3_with_report,
)
from src.data.trackid_processed import (
    build_trackid_processed,
    load_trackid_processed,
    save_trackid_processed,
)
from src.data.trackid_split import (
    assign_domain_splits,
    select_domain_subset,
    select_temporal_windows,
)
from src.features.joint_angles import (
    TRACKID10_ANGLE_NAMES,
    angle_at_joint,
    compute_joint_angles,
    compute_trackid10_angles,
)
from src.features.skeleton_features import (
    COCO17_JOINT_NAMES,
    build_graph_representation,
    build_representation,
)
from src.features.temporal_features import compute_motion, temporal_resample
from src.features.trackid_features import (
    build_observed_representation,
    compute_trackid_angles,
    inverse_trackid_representation,
)
from src.utils import load_config


def make_samples(num_subjects: int = 20) -> list[PoseSample]:
    samples: list[PoseSample] = []
    for subject in range(1, num_subjects + 1):
        for action, label in LABEL_TO_INDEX.items():
            points = np.zeros((4 + subject % 3, 17, 2), dtype=np.float32)
            points[..., 0] = np.linspace(1.0, 2.0, points.shape[0])[:, None]
            points[..., 1] = np.arange(17, dtype=np.float32)[None, :] + 1.0
            mask = np.ones(points.shape[:2], dtype=bool)
            samples.append(
                PoseSample(points, mask, label, f"{subject}_{action}_1", str(subject))
            )
    return samples


def test_label_mapping_and_subject_name_parsing() -> None:
    assert label_to_index("d") == 0
    assert label_to_index("P") == 1
    assert label_to_index(np.int64(2)) == 2
    assert parse_sample_name("folder/57_p_2.mp4") == ("57", 1)
    assert parse_sample_name("unrelated57_p_2_extra.more") is None
    with pytest.raises(PoseSchemaError):
        label_to_index("dribble")


def test_mmaction_record_preserves_confidence_and_selects_best_person() -> None:
    pose = np.zeros((2, 3, 17, 2), dtype=np.float32)
    pose[0] = 1.0
    pose[1] = 5.0
    score = np.empty((2, 3, 17), dtype=np.float32)
    score[0] = 0.1
    score[1] = 0.9
    root = {
        "annotations": [
            {
                "frame_dir": "7_s_1",
                "label": 2,
                "keypoint": pose,
                "keypoint_score": score,
            }
        ],
        "fixture_metadata": {"synthetic": True},
    }
    sample = samples_from_object(root)[0]
    assert sample.keypoints.shape == (3, 17, 2)
    assert np.all(sample.keypoints == 5.0)
    assert sample.confidence is not None and np.allclose(sample.confidence, 0.9)
    assert sample.original_person_count == 2
    assert sample.person_selection_policy == "highest_aggregate_keypoint_score"


def test_name_label_mismatch_fails_loudly() -> None:
    record = {
        "frame_dir": "4_d_1",
        "label": 2,
        "keypoint": np.ones((1, 3, 17, 2), dtype=np.float32),
        "keypoint_score": np.ones((1, 3, 17), dtype=np.float32),
    }
    with pytest.raises(PoseSchemaError, match="conflicts"):
        samples_from_object([record])


def test_no_subject_leakage() -> None:
    samples = make_samples(20)
    assignment = split_subjects([str(index) for index in range(1, 21)], seed=42)
    assert_no_subject_leakage(
        [sample.subject_id for sample in samples],
        [assignment[sample.subject_id] for sample in samples],
    )
    with pytest.raises(AssertionError, match="Subject leakage"):
        assert_no_subject_leakage(["1", "1", "2"], ["train", "test", "val"])


def test_exact_split_counts_for_primary_sizes() -> None:
    assert exact_split_counts(200) == (140, 30, 30)
    assert exact_split_counts(100) == (70, 15, 15)
    assert sum(exact_split_counts(18)) == 18


def test_deterministic_balanced_subject_selection() -> None:
    samples = make_samples(30)
    first = select_subjects(samples, 10, seed=42)
    second = select_subjects(list(reversed(samples)), 10, seed=42)
    assert first == second
    assert len(first) == 10
    assert first != select_subjects(samples, 10, seed=123)
    selected_samples = [sample for sample in samples if sample.subject_id in first]
    assert np.bincount([sample.label for sample in selected_samples], minlength=3).tolist() == [10, 10, 10]


def test_temporal_uniform_interpolation_and_motion() -> None:
    points = np.zeros((2, 1, 2), dtype=np.float32)
    points[1] = 10.0
    mask = np.ones((2, 1), dtype=bool)
    output, output_mask, confidence = temporal_resample(points, mask, 5)
    assert confidence is None
    assert output.shape == (5, 1, 2)
    assert output_mask.all()
    assert np.allclose(output[:, 0, 0], [0.0, 2.5, 5.0, 7.5, 10.0])
    motion, motion_mask = compute_motion(output, output_mask)
    assert np.allclose(motion[1:, 0, 0], 2.5)
    assert not motion_mask[0, 0] and motion_mask[1:, 0].all()


def test_normalization_centres_hips_and_scales_shoulders() -> None:
    points = np.zeros((1, 17, 2), dtype=np.float32)
    mask = np.zeros((1, 17), dtype=bool)
    points[0, 11] = (-1.0, 0.0)
    points[0, 12] = (1.0, 0.0)
    points[0, 5] = (-1.0, 2.0)
    points[0, 6] = (1.0, 2.0)
    mask[0, [5, 6, 11, 12]] = True
    normalized, output_mask = normalize_pose(points, mask, COCO17_JOINT_NAMES)
    assert output_mask[0, [5, 6, 11, 12]].all()
    assert np.allclose((normalized[0, 11] + normalized[0, 12]) / 2, 0.0)
    assert np.isclose(np.linalg.norm(normalized[0, 5] - normalized[0, 6]), 1.0)
    assert np.all(normalized[~output_mask] == 0.0)


def test_joint_angle_computation_and_degenerate_mask() -> None:
    angle, valid = angle_at_joint(
        np.asarray([[1.0, 0.0]]),
        np.asarray([[0.0, 0.0]]),
        np.asarray([[0.0, 1.0]]),
    )
    assert valid.item()
    assert np.isclose(angle.item(), np.pi / 2)
    points = np.zeros((1, 17, 2), dtype=np.float32)
    mask = np.zeros((1, 17), dtype=bool)
    points[0, 5], points[0, 7], points[0, 9] = (1, 0), (0, 0), (0, 1)
    mask[0, [5, 7, 9]] = True
    angles, angle_mask, names = compute_joint_angles(points, mask, COCO17_JOINT_NAMES)
    left_elbow = names.index("left_elbow")
    assert angle_mask[0, left_elbow]
    assert np.isclose(angles[0, left_elbow], np.pi / 2)
    assert not angle_mask[0, names.index("right_elbow")]


def test_representation_shapes_and_masks() -> None:
    samples = make_samples(3)
    assignment = split_subjects(["1", "2", "3"], seed=42)
    arrays = preprocess_samples(samples, assignment, sequence_length=8, coordinate_dims=2)
    features, feature_mask, names = build_representation(
        arrays["keypoints"], arrays["valid_mask"], "normalized_motion_angles", arrays["joint_names"]
    )
    assert features.shape == feature_mask.shape == (9, 8, 76)
    assert len(names) == 76
    graph, graph_mask, channels = build_graph_representation(
        arrays["keypoints"], arrays["valid_mask"], "normalized_motion", arrays["joint_names"]
    )
    assert graph.shape == graph_mask.shape == (9, 8, 17, 4)
    assert channels == ["x", "y", "motion_x", "motion_y"]


def test_processed_contract_and_dataset_train_only_augmentation(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    samples = make_samples(6)
    assignment = split_subjects([str(index) for index in range(1, 7)], seed=42)
    arrays = preprocess_samples(samples, assignment, sequence_length=8, coordinate_dims=2)
    path = tmp_path / "dataset.npz"
    np.savez_compressed(path, **arrays)
    loaded = load_processed(path)
    assert loaded["keypoints"].dtype == np.float32
    assert loaded["keypoints"].shape == (18, 8, 17, 2)
    assert loaded["valid_mask"].dtype == bool
    assert loaded["confidence"].shape == (18, 8, 17)
    assert loaded["original_lengths"].shape == (18,)
    train = PoseDataset(path, split="train", representation="normalized", augment=True, seed=42)
    train.set_epoch(0)
    first = train[0]["features"].numpy()
    second = train[0]["features"].numpy()
    assert np.array_equal(first, second)
    train.set_epoch(1)
    assert not np.array_equal(first, train[0]["features"].numpy())
    with pytest.raises(ValueError, match="only"):
        PoseDataset(path, split="test", augment=True)


def test_pose_augmenter_is_conservative_and_deterministic() -> None:
    points = np.ones((8, 17, 2), dtype=np.float32)
    mask = np.ones((8, 17), dtype=bool)
    config = {
        "jitter_std": 0.001,
        "scale_range": (0.99, 1.01),
        "translation_std": 0.001,
        "frame_dropout": 0.0,
        "joint_dropout": 0.0,
    }
    first = PoseAugmenter(config, seed=7)(points, mask)
    second = PoseAugmenter(config, seed=7)(points, mask)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.max(np.abs(first[0] - points)) < 0.05


def test_allowed_archive_discovery_is_recursive_and_extraction_is_safe(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "download"
    nested.mkdir(parents=True)
    archive = nested / "MultiSubjects_SA.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("payload/train.pkl", b"trusted-fixture-placeholder")
    assert discover_allowed_archive(tmp_path) == archive.resolve()
    extracted = safe_extract_allowed_archive(archive, tmp_path / "extract")
    assert [path.relative_to(tmp_path / "extract").as_posix() for path in extracted] == [
        "payload/train.pkl"
    ]


def test_archive_traversal_is_rejected_before_writing(tmp_path: Path) -> None:
    archive = tmp_path / "MultiSubjects_SA.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.pkl", b"unsafe")
    with pytest.raises(ArchiveSafetyError, match="Unsafe"):
        safe_extract_allowed_archive(archive, tmp_path / "extract")
    assert not (tmp_path / "escape.pkl").exists()


def _trackid_instance(
    domain: str,
    frame: int,
    player: str,
    *,
    video: str | None,
) -> PoseInstance:
    points = np.stack(
        [np.arange(10, dtype=np.float32) + 10, np.arange(10, dtype=np.float32) + 20],
        axis=-1,
    )
    source = "pose_tight_fallback" if domain == "Drone" else "mot_xywh"
    return PoseInstance(
        keypoints=points,
        valid_mask=np.ones(10, dtype=bool),
        domain=domain,
        frame_index=frame,
        player_id=player,
        image_size=(400, 300),
        sample_id=f"{domain}:{video or 'none'}:{frame}:{player}",
        video_id=video,
        bbox_xywh=np.asarray([5, 5, 40, 60], dtype=np.float32),
        bbox_source=source,
        mot_track_id=None if domain == "Drone" else "1",
        association_confidence=np.nan if domain == "Drone" else 1.0,
        temporal_reliable=domain != "Drone",
    )


def test_trackid_exact_joint_mapping_has_no_invented_joint() -> None:
    assert TRACKID10_JOINT_NAMES == (
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "center",
        "left_ankle",
        "right_ankle",
    )
    assert TRACKID10_TO_COCO17 == (0, 5, 6, 7, 8, 9, 10, None, 15, 16)
    numbered = _trackid_instance("Outdoor", 0, "Green_11", video="single")
    assert numbered.player_number == "11"
    assert _trackid_instance("Indoor", 0, "Green_Jersey", video="v").player_number is None


def test_trackid_json_layout_and_mot_bbox_provenance(tmp_path: Path) -> None:
    ground_truth = tmp_path / "TrackID3x3" / "ground_truth"
    for domain in ("Indoor", "Outdoor", "Drone"):
        (ground_truth / domain / "pose").mkdir(parents=True)
    (ground_truth / "Indoor" / "MOT").mkdir()
    first = [[10 + index % 2, 10 + index] for index in range(10)]
    second = [[110 + index % 2, 10 + index] for index in range(10)]
    indoor = {
        "video": {
            "file_name": "tiny.mp4",
            "resolution": [400, 200],
            "frame_rate": 20,
            "total_frames": 1,
        },
        "annotations": {"frame_0": {"Blue": second, "Red": first}},
    }
    (ground_truth / "Indoor" / "pose" / "tiny.json").write_text(
        json.dumps(indoor), encoding="utf-8"
    )
    (ground_truth / "Indoor" / "MOT" / "tiny.txt").write_text(
        "1,1,0,0,50,100,-1,-1,-1,-1\n1,2,100,0,50,100,-1,-1,-1,-1\n",
        encoding="utf-8",
    )
    drone = {
        "image": {"file_name": "frame_7", "resolution": [400, 200]},
        "annotations": {"White_0": first},
    }
    (ground_truth / "Drone" / "pose" / "frame_7.json").write_text(
        json.dumps(drone), encoding="utf-8"
    )

    indoor_result = load_trackid3x3_with_report(tmp_path, domains=("Indoor",))
    assert len(indoor_result.instances) == 2
    assert {item.bbox_source for item in indoor_result.instances} == {"mot_xywh"}
    assert {item.player_id: item.mot_track_id for item in indoor_result.instances} == {
        "Blue": "2",
        "Red": "1",
    }
    drone_result = load_trackid3x3_with_report(tmp_path, domains=("Drone",))
    assert len(drone_result.instances) == 1
    assert drone_result.instances[0].bbox_source == "pose_tight_fallback"
    assert drone_result.instances[0].video_id is None
    assert not drone_result.instances[0].temporal_reliable


def test_real_trackid_source_counts_when_checkout_is_present() -> None:
    checkout = ROOT / "external" / "TrackID3x3"
    if not checkout.is_dir():
        pytest.skip("External TrackID3x3 checkout is unavailable.")
    result = load_trackid3x3_with_report(checkout, attach_bboxes=False)
    facts = result.diagnostics["domains"]
    assert (facts["Indoor"]["annotated_frames"], facts["Indoor"]["pose_instances_input"]) == (
        2601,
        15606,
    )
    assert (facts["Outdoor"]["annotated_frames"], facts["Outdoor"]["pose_instances_input"]) == (
        3600,
        21528,
    )
    assert (facts["Drone"]["annotated_frames"], facts["Drone"]["pose_instances_input"]) == (
        500,
        3000,
    )
    assert facts["Outdoor"]["missing_keypoints"] == 1


def test_trackid_group_splits_and_subset_are_deterministic() -> None:
    instances: list[PoseInstance] = []
    for video_index in range(13):
        for frame in range(9):
            instances.append(
                _trackid_instance("Indoor", frame, "Red", video=f"video_{video_index}")
            )
    for domain in ("Outdoor", "Drone"):
        for player_index in range(6):
            for frame in range(9):
                instances.append(
                    _trackid_instance(
                        domain,
                        frame,
                        f"player_{player_index}",
                        video="single" if domain == "Outdoor" else None,
                    )
                )
    assigned = assign_domain_splits(instances, seed=42)
    group_counts = {
        domain: Counter(
            next(item.split for item in assigned if item.split_group == group)
            for group in {item.split_group for item in assigned if item.domain == domain}
        )
        for domain in ("Indoor", "Outdoor", "Drone")
    }
    assert group_counts["Indoor"] == {"train": 9, "val": 2, "test": 2}
    assert group_counts["Outdoor"] == {"train": 4, "val": 1, "test": 1}
    assert group_counts["Drone"] == {"train": 4, "val": 1, "test": 1}
    first, first_report = select_domain_subset(
        instances, targets={"Indoor": 20, "Outdoor": 12, "Drone": 12}, seed=42
    )
    second, _ = select_domain_subset(
        list(reversed(instances)),
        targets={"Indoor": 20, "Outdoor": 12, "Drone": 12},
        seed=42,
    )
    assert [item.sample_id for item in first] == [item.sample_id for item in second]
    assert len(first) == 44
    assert first_report["domains"]["Indoor"]["selected_instances"] == 20


def test_temporal_selection_uses_pre_thinning_reliable_streams() -> None:
    instances = []
    for domain in ("Indoor", "Outdoor", "Drone"):
        for frame in range(48):
            instances.append(
                _trackid_instance(
                    domain,
                    frame,
                    "player_0",
                    video="video_0" if domain != "Drone" else None,
                )
            )
    windows, report = select_temporal_windows(
        instances,
        length=4,
        frame_step=3,
        target_observations={"Indoor": 100, "Outdoor": 100, "Drone": 100},
    )
    assert windows
    assert {window.instances[0].domain for window in windows} == {"Indoor", "Outdoor"}
    assert all(np.all(np.diff([item.frame_index for item in window.instances]) == 3) for window in windows)
    assert report["domains"]["Drone"]["selected_windows"] == 0
    assert report["domains"]["Drone"]["exclusion"]


def test_observed_only_root_scale_does_not_leak_hidden_target() -> None:
    points = np.zeros((1, 10, 2), dtype=np.float32)
    points[0, 1] = (0, 0)
    points[0, 2] = (2, 0)
    points[0, 7] = (1, 2)
    points[0, 3] = (-1, 1)
    points[0, 4] = (3, 1)
    points[0, 5] = (-2, 2)
    points[0, 6] = (4, 2)
    points[0, 8] = (0, 5)
    points[0, 9] = (2, 5)
    points[0, 0] = (1, -1)
    valid = np.ones((1, 10), dtype=bool)
    observed = valid.copy()
    observed[0, 7] = False
    boxes = np.asarray([[0, 0, 10, 10]], dtype=np.float32)
    first = build_observed_representation(
        points, valid, observed, "scale_normalized", bboxes=boxes
    )
    changed = points.copy()
    changed[0, 7] = (10000, -10000)
    second = build_observed_representation(
        changed, valid, observed, "scale_normalized", bboxes=boxes
    )
    assert np.array_equal(first["inputs"], second["inputs"])
    assert np.array_equal(first["inverse_offset"], second["inverse_offset"])
    assert np.array_equal(first["inverse_scale"], second["inverse_scale"])
    assert first["center_source"].item() == "observed_shoulders"
    reconstructed = inverse_trackid_representation(
        first["targets"], first["inverse_offset"], first["inverse_scale"]
    )
    assert np.allclose(reconstructed, points)


def test_trackid_bbox_normalization_angles_and_missing_masks() -> None:
    points = np.zeros((1, 10, 2), dtype=np.float32)
    points[0, 1], points[0, 3], points[0, 5] = (1, 0), (0, 0), (0, 1)
    points[0, 2], points[0, 4], points[0, 6] = (3, 0), (4, 0), (4, 1)
    points[0, 7] = (2, 2)
    mask = np.zeros((1, 10), dtype=bool)
    mask[0, [1, 2, 3, 4, 5, 6, 7]] = True
    bbox = np.asarray([[10, 20, 100, 200]], dtype=np.float32)
    represented = build_observed_representation(
        points, mask, mask, "bbox_normalized", bboxes=bbox
    )
    restored = inverse_trackid_representation(
        represented["targets"], represented["inverse_offset"], represented["inverse_scale"]
    )
    assert np.allclose(restored[mask], points[mask])
    angles, angle_mask, names = compute_trackid_angles(points, mask)
    assert angles.shape == angle_mask.shape == (1, 5)
    assert angle_mask[0, names.index("left_elbow")]
    assert not angle_mask[0, names.index("shoulder_center")] or np.isfinite(
        angles[0, names.index("shoulder_center")]
    )
    assert np.all(angles[~angle_mask] == 0)


def test_trackid_angle_api_clamps_cosine_and_masks_epsilon_degeneracy() -> None:
    near_collinear, valid = angle_at_joint(
        np.asarray([[1.0, 1e-7]], dtype=np.float32),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([[2.0, 2e-7]], dtype=np.float32),
    )
    assert valid.item() and np.isfinite(near_collinear.item())
    points = np.zeros((1, 10, 2), dtype=np.float32)
    points[0, 1] = (1, 0)
    points[0, 3] = (0, 0)
    points[0, 5] = (0, 1)
    mask = np.ones((1, 10), dtype=bool)
    angles, angle_mask, names = compute_trackid10_angles(points, mask, epsilon=1e-6)
    assert names == TRACKID10_ANGLE_NAMES
    assert angle_mask[0, names.index("left_elbow")]
    assert np.isclose(angles[0, names.index("left_elbow")], np.pi / 2)
    assert not angle_mask[0, names.index("right_elbow")]
    assert angles[0, names.index("right_elbow")] == 0


def test_drone_pose_tight_bbox_transform_uses_observed_joints_only() -> None:
    points = np.stack(
        [np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32) * 2], axis=-1
    )[None]
    valid = np.ones((1, 10), dtype=bool)
    observed = valid.copy()
    observed[0, 9] = False
    stored_full_pose_bbox = np.asarray([[0, 0, 9, 18]], dtype=np.float32)
    first = build_observed_representation(
        points,
        valid,
        observed,
        "bbox_normalized",
        bboxes=stored_full_pose_bbox,
        bbox_sources=np.asarray(["pose_tight_fallback"]),
    )
    changed = points.copy()
    changed[0, 9] = (100000, -100000)
    second = build_observed_representation(
        changed,
        valid,
        observed,
        "bbox_normalized",
        bboxes=stored_full_pose_bbox,
        bbox_sources=np.asarray(["pose_tight_fallback"]),
    )
    assert np.array_equal(first["inputs"], second["inputs"])
    assert np.array_equal(first["inverse_offset"], second["inverse_offset"])
    assert np.array_equal(first["inverse_scale"], second["inverse_scale"])
    assert first["center_source"].item() == "observed_pose_tight"
    assert np.allclose(first["transform_bbox_xywh"], [[0, 0, 8, 16]])


def test_trackid_processed_contract_aliases_and_shapes(tmp_path: Path) -> None:
    samples = assign_domain_splits(
        [
            _trackid_instance("Indoor", 0, "Red", video="video_0"),
            _trackid_instance("Drone", 7, "White_0", video=None),
        ]
    )
    arrays = build_trackid_processed(samples, mask_rate=0.3, seed=42)
    assert arrays["coords"].shape == (2, 1, 10, 2)
    assert arrays["valid_mask"].shape == arrays["input_mask"].shape == arrays["target_mask"].shape
    assert np.array_equal(arrays["input_mask"] | arrays["target_mask"], arrays["valid_mask"])
    assert arrays["bbox"].shape == (2, 1, 4)
    assert arrays["angles"].shape == (2, 1, 5)
    save_trackid_processed(tmp_path, arrays, {"synthetic": True})
    loaded = load_trackid_processed(tmp_path)
    assert loaded["keypoints"] is loaded["coords"]
    assert loaded["bboxes"] is loaded["bbox"]
    assert loaded["domains"].tolist() == ["Indoor", "Drone"]


def test_trackid_storage_manifest_and_preflight(tmp_path: Path) -> None:
    canonical = tmp_path / "data" / "processed" / "trackid3x3"
    temporal = tmp_path / "data" / "processed" / "trackid3x3_t16"
    canonical.mkdir(parents=True)
    temporal.mkdir(parents=True)
    (canonical / "dataset.npz").write_bytes(b"123")
    (temporal / "dataset.npz").write_bytes(b"4567")
    entries = build_storage_manifest(tmp_path)
    processed_entries = [entry for entry in entries if entry.category == "processed"]
    assert len(processed_entries) == 1
    assert processed_entries[0].path == "data/processed"
    assert processed_entries[0].size_mb == 0.000007
    output = write_storage_manifest(tmp_path / "storage.csv", entries)
    with output.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["path", "category", "size_mb", "required", "source"]
        assert {row["category"] for row in reader}.issubset(
            {"annotations", "videos", "processed", "checkpoints", "figures", "temporary"}
        )
    estimate = estimate_trackid_processed_bytes(2500, 1)
    assert MAX_STORAGE_MB == 600 and TARGET_STORAGE_MB == 500
    approved = storage_preflight(
        tmp_path, estimate, max_artifact_bytes=estimate, minimum_free_bytes=0
    )
    assert approved["approved"]
    with pytest.raises(StorageBudgetError):
        storage_preflight(
            tmp_path, estimate, max_artifact_bytes=estimate - 1, minimum_free_bytes=0
        )

    report = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_storage.py"),
            "--project-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "reported.csv"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert report.returncode == 0, report.stderr
    for label in (
        "Total raw data size:",
        "Total processed data size:",
        "Checkpoint size:",
        "Total experiment footprint:",
    ):
        assert label in report.stdout


@pytest.mark.parametrize(
    "name",
    (
        "sample_50.yaml",
        "sample_100.yaml",
        "sample_200.yaml",
        "sample_500.yaml",
        "full_skeleton.yaml",
        "smoke_synthetic.yaml",
        "smoke_synthetic_aug.yaml",
        "smoke_ablation.yaml",
    ),
)
def test_legacy_henu_configs_are_explicitly_quarantined(name: str) -> None:
    config = load_config(ROOT / "config" / name)
    legacy = config["legacy_workflow"]
    assert legacy["status"] == "disabled"
    assert legacy["dataset_family"] in {
        "Henu_MultiSubjects_action_classification",
        "fabricated_Henu_MMAAction_classification",
    }
    assert legacy["trackid3x3_compatible"] is False
    assert legacy["research_results"] is False
    assert legacy["replacement_config"] == "config/trackid3x3.yaml"
    assert config["data"]["processed_dir"] == "data/processed/LEGACY_HENU_CONFIG_DISABLED"
    assert config["features"]["representation"] == "legacy_henu_classification_disabled"


def test_legacy_henu_config_fails_fast_in_public_training_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--config",
            str(ROOT / "config" / "sample_50.yaml"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "legacy_henu_classification_disabled" in result.stderr


def test_legacy_fixture_generator_is_disabled_without_explicit_opt_in(
    tmp_path: Path,
) -> None:
    script = ROOT / "scripts" / "create_synthetic_fixture.py"
    denied_output = tmp_path / "denied"
    denied = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(denied_output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert denied.returncode == 2
    assert "disabled for the public TrackID3x3 workflow" in denied.stderr
    assert not denied_output.exists()

    allowed_output = tmp_path / "legacy_fixture"
    allowed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--allow-legacy-henu-fixture",
            "--output-dir",
            str(allowed_output),
            "--num-subjects",
            "6",
            "--seed",
            "42",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert allowed.returncode == 0, allowed.stderr
    assert "not TrackID3x3 and not research evidence" in allowed.stderr
    assert "TrackID3x3 compatible: false" in (
        allowed_output / "LEGACY_HENU_FIXTURE_ONLY.txt"
    ).read_text(encoding="utf-8")
    with (allowed_output / "train.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    metadata = payload["fixture_metadata"]
    assert metadata["synthetic"] is True
    assert metadata["research_results"] is False
    assert metadata["legacy_henu_fixture"] is True
    assert metadata["trackid3x3_compatible"] is False
