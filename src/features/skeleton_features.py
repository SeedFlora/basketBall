"""Joint layouts and composable skeleton feature representations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

COCO17_JOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

MEDIAPIPE33_JOINT_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

COCO17_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

MEDIAPIPE33_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
)


def infer_joint_layout(num_joints: int) -> str:
    """Return a named layout only when joint count uniquely identifies it."""

    if num_joints == 17:
        return "coco17"
    if num_joints == 33:
        return "mediapipe33"
    return "unknown"


def joint_names_and_edges(num_joints: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Return names/graph edges with a safe empty-graph fallback."""

    layout = infer_joint_layout(num_joints)
    if layout == "coco17":
        names = COCO17_JOINT_NAMES
        edges = COCO17_EDGES
    elif layout == "mediapipe33":
        names = MEDIAPIPE33_JOINT_NAMES
        edges = MEDIAPIPE33_EDGES
    else:
        names = tuple(f"joint_{index}" for index in range(num_joints))
        edges = ()
    edge_array = (
        np.asarray(edges, dtype=np.int64).T
        if edges
        else np.empty((2, 0), dtype=np.int64)
    )
    return np.asarray(names, dtype=np.str_), edge_array, layout


def _coordinate_names(coordinate_count: int) -> tuple[str, ...]:
    standard = ("x", "y", "z", "c3")
    return tuple(standard[index] if index < len(standard) else f"c{index}" for index in range(coordinate_count))


def build_representation(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    representation: str,
    joint_names: Sequence[str] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build raw, normalised, motion, angle, or fused temporal features.

    Args:
        keypoints: ``[T,V,C]`` or batched ``[N,T,V,C]`` coordinates.
        valid_mask: Matching ``[T,V]`` or ``[N,T,V]`` joint mask.
        representation: One of ``raw``, ``normalized``, ``angles``,
            ``normalized_motion``, ``normalized_angles`` or
            ``normalized_motion_angles``.  Common ``skeleton+...`` aliases are
            accepted after replacing ``+`` with ``_``.
        joint_names: Optional exact layout names.  Joint count is used only for
            the unambiguous COCO17/MediaPipe33 fallback.

    Returns:
        ``(features, feature_mask, feature_names)``.  The first two arrays mirror
        the input batch rank and end in feature dimension ``F``.
    """

    from src.data.normalization import normalize_pose
    from src.features.joint_angles import compute_joint_angles
    from src.features.temporal_features import compute_motion

    points = np.asarray(keypoints, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    single = points.ndim == 3
    if single:
        points = points[None, ...]
        mask = mask[None, ...]
    if points.ndim != 4 or mask.shape != points.shape[:3]:
        raise ValueError(
            f"Expected keypoints [N,T,V,C] and mask [N,T,V], got {points.shape}, {mask.shape}."
        )
    if points.shape[-1] < 2:
        raise ValueError("At least x/y coordinates are required.")
    if joint_names is None:
        inferred_names, _, _ = joint_names_and_edges(points.shape[2])
        names = inferred_names.tolist()
    else:
        names = [str(item) for item in np.asarray(joint_names).tolist()]
        if len(names) != points.shape[2]:
            raise ValueError(f"joint_names has {len(names)} entries, expected {points.shape[2]}.")

    token = representation.strip().lower().replace("+", "_").replace("-", "_")
    aliases = {
        "joint_angles": "angles",
        "normalized_skeleton": "normalized",
        "normalised": "normalized",
        "normalized_skeleton_motion": "normalized_motion",
        "normalized_skeleton_angles": "normalized_angles",
        "skeleton_angles": "normalized_angles",
        "normalized_skeleton_motion_angles": "normalized_motion_angles",
    }
    token = aliases.get(token, token)
    allowed = {
        "raw",
        "normalized",
        "angles",
        "normalized_motion",
        "normalized_angles",
        "normalized_motion_angles",
    }
    if token not in allowed:
        raise ValueError(f"Unknown representation {representation!r}; expected one of {sorted(allowed)}.")

    normalized = np.empty_like(points)
    normalized_mask = np.empty_like(mask)
    if token != "raw":
        for index in range(points.shape[0]):
            normalized[index], normalized_mask[index] = normalize_pose(points[index], mask[index], names)
    else:
        normalized = points.copy()
        normalized_mask = mask.copy()
        normalized[~normalized_mask] = 0.0

    pieces: list[np.ndarray] = []
    mask_pieces: list[np.ndarray] = []
    feature_names: list[str] = []
    coordinate_names = _coordinate_names(points.shape[-1])

    if token != "angles":
        skeleton = normalized.reshape(*normalized.shape[:2], -1)
        skeleton_mask = np.repeat(normalized_mask[..., None], points.shape[-1], axis=-1).reshape(
            *normalized_mask.shape[:2], -1
        )
        pieces.append(skeleton)
        mask_pieces.append(skeleton_mask)
        prefix = "raw" if token == "raw" else "normalized"
        feature_names.extend(
            f"{prefix}.{joint}.{coordinate}"
            for joint in names
            for coordinate in coordinate_names
        )

    if "motion" in token:
        motions: list[np.ndarray] = []
        motion_masks: list[np.ndarray] = []
        for index in range(points.shape[0]):
            motion, motion_mask = compute_motion(normalized[index], normalized_mask[index])
            motions.append(motion)
            motion_masks.append(motion_mask)
        motion_array = np.stack(motions).reshape(*normalized.shape[:2], -1)
        motion_mask_array = np.repeat(
            np.stack(motion_masks)[..., None], points.shape[-1], axis=-1
        ).reshape(*normalized_mask.shape[:2], -1)
        pieces.append(motion_array)
        mask_pieces.append(motion_mask_array)
        feature_names.extend(
            f"motion.{joint}.{coordinate}" for joint in names for coordinate in coordinate_names
        )

    if "angles" in token or token == "angles":
        angle_arrays: list[np.ndarray] = []
        angle_masks: list[np.ndarray] = []
        angle_names: list[str] | None = None
        for index in range(points.shape[0]):
            angles, angles_mask, current_names = compute_joint_angles(
                normalized[index], normalized_mask[index], names
            )
            if angle_names is None:
                angle_names = current_names
            angle_arrays.append(angles)
            angle_masks.append(angles_mask)
        if angle_names is None or not angle_names:
            raise ValueError(
                f"No safe joint-angle mapping is available for {points.shape[2]} joints/names."
            )
        pieces.append(np.stack(angle_arrays))
        mask_pieces.append(np.stack(angle_masks))
        feature_names.extend(f"angle.{name}" for name in angle_names)

    features = np.concatenate(pieces, axis=-1).astype(np.float32, copy=False)
    feature_mask = np.concatenate(mask_pieces, axis=-1).astype(bool, copy=False)
    features = np.where(feature_mask, features, 0.0).astype(np.float32, copy=False)
    if single:
        return features[0], feature_mask[0], feature_names
    return features, feature_mask, feature_names


def build_graph_representation(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    representation: str = "normalized",
    joint_names: Sequence[str] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build graph-preserving ``[...,T,V,Cg]`` input for ST-GCN models.

    ``raw`` and ``normalized`` retain coordinate channels; ``normalized_motion``
    concatenates displacement channels at every joint.  Global joint-angle
    features are intentionally rejected because assigning an A-B-C angle to all
    graph nodes would invent a graph semantics.
    """

    from src.data.normalization import normalize_pose
    from src.features.temporal_features import compute_motion

    points = np.asarray(keypoints, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    single = points.ndim == 3
    if single:
        points = points[None, ...]
        mask = mask[None, ...]
    if points.ndim != 4 or mask.shape != points.shape[:3]:
        raise ValueError(f"Expected [N,T,V,C] and [N,T,V], got {points.shape}, {mask.shape}.")
    token = representation.strip().lower().replace("+", "_").replace("-", "_")
    token = {
        "normalized_skeleton": "normalized",
        "normalized_skeleton_motion": "normalized_motion",
    }.get(token, token)
    if token not in {"raw", "normalized", "normalized_motion"}:
        raise ValueError(
            "Graph representation supports raw, normalized, or normalized_motion; "
            "joint angles are global features and are not graph-aligned."
        )
    if joint_names is None:
        inferred, _, _ = joint_names_and_edges(points.shape[2])
        names = inferred.tolist()
    else:
        names = [str(item) for item in np.asarray(joint_names).tolist()]
    if len(names) != points.shape[2]:
        raise ValueError(f"joint_names has {len(names)} entries, expected {points.shape[2]}.")
    graph = np.empty_like(points)
    graph_mask = mask.copy()
    if token == "raw":
        graph = np.where(mask[..., None], points, 0.0)
    else:
        for index in range(points.shape[0]):
            graph[index], graph_mask[index] = normalize_pose(points[index], mask[index], names)
    channel_names = list(_coordinate_names(points.shape[-1]))
    if token == "normalized_motion":
        motions = np.zeros_like(graph)
        motion_masks = np.zeros_like(graph_mask)
        for index in range(points.shape[0]):
            motions[index], motion_masks[index] = compute_motion(graph[index], graph_mask[index])
        graph = np.concatenate([graph, motions], axis=-1)
        # Channel-level validity is needed because motion at t=0 is undefined.
        channel_mask = np.concatenate(
            [
                np.repeat(graph_mask[..., None], points.shape[-1], axis=-1),
                np.repeat(motion_masks[..., None], points.shape[-1], axis=-1),
            ],
            axis=-1,
        )
        channel_names += [f"motion_{name}" for name in channel_names]
    else:
        channel_mask = np.repeat(graph_mask[..., None], graph.shape[-1], axis=-1)
    graph = np.where(channel_mask, graph, 0.0).astype(np.float32, copy=False)
    if single:
        return graph[0], channel_mask[0], channel_names
    return graph, channel_mask, channel_names
