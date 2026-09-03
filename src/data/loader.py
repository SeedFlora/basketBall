"""Archived schema-aware loading for pre-pivot pose pickle fixtures.

This compatibility module cannot acquire Henu data and is not imported by the
public TrackID3x3 JSON/MOT pipeline. It recognises a deliberately small set of
historical pickle structures and fails on ambiguity instead of guessing fields.
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LABEL_TO_INDEX = {"d": 0, "p": 1, "s": 2}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}

_POSE_KEYS = {
    "keypoints",
    "keypoint",
    "key_points",
    "pose",
    "poses",
    "skeleton",
    "skeletons",
    "coordinates",
    "coords",
    "data",
    "x",
}
_LABEL_KEYS = {"label", "labels", "action", "actions", "class", "target", "targets", "y"}
_ID_KEYS = {
    "sample_id",
    "sample_ids",
    "sample_name",
    "sample_names",
    "name",
    "names",
    "filename",
    "filenames",
    "file_name",
    "file_names",
    "video_id",
    "video_ids",
    "video_name",
    "video_names",
    "frame_dir",
    "frame_dirs",
    "id",
    "ids",
}
_SUBJECT_KEYS = {"subject_id", "subject_ids", "subject", "subjects", "person_id", "person_ids"}
_MASK_KEYS = {
    "valid_mask",
    "valid_masks",
    "mask",
    "masks",
    "keypoint_mask",
}
_CONFIDENCE_KEYS = {
    "keypoint_score",
    "keypoint_scores",
    "confidence",
    "confidences",
}
_VALIDITY_KEYS = _MASK_KEYS | _CONFIDENCE_KEYS
_KNOWN_JOINT_COUNTS = {15, 16, 17, 18, 20, 21, 25, 26, 33}
_NAME_PATTERN = re.compile(
    r"^(?P<subject>\d+)[_-](?P<action>[dps])(?:[_-](?P<take>[A-Za-z0-9][A-Za-z0-9_-]*))?$",
    flags=re.IGNORECASE,
)


class PoseSchemaError(ValueError):
    """Raised when a pickle record cannot be interpreted without guessing."""


@dataclass(slots=True)
class PoseSample:
    """Canonical representation of one labelled pose sequence.

    Attributes:
        keypoints: Floating point coordinates with shape ``[T, V, C]``.
        valid_mask: Joint validity mask with shape ``[T, V]``.
        label: Integer action index (dribble=0, lay-up=1, shoot=2).
        sample_id: Stable sample identifier recovered from the source record.
        subject_id: Stable subject identifier recovered explicitly or from the
            conservative ``<subject>_<action>_<take>`` name convention.
        source: Optional source pickle name, used only for audit messages.
    """

    keypoints: np.ndarray
    valid_mask: np.ndarray
    label: int
    sample_id: str
    subject_id: str
    source: str = ""
    confidence: np.ndarray | None = None
    original_person_count: int = 1
    person_selection_policy: str = "single_person"

    def __post_init__(self) -> None:
        self.keypoints = np.asarray(self.keypoints, dtype=np.float32)
        self.valid_mask = np.asarray(self.valid_mask, dtype=bool)
        self.label = label_to_index(self.label)
        self.sample_id = str(self.sample_id).strip()
        self.subject_id = str(self.subject_id).strip()
        if self.keypoints.ndim != 3:
            raise PoseSchemaError(
                f"Canonical keypoints must have shape [T,V,C], got {self.keypoints.shape}."
            )
        if self.keypoints.shape[0] < 1 or self.keypoints.shape[1] < 1:
            raise PoseSchemaError(f"Pose sequence has an empty time/joint axis: {self.keypoints.shape}.")
        if self.keypoints.shape[2] < 2:
            raise PoseSchemaError(f"Pose coordinates need at least x/y, got C={self.keypoints.shape[2]}.")
        if self.valid_mask.shape != self.keypoints.shape[:2]:
            raise PoseSchemaError(
                f"valid_mask {self.valid_mask.shape} does not match [T,V] {self.keypoints.shape[:2]}."
            )
        if not self.sample_id:
            raise PoseSchemaError("sample_id is empty; a stable source identifier is required.")
        if not self.subject_id:
            raise PoseSchemaError("subject_id is empty; subject-disjoint splitting is impossible.")
        # Invalid values never enter downstream numeric features.
        finite = np.isfinite(self.keypoints).all(axis=-1)
        self.valid_mask &= finite
        self.keypoints = np.where(self.valid_mask[..., None], self.keypoints, 0.0).astype(
            np.float32, copy=False
        )
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, dtype=np.float32)
            if self.confidence.shape != self.valid_mask.shape:
                raise PoseSchemaError(
                    f"confidence {self.confidence.shape} does not match [T,V] {self.valid_mask.shape}."
                )
            self.confidence = np.where(
                self.valid_mask & np.isfinite(self.confidence), self.confidence, 0.0
            ).astype(np.float32, copy=False)
        if self.original_person_count < 1:
            raise PoseSchemaError("original_person_count must be positive.")


def label_to_index(value: Any) -> int:
    """Map the documented d/p/s labels to 0/1/2, rejecting unknown values."""

    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (np.integer, int)) and not isinstance(value, (bool, np.bool_)):
        label = int(value)
        if label in INDEX_TO_LABEL:
            return label
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PoseSchemaError("Label bytes are not valid UTF-8.") from exc
    if isinstance(value, str):
        token = value.strip().lower()
        if token in LABEL_TO_INDEX:
            return LABEL_TO_INDEX[token]
        if token in {"0", "1", "2"}:
            return int(token)
    raise PoseSchemaError(
        f"Unsupported action label {value!r}; expected one of d/p/s or integer 0/1/2."
    )


def parse_sample_name(value: Any) -> tuple[str, int] | None:
    """Parse ``<numeric-subject>_<d|p|s>_<take>`` from a file/sample name.

    The match is intentionally anchored.  Returning ``None`` is safer than
    treating an unrelated number embedded in a filename as a subject identity.
    """

    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(value, (str, Path)):
        return None
    name = str(value).replace("\\", "/").rsplit("/", 1)[-1].strip()
    # Remove common media/data suffixes while retaining dots inside identifiers.
    for _ in range(2):
        suffix = Path(name).suffix.lower()
        if suffix in {".avi", ".mp4", ".mov", ".npy", ".npz", ".pkl", ".pickle", ".csv"}:
            name = name[: -len(suffix)]
        else:
            break
    match = _NAME_PATTERN.fullmatch(name)
    if match is None:
        return None
    return match.group("subject"), LABEL_TO_INDEX[match.group("action").lower()]


def _numeric_array(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray) and value.dtype != object:
        array = value
    elif isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return None
        if array.dtype == object:
            return None
    else:
        return None
    if array.ndim < 2 or array.ndim > 5 or not np.issubdtype(array.dtype, np.number):
        return None
    return array


def _axis_candidates(shape: tuple[int, ...]) -> list[tuple[int, int, int | None, int]]:
    """Return scored (coord, joint, person, score) layout candidates."""

    ndim = len(shape)
    candidates: list[tuple[int, int, int | None, int]] = []
    coord_axes = [axis for axis, size in enumerate(shape) if 2 <= size <= 4]
    known_joint_axes = [axis for axis, size in enumerate(shape) if size in _KNOWN_JOINT_COUNTS]
    joint_axes = known_joint_axes or [axis for axis, size in enumerate(shape) if 5 <= size <= 100]
    for coord in coord_axes:
        for joint in joint_axes:
            if coord == joint:
                continue
            if ndim == 2:
                score = (6 if coord == 1 else 3) + (2 if shape[joint] in _KNOWN_JOINT_COUNTS else 0)
                candidates.append((coord, joint, None, score))
                continue
            remaining = [axis for axis in range(ndim) if axis not in {coord, joint}]
            if ndim == 3:
                score = (6 if (coord, joint) == (2, 1) else 0)
                score += 5 if (coord, joint) == (0, 2) else 0
                score += 3 if coord == 2 else 0
                score += 2 if shape[joint] in _KNOWN_JOINT_COUNTS else 0
                candidates.append((coord, joint, None, score))
                continue
            if ndim == 4:
                for person in remaining:
                    if not 1 <= shape[person] <= 8:
                        continue
                    score = 0
                    if (coord, joint, person) == (0, 2, 3):  # C,T,V,M
                        score += 12
                    elif (coord, joint, person) == (2, 1, 3):  # T,V,C,M
                        score += 11
                    elif (coord, joint, person) == (3, 2, 0):  # M,T,V,C
                        score += 10
                    elif (coord, joint, person) == (3, 1, 2):  # T,V,M,C
                        score += 9
                    score += 2 if shape[joint] in _KNOWN_JOINT_COUNTS else 0
                    candidates.append((coord, joint, person, score))
    return candidates


def _align_confidence(
    confidence_value: Any,
    *,
    target_shape: tuple[int, ...],
    original_pose_shape: tuple[int, ...],
    coord_axis: int,
    output_axes: tuple[int, ...],
) -> np.ndarray:
    confidence = np.asarray(confidence_value, dtype=np.float32)
    while confidence.ndim > len(target_shape) and 1 in confidence.shape:
        confidence = np.squeeze(confidence, axis=next(i for i, size in enumerate(confidence.shape) if size == 1))
    if confidence.ndim != len(target_shape):
        raise PoseSchemaError(
            f"Confidence array {confidence.shape} cannot align to expected {target_shape}."
        )
    remaining_original_axes = [axis for axis in range(len(original_pose_shape)) if axis != coord_axis]
    expected_original_shape = tuple(original_pose_shape[axis] for axis in remaining_original_axes)
    if confidence.shape == expected_original_shape:
        permutation = tuple(remaining_original_axes.index(axis) for axis in output_axes)
        aligned = np.transpose(confidence, permutation)
        if aligned.shape == target_shape:
            return aligned
    if confidence.shape == target_shape:
        return confidence
    # Permit a uniquely determined transpose, but reject equal-sized-axis
    # ambiguity rather than silently assigning confidence to the wrong joint.
    import itertools

    matches = [
        order
        for order in itertools.permutations(range(confidence.ndim))
        if tuple(confidence.shape[index] for index in order) == target_shape
    ]
    if len(matches) == 1:
        return np.transpose(confidence, matches[0])
    raise PoseSchemaError(
        f"Confidence array {confidence.shape} cannot be unambiguously aligned to {target_shape}."
    )


def _canonicalize_pose_with_info(
    value: Any, confidence_value: Any = None, layout_hint: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int, str]:
    """Convert a common pose layout to ``[T,V,C]`` and infer a validity mask.

    Supported single-sample layouts include ``T,V,C``, ``C,T,V``, ``V,C`` and
    their common one-person/multi-person four-dimensional variants.  A primary
    person is selected globally by the number of finite, non-zero joint entries.
    Ambiguous or batched arrays fail explicitly.
    """

    array = _numeric_array(value)
    if array is None:
        raise PoseSchemaError("Pose value is not a dense numeric array with 2-4 sample dimensions.")
    if array.ndim == 5:
        raise PoseSchemaError(
            f"Array {array.shape} appears batched; split its sample axis before canonicalization."
        )
    candidates = _axis_candidates(tuple(int(size) for size in array.shape))
    if layout_hint is not None:
        token = layout_hint.upper().replace("_", "").replace("-", "")
        if token != "MTVC" or array.ndim != 4:
            raise PoseSchemaError(
                f"Unsupported/incompatible explicit layout hint {layout_hint!r} for {array.shape}."
            )
        # MMAAction annotation keypoint arrays explicitly use [M,T,V,C].
        if not (2 <= array.shape[3] <= 4 and 5 <= array.shape[2] <= 100):
            raise PoseSchemaError(f"Explicit MTVC layout is incompatible with shape {array.shape}.")
        candidates = [(3, 2, 0, 10_000)]
    if not candidates:
        raise PoseSchemaError(
            f"Cannot identify coordinate/joint axes in pose array {array.shape}; "
            "supported coordinate sizes are 2-4 and joint axes are 5-100."
        )
    if layout_hint is None and array.ndim == 4:
        common = {(coord, joint, person) for coord, joint, person, score in candidates if score >= 9}
        if len(common) > 1:
            raise PoseSchemaError(
                f"Ambiguous multi-person pose layout for shape {array.shape}: {sorted(common)}. "
                "Provide an explicit layout or paired confidence whose axes establish [M,T,V,C]."
            )
    candidates.sort(key=lambda item: item[3], reverse=True)
    if len(candidates) > 1 and candidates[0][3] == candidates[1][3]:
        best_layouts = [(c, j, p) for c, j, p, score in candidates if score == candidates[0][3]]
        raise PoseSchemaError(
            f"Ambiguous pose array axes for shape {array.shape}; equally likely layouts {best_layouts}."
        )
    coord_axis, joint_axis, person_axis, _ = candidates[0]
    confidence: np.ndarray | None = None
    person_count = 1
    policy = "single_person"
    if array.ndim == 2:
        canonical = np.moveaxis(array, (joint_axis, coord_axis), (0, 1))[None, ...]
        time_axis = None
    elif array.ndim == 3:
        time_axis = next(axis for axis in range(3) if axis not in {coord_axis, joint_axis})
        canonical = np.transpose(array, (time_axis, joint_axis, coord_axis))
    else:
        assert person_axis is not None
        time_axis = next(
            axis for axis in range(4) if axis not in {coord_axis, joint_axis, person_axis}
        )
        people = np.transpose(array, (person_axis, time_axis, joint_axis, coord_axis))
        person_count = int(people.shape[0])
        coord_count = min(3, people.shape[-1])
        finite = np.isfinite(people[..., :coord_count]).all(axis=-1)
        nonzero = np.any(np.abs(people[..., :coord_count]) > 1e-12, axis=-1)
        observed = finite & nonzero
        aligned_confidence: np.ndarray | None = None
        if confidence_value is not None:
            aligned_confidence = _align_confidence(
                confidence_value,
                target_shape=tuple(int(size) for size in people.shape[:3]),
                original_pose_shape=tuple(int(size) for size in array.shape),
                coord_axis=coord_axis,
                output_axes=(person_axis, time_axis, joint_axis),
            )
            aggregate = np.where(
                observed & np.isfinite(aligned_confidence) & (aligned_confidence > 0),
                aligned_confidence,
                0.0,
            ).sum(axis=(1, 2))
            policy = "highest_aggregate_keypoint_score"
        else:
            aggregate = np.count_nonzero(observed, axis=(1, 2))
            policy = "most_valid_joint_observations" if person_count > 1 else "single_person"
        selected_person = int(np.argmax(aggregate))
        canonical = people[selected_person]
        if aligned_confidence is not None:
            confidence = aligned_confidence[selected_person]
    canonical = np.asarray(canonical, dtype=np.float32)
    coord_count = min(3, canonical.shape[-1])
    finite = np.isfinite(canonical).all(axis=-1)
    # Pose exporters conventionally use an all-zero joint for missing detections.
    nonzero = np.any(np.abs(canonical[..., :coord_count]) > 1e-12, axis=-1)
    mask = finite & nonzero
    if array.ndim <= 3 and confidence_value is not None:
        if array.ndim == 2:
            confidence = np.asarray(confidence_value, dtype=np.float32)
            confidence = np.squeeze(confidence)
            if confidence.shape == (canonical.shape[1],):
                confidence = confidence[None, :]
            if confidence.shape != canonical.shape[:2]:
                raise PoseSchemaError(
                    f"Confidence {confidence.shape} does not match canonical pose {canonical.shape[:2]}."
                )
        else:
            assert time_axis is not None
            confidence = _align_confidence(
                confidence_value,
                target_shape=tuple(int(size) for size in canonical.shape[:2]),
                original_pose_shape=tuple(int(size) for size in array.shape),
                coord_axis=coord_axis,
                output_axes=(time_axis, joint_axis),
            )
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float32)
        mask &= np.isfinite(confidence) & (confidence > 0)
        confidence = np.where(mask, confidence, 0.0).astype(np.float32, copy=False)
    canonical = np.where(mask[..., None], canonical, 0.0).astype(np.float32, copy=False)
    return canonical, mask, confidence, person_count, policy


def canonicalize_pose_array(
    value: Any, *, layout_hint: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a common pose layout to ``[T,V,C]`` and infer a validity mask.

    This backwards-compatible public helper omits optional confidence metadata;
    record loading preserves it through :class:`PoseSample`.
    """

    keypoints, mask, _, _, _ = _canonicalize_pose_with_info(value, layout_hint=layout_hint)
    return keypoints, mask


def _matching_items(record: Mapping[Any, Any], names: set[str]) -> list[tuple[Any, Any]]:
    return [(key, value) for key, value in record.items() if str(key).strip().lower() in names]


def _single_named_value(
    record: Mapping[Any, Any], names: set[str], description: str, *, required: bool = False
) -> Any:
    matches = _matching_items(record, names)
    if len(matches) > 1:
        keys = [str(key) for key, _ in matches]
        raise PoseSchemaError(f"Multiple possible {description} fields {keys}; schema is ambiguous.")
    if not matches:
        if required:
            raise PoseSchemaError(f"No explicit {description} field was found.")
        return None
    return matches[0][1]


def _scalar(value: Any) -> bool:
    return np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim == 0)


def _as_sequence(value: Any) -> list[Any] | None:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            return None
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _is_label_vector(value: Any) -> bool:
    sequence = _as_sequence(value)
    if sequence is None or not sequence:
        return False
    try:
        for item in sequence:
            label_to_index(item)
    except PoseSchemaError:
        return False
    return True


def _is_name_vector(value: Any) -> bool:
    sequence = _as_sequence(value)
    return bool(sequence) and all(isinstance(item, (str, bytes, Path)) for item in sequence)


def _pose_collection_length(value: Any) -> int | None:
    if isinstance(value, np.ndarray) and value.dtype != object and value.ndim in {4, 5}:
        return int(value.shape[0])
    if isinstance(value, (list, tuple)) and value and all(
        (array := _numeric_array(item)) is not None and array.ndim <= 4 for item in value
    ):
        return len(value)
    if (
        isinstance(value, np.ndarray)
        and value.dtype == object
        and value.ndim == 1
        and all(_numeric_array(item) is not None for item in value.tolist())
    ):
        return int(value.shape[0])
    return None


def _pose_collection_item(value: Any, index: int) -> Any:
    return value[index]


def _record_pose_value(record: Mapping[Any, Any]) -> Any:
    named = [(key, value) for key, value in _matching_items(record, _POSE_KEYS) if _numeric_array(value) is not None]
    if len(named) > 1:
        raise PoseSchemaError(f"Multiple numeric pose fields {[str(key) for key, _ in named]}.")
    if len(named) == 1:
        return named[0][1]
    excluded = _VALIDITY_KEYS | _LABEL_KEYS | _ID_KEYS | _SUBJECT_KEYS | {
        "img_shape",
        "original_shape",
        "total_frames",
    }
    candidates = [
        (key, value)
        for key, value in record.items()
        if str(key).strip().lower() not in excluded and _numeric_array(value) is not None
    ]
    if len(candidates) != 1:
        raise PoseSchemaError(
            "Expected exactly one dense numeric pose array in record; found fields "
            f"{[str(key) for key, _ in candidates]}."
        )
    return candidates[0][1]


def _coerce_identifier(value: Any, description: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PoseSchemaError(f"{description} bytes are not valid UTF-8.") from exc
    if not isinstance(value, (str, Path, int, np.integer)):
        raise PoseSchemaError(f"{description} must be scalar text/integer, got {type(value).__name__}.")
    result = str(value).strip()
    if not result:
        raise PoseSchemaError(f"{description} is empty.")
    return result


def _make_sample(
    pose_value: Any,
    label_value: Any,
    sample_id_value: Any,
    subject_value: Any,
    mask_value: Any,
    confidence_value: Any = None,
    *,
    source: str,
    layout_hint: str | None = None,
) -> PoseSample:
    keypoints, inferred_mask, confidence, person_count, policy = _canonicalize_pose_with_info(
        pose_value, confidence_value, layout_hint
    )
    sample_id = _coerce_identifier(sample_id_value, "sample_id")
    parsed = parse_sample_name(sample_id) if sample_id is not None else None
    if sample_id is None:
        raise PoseSchemaError(
            "Record has no stable sample identifier; refusing to synthesize one from row position."
        )
    if label_value is None:
        if parsed is None:
            raise PoseSchemaError(
                f"Sample {sample_id!r} has no explicit label and its name does not match "
                "<numeric-subject>_<d|p|s>_<take>."
            )
        label = parsed[1]
    else:
        label = label_to_index(label_value)
        if parsed is not None and parsed[1] != label:
            raise PoseSchemaError(
                f"Explicit label {label} conflicts with action encoded in sample name {sample_id!r}."
            )
    subject_id = _coerce_identifier(subject_value, "subject_id")
    if subject_id is None:
        if parsed is None:
            raise PoseSchemaError(
                f"Cannot recover subject for {sample_id!r}; expected an explicit subject field or "
                "the anchored <numeric-subject>_<d|p|s>_<take> naming convention."
            )
        subject_id = parsed[0]
    elif parsed is not None and subject_id != parsed[0]:
        raise PoseSchemaError(
            f"Explicit subject {subject_id!r} conflicts with sample name {sample_id!r}."
        )
    if mask_value is not None:
        explicit_mask = np.asarray(mask_value)
        # A singleton person axis is common in MMAAction-style annotations.
        if explicit_mask.ndim == 3 and explicit_mask.shape[0] == 1:
            explicit_mask = explicit_mask[0]
        explicit_mask = np.isfinite(explicit_mask) & (explicit_mask > 0)
        if explicit_mask.shape != inferred_mask.shape:
            raise PoseSchemaError(
                f"Explicit mask {explicit_mask.shape} does not match canonical [T,V] {inferred_mask.shape}."
            )
        inferred_mask &= explicit_mask
    return PoseSample(
        keypoints=keypoints,
        valid_mask=inferred_mask,
        label=label,
        sample_id=sample_id,
        subject_id=subject_id,
        source=source,
        confidence=confidence,
        original_person_count=person_count,
        person_selection_policy=policy,
    )


def _record_from_mapping(record: Mapping[Any, Any], *, source: str, outer_id: Any = None) -> PoseSample:
    pose = _record_pose_value(record)
    label = _single_named_value(record, _LABEL_KEYS, "label")
    sample_id = _single_named_value(record, _ID_KEYS, "sample identifier")
    subject = _single_named_value(record, _SUBJECT_KEYS, "subject identifier")
    mask = _single_named_value(record, _MASK_KEYS, "validity mask")
    confidence = _single_named_value(record, _CONFIDENCE_KEYS, "confidence")
    if sample_id is None:
        sample_id = outer_id
    layout_hint = None
    pose_array = _numeric_array(pose)
    score_array = np.asarray(confidence) if confidence is not None else None
    if (
        pose_array is not None
        and pose_array.ndim == 4
        and score_array is not None
        and score_array.shape == pose_array.shape[:3]
    ):
        # This structural relationship is observed directly in MMAAction-style
        # records: pose [M,T,V,C] and confidence [M,T,V].
        layout_hint = "MTVC"
    return _make_sample(
        pose,
        label,
        sample_id,
        subject,
        mask,
        confidence,
        source=source,
        layout_hint=layout_hint,
    )


def _columnar_mapping(record: Mapping[Any, Any], *, source: str) -> list[PoseSample] | None:
    pose_matches = [
        (key, value)
        for key, value in _matching_items(record, _POSE_KEYS)
        if _pose_collection_length(value) is not None
    ]
    if not pose_matches:
        return None
    if len(pose_matches) > 1:
        raise PoseSchemaError(f"Multiple columnar pose fields {[str(key) for key, _ in pose_matches]}.")
    pose_values = pose_matches[0][1]
    count = _pose_collection_length(pose_values)
    assert count is not None

    def column(names: set[str], description: str) -> list[Any] | None:
        value = _single_named_value(record, names, description)
        if value is None:
            return None
        result = _as_sequence(value)
        if result is None or len(result) != count:
            raise PoseSchemaError(
                f"Column {description} must have {count} entries to match pose data."
            )
        return result

    labels = column(_LABEL_KEYS, "labels")
    names = column(_ID_KEYS, "sample identifiers")
    subjects = column(_SUBJECT_KEYS, "subject identifiers")
    masks = column(_MASK_KEYS, "validity masks")
    confidences = column(_CONFIDENCE_KEYS, "confidence")
    if names is None:
        raise PoseSchemaError("Columnar pose data has no sample identifier column.")
    return [
        _make_sample(
            _pose_collection_item(pose_values, index),
            None if labels is None else labels[index],
            names[index],
            None if subjects is None else subjects[index],
            None if masks is None else masks[index],
            None if confidences is None else confidences[index],
            source=source,
        )
        for index in range(count)
    ]


def _columnar_sequence(value: Sequence[Any], *, source: str) -> list[PoseSample] | None:
    pose_candidates = [(index, item) for index, item in enumerate(value) if _pose_collection_length(item)]
    if not pose_candidates:
        return None
    if len(pose_candidates) > 1:
        raise PoseSchemaError(
            f"Tuple/list contains multiple possible pose collections at positions "
            f"{[index for index, _ in pose_candidates]}."
        )
    pose_index, poses = pose_candidates[0]
    count = _pose_collection_length(poses)
    assert count is not None
    labels: list[Any] | None = None
    names: list[Any] | None = None
    for index, item in enumerate(value):
        if index == pose_index:
            continue
        sequence = _as_sequence(item)
        if sequence is None or len(sequence) != count:
            continue
        if _is_label_vector(item):
            if labels is not None:
                raise PoseSchemaError("Tuple/list contains multiple possible label vectors.")
            labels = sequence
        elif _is_name_vector(item):
            if names is not None:
                raise PoseSchemaError("Tuple/list contains multiple possible sample-name vectors.")
            names = sequence
    if names is None:
        raise PoseSchemaError("Columnar tuple/list has pose data but no sample-name vector.")
    return [
        _make_sample(
            _pose_collection_item(poses, index),
            None if labels is None else labels[index],
            names[index],
            None,
            None,
            source=source,
        )
        for index in range(count)
    ]


def samples_from_object(value: Any, *, source: str = "") -> list[PoseSample]:
    """Adapt supported pickle structures into canonical samples.

    Supported structures are a single mapping record, a list of mapping/tuple
    records, a mapping keyed by sample name, or a columnar mapping/tuple containing
    one pose collection plus labels/names.  Unsupported and ambiguous structures
    raise :class:`PoseSchemaError` with their exact failure reason.
    """

    if isinstance(value, Mapping):
        columnar = _columnar_mapping(value, source=source)
        if columnar is not None:
            return columnar
        # Some annotation formats wrap records and metadata in a top-level
        # mapping.  Recognise this structurally only when there is exactly one
        # non-empty sequence whose members are all mappings.
        record_sequences = [
            (key, item)
            for key, item in value.items()
            if isinstance(item, (list, tuple)) and item and all(isinstance(row, Mapping) for row in item)
        ]
        if len(record_sequences) == 1:
            wrapper_key, records = record_sequences[0]
            return [
                _record_from_mapping(
                    row,
                    source=f"{source}.{wrapper_key}[{index}]" if source else f"{wrapper_key}[{index}]",
                )
                for index, row in enumerate(records)
            ]
        if len(record_sequences) > 1:
            raise PoseSchemaError(
                "Top-level mapping contains multiple possible record sequences: "
                f"{[str(key) for key, _ in record_sequences]}."
            )
        try:
            return [_record_from_mapping(value, source=source)]
        except PoseSchemaError as direct_error:
            keyed: list[PoseSample] = []
            for key, item in value.items():
                if parse_sample_name(key) is None:
                    keyed = []
                    break
                if isinstance(item, Mapping):
                    keyed.append(_record_from_mapping(item, source=source, outer_id=key))
                elif _numeric_array(item) is not None:
                    keyed.append(_make_sample(item, None, key, None, None, source=source))
                else:
                    keyed = []
                    break
            if keyed:
                return keyed
            raise PoseSchemaError(
                f"Mapping is neither a recognised record, columnar table, nor sample-name map: {direct_error}"
            ) from direct_error

    if isinstance(value, np.ndarray) and value.dtype == object and value.ndim == 1:
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        columnar = _columnar_sequence(value, source=source)
        if columnar is not None:
            return columnar
        samples: list[PoseSample] = []
        for index, item in enumerate(value):
            item_source = f"{source}[{index}]" if source else f"record[{index}]"
            if isinstance(item, Mapping):
                samples.append(_record_from_mapping(item, source=item_source))
            elif isinstance(item, (list, tuple)) and 2 <= len(item) <= 4:
                pose_items = [candidate for candidate in item if _numeric_array(candidate) is not None]
                scalar_labels = [candidate for candidate in item if _scalar(candidate)]
                label_items: list[Any] = []
                name_items: list[Any] = []
                for candidate in scalar_labels:
                    try:
                        label_to_index(candidate)
                        label_items.append(candidate)
                    except PoseSchemaError:
                        if isinstance(candidate, (str, bytes, Path)):
                            name_items.append(candidate)
                if len(pose_items) != 1 or len(name_items) != 1 or len(label_items) > 1:
                    raise PoseSchemaError(
                        f"{item_source} tuple must contain one pose array, one sample name, "
                        "and at most one label."
                    )
                samples.append(
                    _make_sample(
                        pose_items[0],
                        label_items[0] if label_items else None,
                        name_items[0],
                        None,
                        None,
                        source=item_source,
                    )
                )
            else:
                raise PoseSchemaError(
                    f"{item_source} has unsupported type {type(item).__name__}; expected a record mapping."
                )
        if samples:
            return samples
    raise PoseSchemaError(
        f"Unsupported pickle root type {type(value).__name__}; expected mapping/list/tuple records."
    )


def object_is_synthetic(value: Any) -> bool:
    """Return true only when an input object explicitly marks itself synthetic."""

    if not isinstance(value, Mapping):
        return False
    metadata = value.get("fixture_metadata")
    if isinstance(metadata, Mapping) and metadata.get("synthetic") is True:
        return True
    marker = value.get("synthetic")
    return marker is True


def load_pickle(path: str | Path) -> Any:
    """Load a trusted local dataset pickle.

    Pickle can execute code while loading; callers must only pass the downloaded,
    gated dataset or a fixture they trust.
    """

    path = Path(path)
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:
        raise PoseSchemaError(f"Could not load trusted pickle {path}: {exc}") from exc


def load_pose_pickle(path: str | Path) -> list[PoseSample]:
    """Load and adapt one trusted pose pickle."""

    path = Path(path)
    samples = samples_from_object(load_pickle(path), source=path.name)
    if not samples:
        raise PoseSchemaError(f"No pose samples were found in {path}.")
    return samples


def load_pose_pickles(paths: Iterable[str | Path]) -> list[PoseSample]:
    """Load multiple pickles and reject duplicate sample identifiers."""

    samples: list[PoseSample] = []
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item).lower()):
        samples.extend(load_pose_pickle(path))
    seen: set[str] = set()
    duplicates: set[str] = set()
    for sample in samples:
        if sample.sample_id in seen:
            duplicates.add(sample.sample_id)
        seen.add(sample.sample_id)
    if duplicates:
        preview = sorted(duplicates)[:10]
        raise PoseSchemaError(f"Duplicate sample identifiers across pickle files: {preview}.")
    return samples


def iter_numeric_arrays(value: Any, *, path: str = "root", max_depth: int = 5) -> Iterator[tuple[str, np.ndarray]]:
    """Yield dense numeric arrays for schema inspection without assuming keys."""

    if max_depth < 0:
        return
    array = _numeric_array(value)
    if array is not None:
        yield path, array
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from iter_numeric_arrays(item, path=f"{path}.{key}", max_depth=max_depth - 1)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value[:20]):
            yield from iter_numeric_arrays(item, path=f"{path}[{index}]", max_depth=max_depth - 1)


def describe_schema(value: Any, *, max_items: int = 5) -> list[str]:
    """Return a concise, non-speculative structural description for a report."""

    lines = [f"Root type: `{type(value).__name__}`"]
    if isinstance(value, Mapping):
        keys = list(value.keys())
        lines.append(f"Number of top-level entries/fields: {len(keys)}")
        lines.append(f"Top-level keys (up to {max_items}): `{[str(key) for key in keys[:max_items]]}`")
        for key in keys[:max_items]:
            item = value[key]
            array = _numeric_array(item)
            detail = f"array shape={tuple(array.shape)}, dtype={array.dtype}" if array is not None else type(item).__name__
            lines.append(f"- `{key}`: {detail}")
    elif isinstance(value, (list, tuple, np.ndarray)):
        try:
            length = len(value)
        except TypeError:
            length = 0
        lines.append(f"Number of top-level entries: {length}")
        for index in range(min(length, max_items)):
            item = value[index]
            array = _numeric_array(item)
            detail = f"array shape={tuple(array.shape)}, dtype={array.dtype}" if array is not None else type(item).__name__
            if isinstance(item, Mapping):
                detail += f", keys={[str(key) for key in list(item.keys())[:max_items]]}"
            lines.append(f"- Entry {index}: {detail}")
    arrays = list(iter_numeric_arrays(value))
    if arrays:
        lines.append("Observed numeric arrays (first 10):")
        for array_path, array in arrays[:10]:
            lines.append(f"- `{array_path}`: shape={tuple(array.shape)}, dtype={array.dtype}")
    else:
        lines.append("Observed numeric arrays: none in the inspected prefix")
    return lines
