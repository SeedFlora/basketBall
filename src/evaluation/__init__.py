"""Evaluation, robustness, and error-analysis tools."""

from .error_analysis import (
    build_error_analysis_table,
    build_per_joint_reconstruction_table,
    build_reconstruction_error_table,
    confusion_pair_summary,
)
from .metrics import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_PCK_THRESHOLDS,
    RECONSTRUCTION_SUMMARY_METRICS,
    ReconstructionErrorArrays,
    aggregate_metric_runs,
    bbox_normalization_scale,
    classification_metrics,
    compute_classification_metrics,
    compute_metrics,
    compute_reconstruction_metrics,
    reconstruction_error_arrays,
    reconstruction_metrics,
)
from .robustness import (
    SUPPORTED_STRATEGIES,
    apply_missing_joints,
    evaluate_robustness,
    generate_missing_joint_mask,
    temporal_interpolate_missing,
)

__all__ = [
    "DEFAULT_CLASS_NAMES",
    "DEFAULT_PCK_THRESHOLDS",
    "RECONSTRUCTION_SUMMARY_METRICS",
    "SUPPORTED_STRATEGIES",
    "ReconstructionErrorArrays",
    "aggregate_metric_runs",
    "apply_missing_joints",
    "bbox_normalization_scale",
    "build_error_analysis_table",
    "build_per_joint_reconstruction_table",
    "build_reconstruction_error_table",
    "classification_metrics",
    "compute_classification_metrics",
    "compute_metrics",
    "compute_reconstruction_metrics",
    "confusion_pair_summary",
    "evaluate_robustness",
    "generate_missing_joint_mask",
    "reconstruction_error_arrays",
    "reconstruction_metrics",
    "temporal_interpolate_missing",
]
