"""Publication-oriented visualizations."""

from .confusion_matrix import plot_confusion_matrix, save_confusion_matrix
from .embedding_plot import (
    pca_projection,
    plot_embedding,
    project_embeddings,
    save_embedding_projections,
    tsne_projection,
)
from .reconstruction import (
    missingness_by_domain_joint,
    plot_coordinate_distributions,
    plot_cross_domain_matrix,
    plot_domain_summary,
    plot_missingness_heatmap,
    plot_nmje_by_model,
    plot_pck_curves,
    plot_per_joint_error,
    plot_pose_instance_counts,
    plot_reconstruction_data_efficiency,
    plot_reconstruction_example,
)
from .results import (
    plot_ablation_comparison,
    plot_class_distribution,
    plot_joint_angle_distributions,
    plot_joint_angle_example,
    plot_per_class_f1,
    plot_robustness_curve,
    plot_sample_size_curve,
    plot_subject_distribution,
)
from .skeleton_plot import plot_skeleton, plot_skeleton_sequence

__all__ = [
    "missingness_by_domain_joint",
    "pca_projection",
    "plot_ablation_comparison",
    "plot_class_distribution",
    "plot_confusion_matrix",
    "plot_coordinate_distributions",
    "plot_cross_domain_matrix",
    "plot_domain_summary",
    "plot_embedding",
    "plot_joint_angle_distributions",
    "plot_joint_angle_example",
    "plot_missingness_heatmap",
    "plot_nmje_by_model",
    "plot_pck_curves",
    "plot_per_class_f1",
    "plot_per_joint_error",
    "plot_pose_instance_counts",
    "plot_reconstruction_data_efficiency",
    "plot_reconstruction_example",
    "plot_robustness_curve",
    "plot_sample_size_curve",
    "plot_skeleton",
    "plot_skeleton_sequence",
    "plot_subject_distribution",
    "project_embeddings",
    "save_confusion_matrix",
    "save_embedding_projections",
    "tsne_projection",
]
