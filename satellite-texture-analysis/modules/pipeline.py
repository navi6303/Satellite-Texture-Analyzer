"""
End-to-end texture analysis after patches have been extracted.
Keeps app.py and the batch runner on one code path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from modules.analysis_checks import run_analysis_checks
from modules.glcm_analysis import (
    analyze_patch_glcm,
    build_classic_table,
    build_eigenvalue_table,
    build_feature_matrix,
    class_mean_distance_matrix,
    class_mean_std_table,
    compute_eigenvalues,
    fisher_ratios,
    generate_separability_interpretation,
    pairwise_distance_matrix,
    zscore_features,
)
from modules.texture_extraction import TexturePatch
from modules.validation import compare_feature_sets
from modules.matrix_audit import matrix_audit, directional_covariance


def analyze_all_patches(
    patches: List[TexturePatch],
    distances: Sequence[int],
    angles_deg: Sequence[float],
    include_transition: bool = True,
    symmetric: bool = True,
    normed: bool = True,
) -> List[TexturePatch]:
    d0 = int(distances[0]) if distances else 1
    for p in patches:
        analyze_patch_glcm(
            p,
            distance=d0,
            angle_rad=float(np.radians(float(angles_deg[0]) if angles_deg else 0.0)),
            symmetric=symmetric,
            normed=normed,
            angles_deg=list(angles_deg),
            distances=list(distances),
            compute_transition=include_transition,
        )
    return patches


def class_glcm_eigenvalue_vectors(
    patches: List[TexturePatch],
) -> Tuple[np.ndarray, List[str]]:
    """Return one signed spectrum per class from equally weighted patch GLCMs."""
    matrices = {}
    for patch in patches:
        if patch.glcm is not None:
            matrices.setdefault(patch.label, []).append(patch.glcm)
    labels = list(matrices)
    # Each input is already a symmetric, normalized patch GLCM. Its mean
    # preserves both properties and gives each selected patch equal weight.
    vectors = np.asarray([
        compute_eigenvalues(np.mean(matrices[label], axis=0)) for label in labels
    ], dtype=np.float64).reshape(-1, 5)
    return vectors, labels


def run_separability_and_cv(
    patches: List[TexturePatch],
    distances: Sequence[int],
    angles_deg: Sequence[float],
    quantization_mode: str,
    include_transition: bool = True,
) -> Dict[str, Any]:
    df_eigs = build_eigenvalue_table(patches)
    df_mean = class_mean_std_table(patches)
    df_classic = build_classic_table(patches, include_transition=include_transition)

    X, names, labels, _blocks = build_feature_matrix(
        patches, feature_set="combined", include_transition=include_transition
    )
    Z, mu, sd = zscore_features(X)
    df_dist_raw = class_mean_distance_matrix(X, labels)
    df_combined_std = class_mean_distance_matrix(Z, labels)
    # Keep eigenvalues_summary / pairwise_distances.csv semantics: raw class-centroid
    # distances on the combined (or eigenvalue) space.
    Xe, _, lab_e, _ = build_feature_matrix(patches, feature_set="eigenvalues")
    df_dist_eigs_raw = class_mean_distance_matrix(Xe, lab_e)

    df_dist_std = class_mean_distance_matrix(zscore_features(Xe)[0], lab_e)
    class_vectors, class_labels = class_glcm_eigenvalue_vectors(patches)
    df_class_glcm_distances = pairwise_distance_matrix(class_vectors, class_labels)
    # Report both aliases, use trace just once in models and distance calculations.
    rank_matrix, rank_names = (
        np.column_stack([X, X[:, names.index("glcm_trace")]]),
        names + ["diagonal_share"],
    )
    if include_transition:
        Xt, tn, _, _ = build_feature_matrix(patches, "transition", True)
        rank_matrix = np.column_stack([rank_matrix, Xt])
        rank_names += tn
    df_fisher = fisher_ratios(rank_matrix, labels, rank_names)
    cv_results = compare_feature_sets(patches, include_transition=include_transition)
    d0 = int(distances[0]) if distances else 1
    check_bundle = run_analysis_checks(patches, df_fisher=df_fisher, distance=d0)
    interp = generate_separability_interpretation(
        patches,
        glcm_distance=d0,
        glcm_angle_deg=float(angles_deg[0]) if angles_deg else 0.0,
        df_fisher=df_fisher,
        df_dist_raw=df_dist_eigs_raw,
        df_dist_std=df_dist_std,
        cv_results=cv_results,
        quantization_mode=quantization_mode,
        angles_deg=list(angles_deg),
    )
    return {
        "matrix_audit": matrix_audit(patches),
        "directional_pca": directional_covariance(patches),
        "df_eigenvalues": df_eigs,
        "df_distances": df_dist_eigs_raw,
        "df_dist_raw_combined": df_dist_raw,
        "df_dist_std_combined": df_combined_std,
        "df_dist_std": df_dist_std,
        "df_class_glcm_distances": df_class_glcm_distances,
        "df_eigen_mean": df_mean,
        "df_classic": df_classic,
        "df_fisher": df_fisher,
        "feature_mean": mu,
        "feature_std": sd,
        "feature_names": names,
        "cv_results": cv_results,
        "check_bundle": check_bundle,
        "interpretation": interp,
    }
