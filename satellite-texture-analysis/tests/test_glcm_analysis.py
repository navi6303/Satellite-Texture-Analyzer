"""
Tests for GLCM Computation, Eigenvalue Extraction, and Feature Analysis.
Validates:
  - 5x5 GLCM generation, symmetry, and normalization
  - Masked-pixel handling during pair counting
  - Eigenvalue calculation against known analytical symmetric matrices
  - Descending eigenvalue sorting and negative eigenvalue preservation
  - Feature distance tables and scientific interpretation generation
  - Complete ZIP export bundling
"""

import numpy as np
import pandas as pd
import pytest

from modules.export_utils import build_complete_zip_package, generate_markdown_report
from modules.glcm_analysis import (
    analyze_patch_glcm,
    build_eigenvalue_table,
    calculate_feature_distances,
    compute_eigenvalues,
    compute_glcm,
    generate_separability_interpretation
)
from modules.image_loader import LoadedRaster, RasterMetadata
from modules.texture_extraction import extract_patch, get_default_quadrant_crops


def _make_simple_raster(w=64, h=64, n_bands=4, seed=42):
    """Create a minimal in-memory LoadedRaster for testing without the removed synthetic helper."""
    rng = np.random.default_rng(seed)
    data = (rng.random((n_bands, h, w)) * 3000 + 100).astype(np.float32)
    meta = RasterMetadata(
        width=w, height=h, count=n_bands,
        dtypes=tuple("float32" for _ in range(n_bands)),
        crs="CRS not defined"
    )
    raster = LoadedRaster(metadata=meta, band_names=[f"B{i+1}" for i in range(n_bands)], in_memory_data=data)
    return raster, data


def test_glcm_shape_symmetry_normalization():
    """Verify GLCM produces symmetric 5x5 normalized matrix summing to 1.0."""
    patch = np.array([
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0],
        [2, 3, 4, 0, 1],
        [3, 4, 0, 1, 2],
        [4, 0, 1, 2, 3]
    ], dtype=np.uint8)

    glcm, pairs = compute_glcm(patch, distance=1, angle_rad=0.0, symmetric=True, normed=True)

    assert glcm.shape == (5, 5)
    assert pairs > 0
    # Must sum to 1.0
    assert np.isclose(np.sum(glcm), 1.0, atol=1e-7)
    # Must be perfectly symmetric
    assert np.allclose(glcm, glcm.T, atol=1e-7)


def test_glcm_masked_pixel_handling():
    """
    Verify that masked / invalid pixels are NOT counted as gray level zero.
    """
    patch = np.zeros((4, 4), dtype=np.uint8)
    patch[1, 1] = 3
    patch[1, 2] = 3
    
    # Mask: only (1, 1) and (1, 2) are valid
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    mask[1, 2] = True
    
    glcm, valid_pairs = compute_glcm(
        patch,
        valid_mask=mask,
        distance=1,
        angle_rad=0.0,
        symmetric=True,
        normed=True
    )
    
    # Only one directed pair (1,1)->(1,2) with graylevels (3, 3) was valid
    assert valid_pairs == 1
    # Because symmetric=True, glcm[3, 3] should be 1.0, and ALL other entries (including 0,0) must be 0.0!
    assert np.isclose(glcm[3, 3], 1.0, atol=1e-7)
    assert np.isclose(glcm[0, 0], 0.0, atol=1e-7)


def test_eigenvalues_known_symmetric_matrix():
    """
    Verify eigenvalues calculation against a known symmetric matrix with analytical eigenvalues.
    For a diagonal matrix D, the eigenvalues are simply the diagonal entries!
    """
    known_eigs = np.array([0.45, 0.25, 0.15, 0.10, 0.05])  # Sum = 1.0
    diag_matrix = np.diag(known_eigs)

    computed = compute_eigenvalues(diag_matrix, check_symmetry=True)

    assert len(computed) == 5
    assert np.allclose(computed, known_eigs, atol=1e-7)
    # Check descending order
    assert np.all(np.diff(computed) <= 0)


def test_negative_eigenvalues_preserved():
    """
    Verify that negative eigenvalues are mathematically preserved if they occur.
    """
    # Create a symmetric matrix with known negative eigenvalue
    # e.g., block matrix [[0, 0.5], [0.5, 0]] has eigenvalues +0.5 and -0.5
    mat = np.zeros((5, 5), dtype=np.float64)
    mat[0, 1] = 0.5
    mat[1, 0] = 0.5
    mat[2, 2] = 0.4
    mat[3, 3] = 0.3
    mat[4, 4] = 0.2

    computed = compute_eigenvalues(mat, check_symmetry=True)
    assert len(computed) == 5
    # Should contain -0.5
    assert np.any(computed < 0)
    assert np.allclose(computed, [0.5, -0.5, 0.4, 0.3, 0.2], atol=1e-7)


def test_eigenvalue_table_and_distance_metrics():
    """Verify compilation of 4-patch eigenvalue table and pairwise distance matrix."""
    rng = np.random.RandomState(42)
    img = rng.randint(0, 256, size=(128, 128), dtype=np.uint8)

    crops = get_default_quadrant_crops(128, 128, patch_w=32, patch_h=32)
    patches = []
    for c in crops:
        p = extract_patch(img, c["x"], c["y"], c["w"], c["h"], label=c["label"], patch_id=c["id"])
        analyze_patch_glcm(p, distance=1, angle_rad=0.0)
        patches.append(p)

    df_eigs = build_eigenvalue_table(patches)
    assert len(df_eigs) == 4
    assert "λ1 (Dominant)" in df_eigs.columns
    assert "λ5" in df_eigs.columns

    df_dist = calculate_feature_distances(patches)
    assert df_dist.shape == (4, 4)
    # Diagonal must be 0.0
    for i in range(4):
        assert np.isclose(df_dist.iloc[i, i], 0.0)
    # Off-diagonal must be symmetric
    assert np.allclose(df_dist.values, df_dist.values.T)

    interp = generate_separability_interpretation(patches)
    assert "summary" in interp
    assert len(interp["details"]) >= 3


def test_glcm_angles_and_distances():
    """Verify GLCM handles varied distances and angles consistently."""
    patch = np.random.RandomState(42).randint(0, 5, size=(30, 30), dtype=np.uint8)
    
    # Distance 2, 45 degrees
    glcm_45, pairs_45 = compute_glcm(patch, distance=2, angle_rad=np.pi / 4, symmetric=True, normed=True)
    assert glcm_45.shape == (5, 5)
    assert np.isclose(np.sum(glcm_45), 1.0, atol=1e-7)
    assert np.allclose(glcm_45, glcm_45.T, atol=1e-7)
    assert pairs_45 > 0

    # Distance 1, 90 degrees (vertical)
    glcm_90, pairs_90 = compute_glcm(patch, distance=1, angle_rad=np.pi / 2, symmetric=True, normed=True)
    assert glcm_90.shape == (5, 5)
    assert np.isclose(np.sum(glcm_90), 1.0, atol=1e-7)


def test_glcm_insufficient_pairs_error():
    """Verify GLCM raises ValueError when patch has no adjacent valid pairs."""
    patch = np.zeros((2, 2), dtype=np.uint8)
    mask = np.zeros((2, 2), dtype=bool)
    mask[0, 0] = True  # Only one pixel valid -> 0 neighboring pairs
    
    with pytest.raises(ValueError, match="No valid pixel pairs found"):
        compute_glcm(patch, valid_mask=mask, distance=1, angle_rad=0.0)


def test_markdown_report_structure():
    """Verify Markdown report contains all required academic sections and metadata."""
    raster, data = _make_simple_raster(64, 64)
    pan_img = (data[0] / data[0].max() * 255).astype(np.uint8)
    crops = get_default_quadrant_crops(64, 64, patch_w=16, patch_h=16)
    patches = []
    for c in crops:
        p = extract_patch(pan_img, c["x"], c["y"], c["w"], c["h"], label=c["label"], patch_id=c["id"])
        analyze_patch_glcm(p)
        patches.append(p)

    df_eigs = build_eigenvalue_table(patches)
    df_dist = calculate_feature_distances(patches)
    interp = generate_separability_interpretation(patches)

    report = generate_markdown_report(
        raster=raster,
        selected_bands=[0, 1, 2, 3],
        pan_stats={"total_pixels": 4096, "valid_pixels": 4096, "min": 0, "max": 255, "mean": 128, "std": 30},
        equalized_stats={"total_pixels": 4096, "valid_pixels": 4096},
        patches=patches,
        df_eigenvalues=df_eigs,
        df_distances=df_dist,
        interpretation=interp
    )
    
    assert "Academic Research Report" in report
    assert "Panchromatic-like Image Synthesis" in report
    assert "Histogram Equalization" in report
    assert "Five-Level Gray Level Reduction" in report
    assert "GLCM Computation Parameters" in report
    assert "Numerical Results: GLCM Eigenvalues" in report
    assert "Pairwise Euclidean Distance" in report
    assert "Scientific Interpretation & Separability Discussion" in report
    assert "Methodological Limitations" in report


def test_complete_zip_package_generation():
    """Verify that full ZIP export produces valid archive with required artifacts."""
    raster, data = _make_simple_raster(64, 64)
    pan_img = (data[0] / data[0].max() * 255).astype(np.uint8)

    crops = get_default_quadrant_crops(64, 64, patch_w=16, patch_h=16)
    patches = []
    for c in crops:
        p = extract_patch(pan_img, c["x"], c["y"], c["w"], c["h"], label=c["label"], patch_id=c["id"])
        analyze_patch_glcm(p)
        patches.append(p)

    df_eigs = build_eigenvalue_table(patches)
    df_dist = calculate_feature_distances(patches)
    interp = generate_separability_interpretation(patches)
    hist_dummy = np.zeros(256, dtype=int)

    zip_bytes = build_complete_zip_package(
        raster=raster,
        selected_bands=[0, 1, 2, 3],
        pan_display_8bit=pan_img,
        equalized_8bit=pan_img,
        hist_before=hist_dummy,
        hist_after=hist_dummy,
        patches=patches,
        df_eigenvalues=df_eigs,
        df_distances=df_dist,
        interpretation=interp
    )
    
    assert len(zip_bytes) > 1000  # Non-trivial ZIP archive
    import zipfile, io
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        namelist = zf.namelist()
        assert "analysis_report.md" in namelist
        assert "metadata_summary.json" in namelist
        assert "tables/eigenvalues_summary.csv" in namelist
        assert "plots/figure_1_lambda1.png" in namelist
        assert "plots/figure_5_lambda5.png" in namelist
