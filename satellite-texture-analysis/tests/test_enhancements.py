"""
Tests for SIP Project Enhancements:
  - Interactive bounding-box coordinate mapping & patch extraction
  - Minimum 3 bounding boxes per texture class (minimum 12 patches)
  - 5-level gray quantization mapping
  - Step-by-step GLCM calculation (neighbor pairs, raw count matrix, symmetrization, normalization)
  - Final eigenvalue computation and invariance (trace, magnitude sorting)
  - Individual and combined eigenvalue visualization plots
  - Separability analysis
"""

import numpy as np
import pytest

from modules.glcm_analysis import (
    compute_glcm,
    compute_eigenvalues,
    extract_directional_pixel_pairs,
    get_glcm_demonstration_data,
)
from modules.patch_sampling import boxes_overlap
from modules.texture_extraction import (
    TexturePatch,
    extract_patch,
    quantize_to_5_levels,
    validate_crop,
    QUANT_GLOBAL_EQ_FIXED5,
)
from modules.visualization import (
    plot_individual_class_eigenvalues,
    plot_combined_eigenvalue_spectra,
)


def test_bounding_box_mapping_and_validation():
    orig_w, orig_h = 1000, 800

    # Valid bounding box
    ok, msg = validate_crop(100, 150, 64, 64, orig_w, orig_h, min_size=8)
    assert ok is True

    # Smaller than minimum size (min_size=8)
    ok, msg = validate_crop(100, 150, 6, 6, orig_w, orig_h, min_size=8)
    assert ok is False
    assert "smaller than minimum" in msg

    # Negative coordinates
    ok, msg = validate_crop(-5, 50, 64, 64, orig_w, orig_h, min_size=8)
    assert ok is False
    assert "negative" in msg

    # Exceeding raster boundaries
    ok, msg = validate_crop(950, 100, 64, 64, orig_w, orig_h, min_size=8)
    assert ok is False
    assert "exceeds raster width" in msg

    ok, msg = validate_crop(100, 780, 64, 64, orig_w, orig_h, min_size=8)
    assert ok is False
    assert "exceeds raster height" in msg


def test_patch_extraction_contains_exact_pixels():
    # Synthetic raster 100x100 with unique deterministic pixel values
    full_img = np.arange(100 * 100, dtype=np.uint8).reshape((100, 100))
    x, y, w, h = 20, 30, 25, 20

    patch = extract_patch(
        full_img,
        x=x,
        y=y,
        width=w,
        height=h,
        label="TestClass",
        patch_id="patch_test_01",
    )

    # Check dimensions
    assert patch.width == w
    assert patch.height == h
    assert patch.image_8bit.shape == (h, w)

    # Check pixel containment: must exactly match the sub-region of full_img
    expected_pixels = full_img[y : y + h, x : x + w]
    np.testing.assert_array_equal(patch.image_8bit, expected_pixels)

    # Check that extra stores pan_patch and eq_patch
    assert "pan_patch" in patch.extra
    assert "eq_patch" in patch.extra
    assert patch.extra["pan_patch"].shape == (h, w)


def test_bounding_box_overlap_check():
    box_a = {"x": 10, "y": 10, "w": 50, "h": 50}
    # Box clearly overlapping with box_a
    box_overlap = {"x": 30, "y": 30, "w": 40, "h": 40}
    assert boxes_overlap(box_a, box_overlap) is True

    # Box disjoint horizontally
    box_disjoint_h = {"x": 70, "y": 10, "w": 30, "h": 30}
    assert boxes_overlap(box_a, box_disjoint_h) is False

    # Box disjoint vertically
    box_disjoint_v = {"x": 10, "y": 70, "w": 30, "h": 30}
    assert boxes_overlap(box_a, box_disjoint_v) is False


def test_minimum_patch_validation_logic():
    user_classes = ["Water", "Vegetation", "Built-up", "Bare land"]
    confirmed_boxes = {c: [] for c in user_classes}

    def validate_requirements(classes, boxes):
        if len(classes) != 4 or len(set(classes)) != 4:
            return False, "Exactly 4 distinct classes required."
        for c in classes:
            if len(boxes.get(c, [])) < 3:
                return False, f"Class '{c}' has fewer than 3 patches."
        total = sum(len(boxes.get(c, [])) for c in classes)
        if total < 12:
            return False, f"Total patches ({total}) is less than minimum 12."
        return True, "All requirements met."

    # Initially empty
    valid, msg = validate_requirements(user_classes, confirmed_boxes)
    assert valid is False
    assert "fewer than 3" in msg

    # Add 3 patches to 3 classes only
    for c in user_classes[:3]:
        confirmed_boxes[c] = [{"x": i * 20, "y": 10, "w": 16, "h": 16} for i in range(3)]
    valid, msg = validate_requirements(user_classes, confirmed_boxes)
    assert valid is False
    assert "Bare land" in msg

    # Add 3 patches to the 4th class -> total 12 patches
    confirmed_boxes["Bare land"] = [{"x": i * 20, "y": 100, "w": 16, "h": 16} for i in range(3)]
    valid, msg = validate_requirements(user_classes, confirmed_boxes)
    assert valid is True

    # Adding extra patches (e.g. 4 patches for Water) remains valid
    confirmed_boxes["Water"].append({"x": 90, "y": 10, "w": 16, "h": 16})
    valid, msg = validate_requirements(user_classes, confirmed_boxes)
    assert valid is True


def test_five_level_quantization_exact_ranges():
    # Boundary values for 5 intervals:
    # [0, 51] -> 0, [52, 102] -> 1, [103, 153] -> 2, [154, 204] -> 3, [205, 255] -> 4
    test_vals = np.array([0, 25, 51, 52, 75, 102, 103, 128, 153, 154, 180, 204, 205, 230, 255], dtype=np.uint8)
    expected = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]

    quantized, levels = quantize_to_5_levels(test_vals)
    np.testing.assert_array_equal(quantized, expected)
    assert set(levels).issubset({0, 1, 2, 3, 4})


def test_step_by_step_glcm_breakdown():
    # Construct a known 4x4 patch with 5 gray levels
    q_patch = np.array([
        [0, 1, 2, 3],
        [1, 1, 2, 2],
        [2, 2, 3, 4],
        [3, 4, 4, 4],
    ], dtype=np.uint8)

    # Direction: horizontal (theta = 0, distance = 1)
    ref_vals, neigh_vals, raw_counts, valid_pairs = extract_directional_pixel_pairs(
        q_patch, valid_mask=None, distance=1, angle_deg=0.0, levels=5
    )

    # For 4x4 patch at d=1 horizontal, expected pairs = 4 * (4 - 1) = 12
    assert valid_pairs == 12
    assert len(ref_vals) == 12
    assert len(neigh_vals) == 12
    assert raw_counts.shape == (5, 5)
    assert raw_counts.sum() == 12

    # Verify specific pairs in raw_counts:
    # Row 0: (0, 1), (1, 2), (2, 3)
    # Row 1: (1, 1), (1, 2), (2, 2)
    # Row 2: (2, 2), (2, 3), (3, 4)
    # Row 3: (3, 4), (4, 4), (4, 4)
    assert raw_counts[0, 1] == 1
    assert raw_counts[1, 2] == 2
    assert raw_counts[4, 4] == 2

    # Symmetrize
    sym_counts = raw_counts + raw_counts.T
    assert np.allclose(sym_counts, sym_counts.T)
    assert sym_counts.sum() == 2 * valid_pairs == 24

    # Normalize
    norm_mat = sym_counts / sym_counts.sum()
    assert pytest.approx(norm_mat.sum(), rel=1e-9) == 1.0
    assert np.allclose(norm_mat, norm_mat.T)


def test_glcm_demonstration_data_structure():
    # Create sample TexturePatch
    q = np.array([
        [0, 1, 1, 2, 3, 4],
        [1, 1, 2, 2, 3, 4],
        [1, 2, 2, 3, 3, 4],
        [2, 2, 3, 3, 4, 4],
        [3, 3, 4, 4, 4, 4],
        [4, 4, 4, 4, 4, 4],
    ], dtype=np.uint8)
    img_8bit = (q * 51).astype(np.uint8)
    mask = np.ones_like(q, dtype=bool)

    patch = TexturePatch(
        patch_id="patch_1_01",
        label="Water Bodies",
        x=10,
        y=20,
        width=6,
        height=6,
        image_8bit=img_8bit,
        quantized=q,
        valid_mask=mask,
        extra={
            "pan_patch": img_8bit.astype(float),
            "eq_patch": img_8bit,
        }
    )

    demo = get_glcm_demonstration_data(patch, distance=1, angles_deg=[0, 45, 90, 135], levels=5)

    # Check Step 1
    assert demo["step1"]["patch_id"] == "patch_1_01"
    assert demo["step1"]["label"] == "Water Bodies"
    assert demo["step1"]["width"] == 6 and demo["step1"]["height"] == 6

    # Check Step 2 (Panchromatic)
    assert "pan_image" in demo["step2"]
    assert "mean" in demo["step2"]
    assert "sample_matrix" in demo["step2"]

    # Check Step 3 (Equalized)
    assert "eq_image" in demo["step3"]
    assert "explanation" in demo["step3"]

    # Check Step 4 (Quantization)
    assert demo["step4"]["levels"] == [0, 1, 2, 3, 4]
    assert len(demo["step4"]["mapping_table"]) == 5

    # Check Step 5 (Directions)
    assert demo["step5"]["distance"] == 1
    assert 0 in demo["directions"]
    assert 45 in demo["directions"]
    assert 90 in demo["directions"]
    assert 135 in demo["directions"]

    # Check Step 6 & 7 (Neighbor pairs and raw count matrix)
    dir0 = demo["directions"][0]
    assert len(dir0["sample_pairs"]) > 0
    assert dir0["raw_counts"].shape == (5, 5)
    assert dir0["valid_pairs"] > 0

    # Check Step 8 (Symmetric matrix)
    assert np.allclose(dir0["symmetric_counts"], dir0["symmetric_counts"].T)
    assert dir0["symmetric_pair_total"] == 2 * dir0["valid_pairs"]

    # Check Step 9 (Normalized matrix)
    assert pytest.approx(dir0["normalized_sum"], rel=1e-7) == 1.0
    assert pytest.approx(demo["averaged_glcm_sum"], rel=1e-7) == 1.0

    # Check Final Eigenvalues (Requirement 5)
    e_vals = demo["final_eigenvalues"]
    assert len(e_vals) == 5
    # Magnitude ordering
    assert np.all(np.abs(e_vals[:-1]) >= np.abs(e_vals[1:]) - 1e-9)
    # Trace equals sum
    assert pytest.approx(demo["eigenvalue_trace"], rel=1e-7) == float(np.sum(e_vals))


def test_eigenvalue_plots_execution():
    # Build 12 patches (3 per class across 4 classes)
    classes = ["Water", "Vegetation", "Built-up", "Bare land"]
    patches = []
    rng = np.random.default_rng(123)

    for c_idx, c_name in enumerate(classes, 1):
        for p_idx in range(1, 4):
            q = rng.integers(0, 5, size=(16, 16), dtype=np.uint8)
            img = q * 51
            matrix, _ = compute_glcm(q, distance=1, angle_rad=0.0)
            eigs = compute_eigenvalues(matrix)
            p = TexturePatch(
                patch_id=f"patch_{c_idx}_{p_idx:02d}",
                label=c_name,
                x=c_idx * 20,
                y=p_idx * 20,
                width=16,
                height=16,
                image_8bit=img,
                quantized=q,
                valid_mask=np.ones_like(q, dtype=bool),
                glcm=matrix,
                eigenvalues=eigs,
            )
            patches.append(p)

    assert len(patches) == 12

    # Test Plot 1: Individual eigenvalue plots
    fig1, png1 = plot_individual_class_eigenvalues(patches)
    assert fig1 is not None
    assert len(png1) > 1000

    # Test Plot 2: Combined eigenvalue comparison
    fig2, png2 = plot_combined_eigenvalue_spectra(patches)
    assert fig2 is not None
    assert len(png2) > 1000
