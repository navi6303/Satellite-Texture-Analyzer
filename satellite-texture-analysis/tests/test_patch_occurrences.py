"""Regression checks for confusion between native subset arrays and selected boxes."""

import numpy as np
import pytest

from modules.glcm_analysis import analyze_patch_glcm, compute_glcm, get_glcm_demonstration_data
from modules.matrix_audit import directional_table
from modules.texture_extraction import extract_patch, QUANT_GLOBAL_EQ_FIXED5


def enumerate_pairs(image, mask, dy, dx):
    counts = np.zeros((5, 5), dtype=np.int64)
    for y, x in np.ndindex(image.shape):
        r, c = y + dy, x + dx
        if 0 <= r < image.shape[0] and 0 <= c < image.shape[1] and mask[y, x] and mask[r, c]:
            counts[image[y, x], image[r, c]] += 1
    return counts


def test_distinct_touching_boxes_have_independent_complete_occurrence_tables():
    eq = np.full((74, 98), 230, dtype=np.uint8)
    mask = np.ones_like(eq, dtype=bool)
    # Non-square adjacent boxes embedded in an unrelated background. Include several
    # different ordered pairs, plus one invalid pixel and a one-way boundary.
    rng = np.random.default_rng(3306)
    eq[19:27, 31:55] = rng.integers(0, 255, (8, 24), dtype=np.uint8)
    mask[22, 34] = False
    patches = []
    for n, x in enumerate([31, 43]):
        p = extract_patch(eq, x, 19, 12, 8, label="Urban", patch_id=f"box{n}",
                          valid_mask=mask, quantization_mode=QUANT_GLOBAL_EQ_FIXED5)
        analyze_patch_glcm(p, distances=[2], angles_deg=[0, 90, 135])
        patches.append(p)
        demo = get_glcm_demonstration_data(p, distance=2, angles_deg=[0, 90, 135])
        q = np.digitize(eq[19:27, x:x+12], [52, 103, 154, 205])
        for theta, dy, dx in [(0, 0, 2), (90, 2, 0), (135, 1, -1)]:
            expected = enumerate_pairs(q, mask[19:27, x:x+12], dy, dx)
            info = demo["directions"][theta]
            np.testing.assert_array_equal(info["raw_counts"], expected)
            np.testing.assert_array_equal(info["symmetric_counts"], expected + expected.T)
            assert info["possible_pairs"] == (8 - abs(dy)) * (12 - abs(dx))
            rows = info["occurrences"]
            assert len(rows) == np.count_nonzero(expected) > 10
            assert sum(r["Occurrences"] for r in rows) == info["valid_pairs"] == expected.sum()
            assert {r["Pair (i, j)"]: r["Occurrences"] for r in rows} == {
                f"({i}, {j})": expected[i, j] for i, j in np.ndindex(5, 5) if expected[i, j]
            }
            np.testing.assert_allclose(info["normalized_matrix"], (expected + expected.T)/(2*expected.sum()))
        # Changing the demonstration's offset must give eigenvalues of its displayed matrix.
        different = get_glcm_demonstration_data(p, distance=1, angles_deg=[0])
        np.testing.assert_allclose(np.sort(different["final_eigenvalues"]),
                                   np.linalg.eigvalsh(different["averaged_glcm"]))
    audit = directional_table(patches)
    assert len(audit) == 6
    assert list(audit.query("angle_deg == 0")["raw_matrix_sum"]) == [78, 80]
    np.testing.assert_array_equal(audit["symmetric_matrix_sum"], 2 * audit["raw_matrix_sum"])


def test_reject_parent_array_and_mismatched_masks():
    eq = np.full((50, 72), 100, dtype=np.uint8)
    with pytest.raises(ValueError, match="mask shape"):
        extract_patch(eq, 5, 8, 12, 9, valid_mask=np.ones((9, 12), bool))
    p = extract_patch(eq, 5, 8, 12, 9, patch_id="small")
    p.quantized = np.zeros_like(eq)
    with pytest.raises(ValueError, match="small.*quantized image shape"):
        analyze_patch_glcm(p)
    with pytest.raises(ValueError, match="small.*quantized image shape"):
        get_glcm_demonstration_data(p)
    with pytest.raises(ValueError, match="mask shape"):
        compute_glcm(p.quantized, p.valid_mask)
    p.quantized = np.zeros((p.height, p.width), dtype=np.uint8)
    p.extra["pan_patch"] = eq
    with pytest.raises(ValueError, match="small.*pan_patch shape"):
        get_glcm_demonstration_data(p)
