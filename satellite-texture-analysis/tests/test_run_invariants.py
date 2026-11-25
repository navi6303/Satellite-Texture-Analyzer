"""Numerical assertions per completed analysis. The normal suite uses a known fixture."""

import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def run_data(tmp_path_factory):
    if "TEXTURE_CHECK_DIR" in os.environ:
        root = Path(os.environ["TEXTURE_CHECK_DIR"])
    else:
        from modules.glcm_analysis import (
            compute_glcm,
            compute_eigenvalues,
            fisher_ratios,
        )

        root = tmp_path_factory.mktemp("invariants")
        rng = np.random.default_rng(40)
        rows = []
        for i in range(6):
            equalized = rng.integers(0, 256, (9, 12), dtype=np.uint8)
            q = (equalized.astype(np.uint16) * 5 // 256).astype(np.uint8)
            mask = np.ones_like(q, dtype=bool)
            if i % 2:
                mask[1, 4] = False
            matrix, pairs = compute_glcm(q, mask)
            eig = compute_eigenvalues(matrix)
            rows.append(eig)
            np.savez(
                root / f"patch_{i}.npz",
                patch_id=str(i),
                q=q,
                native_shape=q.shape,
                raw_counts=[compute_glcm(q, mask, symmetric=False, normed=False)[0]],
                equalized=equalized,
                mode="global_equalized_fixed5",
                mask=mask,
                matrix=matrix,
                eig=eig,
                pairs=pairs,
                distances=[1],
                angles=[0],
                label="a" if i < 3 else "b",
            )
        fisher = fisher_ratios(np.array(rows), ["a"] * 3 + ["b"] * 3, list("ABCDE"))
        np.save(root / "fisher.npy", fisher.fisher_ratio)
    return root


def patches(folder):
    files = sorted(folder.glob("patch_*.npz"))
    assert files, "No analysis patches"
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            yield data


def test_glcm_symmetric(run_data):
    for p in patches(run_data):
        np.testing.assert_allclose(
            p["matrix"], p["matrix"].T, rtol=0, atol=1e-12, err_msg=str(p["patch_id"])
        )


def test_glcm_probability(run_data):
    for p in patches(run_data):
        assert p["matrix"].shape == (5, 5) and (p["matrix"] >= 0).all(), p["patch_id"]
        assert abs(p["matrix"].sum() - 1) <= 1e-12, p["patch_id"]


def test_trace_identity(run_data):
    for p in patches(run_data):
        assert abs(p["eig"].sum() - np.trace(p["matrix"])) <= 1e-9, p["patch_id"]


def test_descending_spectrum(run_data):
    for p in patches(run_data):
        assert p["eig"].shape == (5,) and (np.diff(np.abs(p["eig"])) <= 1e-14).all(), p[
            "patch_id"
        ]


def test_horizontal_valid_pairs(run_data):
    for p in patches(run_data):
        m = p["mask"]
        h, w = m.shape
        expected = h * (w - 1) if m.all() else np.count_nonzero(m[:, 1:] & m[:, :-1])
        assert p["pairs"] == expected, p["patch_id"]


def test_five_level_domain(run_data):
    for p in patches(run_data):
        values = p["q"][p["mask"]]
        assert values.size and np.isin(values, np.arange(5)).all(), p["patch_id"]


def test_finite_fisher_ratios(run_data):
    ratios = np.load(run_data / "fisher.npy")
    labels = [str(p["label"]) for p in patches(run_data)]
    if min(labels.count(label) for label in set(labels)) < 2:
        assert ratios.size == 0, "A single box cannot estimate within-class scatter"
    else:
        assert ratios.size and np.isfinite(ratios).all()


def test_mean_matrix_reconstructed_from_actual_window(run_data):
    for p in patches(run_data):
        q, mask = p["q"], p["mask"]
        h, w = q.shape
        assert tuple(p["native_shape"]) == q.shape == mask.shape
        matrices = []
        for distance in p["distances"]:
            for theta in np.deg2rad(p["angles"]):
                dy, dx = np.rint(
                    distance * np.array([np.sin(theta), np.cos(theta)])
                ).astype(int)
                # Slice the overlap of the original image and its shifted copy.
                y0, y1 = max(0, -dy), min(h, h - dy)
                x0, x1 = max(0, -dx), min(w, w - dx)
                a = np.s_[y0:y1, x0:x1]
                b = np.s_[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
                valid = mask[a] & mask[b]
                codes = q[a][valid].astype(np.uint16) * 5 + q[b][valid]
                directed = np.bincount(codes, minlength=25).reshape(5, 5)
                raw = p["raw_counts"][len(matrices)]
                np.testing.assert_array_equal(raw, directed, err_msg=str(p["patch_id"]))
                assert raw.sum() == valid.sum()
                symmetric = directed + directed.T
                matrices.append(symmetric / symmetric.sum())
        expected = np.mean(matrices, axis=0)
        np.testing.assert_allclose(p["matrix"], expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            np.sort(p["eig"]), np.linalg.eigvalsh(expected), atol=1e-12, rtol=0
        )


def test_global_fixed_bins_match_equalized_pixels(run_data):
    checked = 0
    for p in patches(run_data):
        if str(p["mode"]) == "global_equalized_fixed5":
            valid = p["mask"]
            expected = np.digitize(p["equalized"][valid], [52, 103, 154, 205])
            np.testing.assert_array_equal(
                p["q"][valid], expected, err_msg=str(p["patch_id"])
            )
            checked += 1
    if not checked:
        pytest.skip("Run uses an alternative quantization mode")
