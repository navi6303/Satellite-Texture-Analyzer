from types import SimpleNamespace

import numpy as np
import pytest

from modules.glcm_analysis import (
    compute_glcm,
    compute_averaged_glcm,
    row_normalized_transition_eigenvalues,
)
from modules.patch_sampling import assign_groups, sample_layout, boxes_overlap
from modules.texture_extraction import (
    QUANT_PER_PATCH_RESCALE,
    QUANT_RAW_PAN_FIXED5,
    quantize_by_mode,
)
from modules.validation import blocked_group_folds


def test_sampling_reserves_manual_crops_and_replays_seed():
    roi = {"a": dict(x=0, y=0, w=144, h=144), "b": dict(x=60, y=60, w=144, h=144)}
    explicit = {"b": [dict(x=72, y=72, w=16, h=16)]}
    args = (["a", "b"], explicit, roi, 5, 16, 48, 42, 204, 204)
    first = sample_layout(*args)
    assert first == sample_layout(*args)
    boxes = first["a"] + first["b"]
    assert len(boxes) == 10
    assert not any(
        boxes_overlap(b, c) for i, b in enumerate(boxes) for c in boxes[i + 1 :]
    )
    with pytest.raises(ValueError, match="overlaps"):
        sample_layout(
            ["a", "b"],
            {"a": explicit["b"], "b": explicit["b"]},
            roi,
            5,
            16,
            48,
            42,
            204,
            204,
        )


def test_crossing_patches_join_every_touched_grid_cell():
    patches = [
        SimpleNamespace(x=x, y=0, width=8, height=8, extra={}) for x in (0, 12, 20, 48)
    ]
    assign_groups(patches, 16, 64)
    groups = [p.block_id for p in patches]
    assert groups[0] == groups[1] == groups[2] != groups[3]
    held = []
    for train, test in blocked_group_folds(groups):
        train_cells = {cell for i in train for cell in patches[i].extra["grid_cells"]}
        test_cells = {cell for i in test for cell in patches[i].extra["grid_cells"]}
        assert train_cells.isdisjoint(test_cells)
        held.extend(test)
    assert sorted(held) == list(range(4))


@pytest.mark.parametrize("angle", [0, 45, 90, 135])
@pytest.mark.parametrize("distance", [1, 2, 3])
def test_masked_direction_matches_masked_brute_force(angle, distance):
    rng = np.random.default_rng(13)
    q = rng.integers(0, 5, (11, 15), dtype=np.uint8)
    mask = rng.random(q.shape) > 0.1
    theta = np.radians(angle)
    dy, dx = np.rint(distance * np.array([np.sin(theta), np.cos(theta)])).astype(int)
    counts = np.zeros((5, 5))
    for y, x in zip(*np.where(mask)):
        r, c = y + dy, x + dx
        if 0 <= r < q.shape[0] and 0 <= c < q.shape[1] and mask[r, c]:
            counts[q[y, x], q[r, c]] += 1
    P, n = compute_glcm(q, mask, distance, theta)
    assert n == counts.sum()
    np.testing.assert_allclose(P, (counts + counts.T) / (2 * n))


def test_four_angle_rotation_and_transition_spectrum():
    q = np.random.default_rng(2).integers(0, 4, (16, 20), dtype=np.uint8)
    result = [
        compute_averaged_glcm(a, angles_deg=[0, 45, 90, 135])[0]
        for a in (q, np.rot90(q))
    ]
    np.testing.assert_allclose(*result)
    P = result[0]
    masses = P.sum(1)[:, None]
    T = np.divide(P, masses, out=np.zeros_like(P), where=masses > 0)
    np.testing.assert_allclose(
        row_normalized_transition_eigenvalues(P),
        np.sort(np.linalg.eigvals(T))[::-1],
        atol=1e-12,
    )


def test_raw_bins_preserve_fractional_native_dn_and_local_range():
    raw = np.array([[51.0, 51.3333, 102.3333, 102.6667, 255.0]])
    assert quantize_by_mode(raw, mode=QUANT_RAW_PAN_FIXED5)[0].tolist() == [
        [0, 1, 1, 2, 4]
    ]
    uniform = np.full((10, 10), 80)
    assert quantize_by_mode(uniform, mode=QUANT_PER_PATCH_RESCALE)[1] == [2]


def test_nonnegative_and_signed_bar_axes():
    import matplotlib.pyplot as plt
    from modules.visualization import _apply_nonneg_ylim

    fig, ax = plt.subplots()
    means = np.array([0.002, 0.003])
    err = np.array([0.005, 0.001])
    _apply_nonneg_ylim(ax, means, err)
    assert ax.get_ylim()[0] == 0
    _apply_nonneg_ylim(ax, means, err, signed_samples=True)
    assert ax.get_ylim()[0] <= -0.003
    plt.close(fig)


def test_geotiff_windows_masked_rgb_and_quantization_modes(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from modules.image_loader import load_geotiff_from_path
    from modules.scene import prepare_scene, equalize_scene, read_patch
    from modules.texture_extraction import QUANT_GLOBAL_EQ_FIXED5

    values = np.random.default_rng(8).integers(40, 131, (3, 70, 83), dtype=np.uint8)
    mask = np.ones(values.shape[1:], np.uint8) * 255
    mask[0, 2:5] = 0
    path = tmp_path / "source.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        dtype="uint8",
        count=3,
        width=83,
        height=70,
        transform=from_origin(5000, 6000, 10, 10),
    ) as dst:
        dst.write(values)
        dst.write_mask(mask)
    r = load_geotiff_from_path(path, path.name)
    r.read_band = lambda *args: pytest.fail("Full-band read")
    scene = equalize_scene(prepare_scene(r, [0, 1, 2], tmp_path / "scene"))
    assert scene.stats["valid_pixels"] == np.count_nonzero(mask)
    box = dict(x=0, y=0, w=23, h=17)
    for mode in (QUANT_PER_PATCH_RESCALE, QUANT_RAW_PAN_FIXED5, QUANT_GLOBAL_EQ_FIXED5):
        patch = read_patch(
            scene, box, "Water Bodies", "p", mode, "percentile", [0, 1, 2]
        )
        sub = values[:, :17, :23]
        valid = mask[:17, :23].astype(bool)
        np.testing.assert_allclose(patch.rgb_mean, sub[:, valid].mean(axis=1))
        if mode == QUANT_RAW_PAN_FIXED5:
            expect = np.floor(5 * sub.mean(axis=0) / 256).astype(np.uint8)
            np.testing.assert_array_equal(patch.quantized[valid], expect[valid])
        assert patch.occupancy.sum() == pytest.approx(1)


def test_blocked_cv_refuses_class_confined_to_one_group():
    from modules.glcm_analysis import analyze_patch_glcm
    from modules.texture_extraction import extract_patch
    from modules.validation import compare_feature_sets

    img = np.arange(256, dtype=np.uint8).reshape(16, 16)
    patches = [
        analyze_patch_glcm(extract_patch(img, 0, 0, 16, 16, label=lab, block_id=block))
        for lab, block in [("a", 1), ("a", 1), ("b", 2), ("b", 3)]
    ]
    result = compare_feature_sets(patches)
    assert result["best_accuracy"] is None
    assert "only one independent block" in result["by_set"]["combined"]["note"]


@pytest.mark.parametrize("angle", [0, 45, 90, 135])
@pytest.mark.parametrize("distance", [1, 2, 3])
def test_numpy_glcm_without_scikit_image(monkeypatch, angle, distance):
    from modules import glcm_analysis
    monkeypatch.setattr(glcm_analysis, "graycomatrix", None)
    q = np.random.default_rng(31).integers(0, 5, (17, 23), dtype=np.uint8)
    theta = np.deg2rad(angle)
    dr, dc = np.rint(distance * np.array([np.sin(theta), np.cos(theta)])).astype(int)
    expected = np.zeros((5, 5))
    for r, c in np.ndindex(q.shape):
        if 0 <= r + dr < q.shape[0] and 0 <= c + dc < q.shape[1]:
            expected[q[r, c], q[r + dr, c + dc]] += 1
    for mask in (None, np.ones(q.shape, dtype=bool)):
        actual, pairs = glcm_analysis.compute_glcm(q, mask, distance, theta, symmetric=False, normed=False)
        assert pairs == expected.sum()
        np.testing.assert_array_equal(actual, expected)
