"""
Tests for Image Loading and Preprocessing Modules.
Validates:
  - Pixel-wise band averaging on known numerical arrays
  - Nodata, NaN, and Inf handling
  - Linear 8-bit normalization (including constant images)
  - Mask-aware histogram equalization
"""

import numpy as np
import pytest

from modules.image_loader import (
    LoadedRaster,
    RasterMetadata,
)
from modules.preprocessing import (
    generate_panchromatic,
    get_image_statistics,
    histogram_equalization,
    normalize_to_8bit
)


def _make_test_raster(bands_list, nodata=None):
    """Helper: create a minimal in-memory LoadedRaster from a list of 2D float32 arrays."""
    n = len(bands_list)
    h, w = bands_list[0].shape
    data = np.stack(bands_list, axis=0).astype(np.float32)
    meta = RasterMetadata(
        width=w, height=h, count=n,
        dtypes=tuple("float32" for _ in range(n)),
        crs="CRS not defined",
        nodata=nodata
    )
    return LoadedRaster(metadata=meta, band_names=[f"B{i+1}" for i in range(n)], in_memory_data=data)



def test_synthetic_sample_generation():
    """Replaced: creates a simple in-memory 4-band raster without the deprecated synthetic helper."""
    b_list = [np.random.rand(128, 128).astype(np.float32) * 1000 for _ in range(4)]
    sample = _make_test_raster(b_list)
    assert sample.count == 4
    assert sample.width == 128
    assert sample.height == 128


def test_band_averaging_known_array():
    """Verify exact pixel-wise mean calculation across selected spectral bands."""
    # 3 bands of shape 2x2
    b1 = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    b2 = np.array([[20.0, 40.0], [60.0, 80.0]], dtype=np.float32)
    b3 = np.array([[30.0, 60.0], [90.0, 120.0]], dtype=np.float32)

    raster = _make_test_raster([b1, b2, b3])

    # Average all 3 bands: (b1 + b2 + b3) / 3
    pan_float, mask, pan_8bit = generate_panchromatic(raster, selected_bands=[0, 1, 2])
    expected = (b1 + b2 + b3) / 3.0

    assert np.allclose(pan_float, expected, atol=1e-6)
    assert np.all(mask)
    assert pan_8bit.shape == (2, 2)
    assert pan_8bit[0, 0] == 0    # Min value maps to 0
    assert pan_8bit[1, 1] == 255  # Max value maps to 255

    # Subset: average only bands 0 and 2: (b1 + b3) / 2
    pan_sub, _, _ = generate_panchromatic(raster, selected_bands=[0, 2])
    expected_sub = (b1 + b3) / 2.0
    assert np.allclose(pan_sub, expected_sub, atol=1e-6)


def test_band_averaging_nodata_and_nan():
    """Verify that pixels containing NaN, Inf, or nodata are excluded and masked."""
    b1 = np.array([[10.0, np.nan], [30.0, 40.0]], dtype=np.float32)
    b2 = np.array([[20.0, 30.0], [np.inf, 80.0]], dtype=np.float32)

    raster = _make_test_raster([b1, b2])

    pan_float, mask, _ = generate_panchromatic(raster, selected_bands=[0, 1])

    # Only pixel (0, 0) should be valid: b1[0,0]=10 is finite, b2[0,0]=20 is finite
    assert mask[0, 0] is True or mask[0, 0] == 1
    assert mask[0, 1] is False or mask[0, 1] == 0  # Due to NaN in b1
    assert mask[1, 0] is False or mask[1, 0] == 0  # Due to Inf in b2
    assert np.isclose(pan_float[0, 0], 15.0, atol=1e-6)


def test_normalize_to_8bit():
    """Test 8-bit normalization under varying dynamic ranges and constant image edge cases."""
    # Standard linear stretch
    arr = np.array([[0.0, 50.0], [100.0, 200.0]])
    res = normalize_to_8bit(arr)
    assert res[0, 0] == 0
    assert res[1, 1] == 255
    assert np.isclose(res[0, 1], np.round(50 / 200 * 255))

    # Constant image (all identical values)
    constant_arr = np.full((10, 10), 42.0)
    res_const = normalize_to_8bit(constant_arr)
    assert np.all(res_const == 128)  # Mid-gray fallback, no crash

    # Empty valid mask with matching shape
    all_masked = np.zeros((10, 10), dtype=bool)
    res_empty = normalize_to_8bit(constant_arr, valid_mask=all_masked)
    assert np.all(res_empty == 0)

    # Mismatched shape raises ValueError
    with pytest.raises(ValueError):
        normalize_to_8bit(constant_arr, valid_mask=np.zeros((5, 5), dtype=bool))


def test_histogram_equalization_properties():
    """Verify global histogram equalization redistributes contrast and preserves masks."""
    # Non-constant gradient image
    gradient = np.tile(np.linspace(20, 180, 50, dtype=np.uint8), (50, 1))
    eq_img, h_before, h_after, bins = histogram_equalization(gradient)
    
    assert eq_img.shape == (50, 50)
    assert len(bins) == 256
    assert np.sum(h_before) == 50 * 50
    assert np.sum(h_after) == 50 * 50
    # Equalization expands dynamic range
    assert int(np.min(eq_img)) <= int(np.min(gradient))
    assert int(np.max(eq_img)) >= int(np.max(gradient))

    # Constant image
    const_img = np.full((30, 30), 100, dtype=np.uint8)
    eq_const, _, _, _ = histogram_equalization(const_img)
    assert np.all(eq_const == 100)

    # Masked image
    masked_img = gradient.copy()
    mask = np.ones((50, 50), dtype=bool)
    mask[:10, :] = False  # Mask first 10 rows
    eq_masked, h_b_m, h_a_m, _ = histogram_equalization(masked_img, valid_mask=mask)
    assert np.sum(h_b_m) == 40 * 50
    assert np.sum(h_a_m) == 40 * 50
    assert np.all(eq_masked[:10, :] == 0)  # Masked pixels remain zero


def test_image_statistics():
    """Verify statistical summary calculations."""
    # Pass a 2D array so get_image_statistics works as expected
    data = np.array([[10.0, 20.0, 30.0, 40.0, 50.0]])  # shape (1, 5)
    mask = np.array([[True, True, True, True, False]])
    stats = get_image_statistics(data, valid_mask=mask)

    assert stats["total_pixels"] == 5
    assert stats["valid_pixels"] == 4
    assert stats["invalid_pixels"] == 1
    assert np.isclose(stats["mean"], 25.0)
    assert np.isclose(stats["min"], 10.0)
    assert np.isclose(stats["max"], 40.0)


def test_image_statistics_none_and_mismatched_safe():
    """Verify get_image_statistics safely returns empty dict when image is None or mismatched."""
    stats_none = get_image_statistics(None, valid_mask=None)
    assert stats_none["total_pixels"] == 0
    assert stats_none["valid_pixels"] == 0

    # Image with mismatched mask should fallback gracefully without raising uncaught exception
    img = np.ones((10, 10))
    mismatched_mask = np.ones((5, 5), dtype=bool)
    stats_mismatch = get_image_statistics(img, valid_mask=mismatched_mask)
    assert stats_mismatch["total_pixels"] == 100


def test_fast_histogram_equalization_speed_and_correctness():
    """Verify vectorized histogram equalization runs quickly and covers dynamic range."""
    large_img = np.random.RandomState(42).randint(30, 220, size=(1000, 1000), dtype=np.uint8)
    eq_img, h_before, h_after, bins = histogram_equalization(large_img)
    
    assert eq_img.shape == (1000, 1000)
    assert int(np.min(eq_img)) <= 10
    assert int(np.max(eq_img)) >= 245
    assert len(bins) == 256
    assert np.sum(h_before) == 1000 * 1000
    assert np.sum(h_after) == 1000 * 1000
