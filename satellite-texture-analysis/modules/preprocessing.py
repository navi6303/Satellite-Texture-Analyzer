"""
Preprocessing Module for Satellite Imagery.
Implements:
  1. Pixel-wise arithmetic band averaging across all selected spectral channels
     at 100% full original spatial resolution:
       P(x, y) = (1 / B) * sum_{b=1}^{B} I_b(x, y)
  2. Nodata and invalid-pixel mask tracking without resolution loss.
  3. Safe linear 8-bit normalization on valid pixels only.
  4. Mask-aware global histogram equalization preserving original image dimensions.
  5. Statistical characterization of raw and enhanced intensities.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from modules.image_loader import LoadedRaster


def generate_panchromatic(
    raster: LoadedRaster,
    selected_bands: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a panchromatic-like grayscale image by computing the pixel-wise
    arithmetic mean of all selected spectral bands at original resolution:

        P(x, y) = (1 / B) * sum_{b=1}^{B} I_b(x, y)

    Parameters:
      raster: LoadedRaster object.
      selected_bands: List of 0-based band indices to average. If None, averages all bands.

    Returns:
      pan_float: 2D float32 array (height, width) of unscaled average intensities.
      valid_mask: 2D bool array (height, width) indicating valid (non-nodata) pixels.
      pan_display_8bit: 2D uint8 array (height, width) linearly stretched for display.
    """
    if selected_bands is None:
        selected_bands = list(range(raster.count))

    if not selected_bands:
        raise ValueError("At least one spectral band must be selected to generate a panchromatic image.")

    for b in selected_bands:
        if b < 0 or b >= raster.count:
            raise IndexError(f"Band index {b} is out of bounds for raster with {raster.count} bands.")

    num_selected = len(selected_bands)
    h, w = raster.height, raster.width

    # Allocate single float32 accumulator at full resolution to minimize memory usage
    pan_sum = np.zeros((h, w), dtype=np.float32)
    combined_valid = np.ones((h, w), dtype=bool)

    for b_idx in selected_bands:
        band_data, band_mask = raster.read_band(b_idx)
        combined_valid &= band_mask
        pan_sum += np.where(band_mask, band_data, 0.0).astype(np.float32)
        del band_data, band_mask  # Immediately free single band memory

    pan_float = np.zeros((h, w), dtype=np.float32)
    if np.any(combined_valid):
        pan_float[combined_valid] = pan_sum[combined_valid] / float(num_selected)

    # Mask out invalid pixels
    pan_float[~combined_valid] = 0.0

    # Linearly scale valid pixels to 8-bit [0, 255] for display and histogram equalization
    pan_display_8bit = normalize_to_8bit(pan_float, valid_mask=combined_valid)

    return pan_float, combined_valid, pan_display_8bit


def accumulate_panchromatic_histogram_tiled(
    raster: LoadedRaster,
    selected_bands: Optional[List[int]] = None,
    tile_size: int = 1024,
) -> Tuple[np.ndarray, int, float, np.ndarray]:
    """
    Build an 8-bit panchromatic histogram by Rasterio windows (no full-scene array).

    Returns
    -------
    hist_8bit : length-256 counts of linearly stretched pan (global min/max pass)
    n_valid : valid pixel count
    scene_mean : mean pan intensity in native units
    lut_params : array [vmin, vmax] used for the 8-bit stretch
    """
    if selected_bands is None:
        selected_bands = list(range(raster.count))
    if not selected_bands:
        raise ValueError("At least one spectral band must be selected.")

    h, w = raster.height, raster.width
    vmin = np.inf
    vmax = -np.inf
    n_valid = 0
    sum_vals = 0.0

    # Pass 1: global min/max/mean of the band-average
    for y0 in range(0, h, tile_size):
        th = min(tile_size, h - y0)
        for x0 in range(0, w, tile_size):
            tw = min(tile_size, w - x0)
            acc = np.zeros((th, tw), dtype=np.float32)
            mask = np.ones((th, tw), dtype=bool)
            for b in selected_bands:
                data, m = raster.read_window(b, x0, y0, tw, th)
                mask &= m
                acc += np.where(m, data, 0.0)
            if not np.any(mask):
                continue
            pan = acc / float(len(selected_bands))
            vals = pan[mask]
            vmin = min(vmin, float(vals.min()))
            vmax = max(vmax, float(vals.max()))
            n_valid += int(vals.size)
            sum_vals += float(vals.sum())

    if n_valid == 0 or not np.isfinite(vmin):
        empty = np.zeros(256, dtype=np.int64)
        return empty, 0, 0.0, np.array([0.0, 1.0], dtype=np.float64)

    hist = np.zeros(256, dtype=np.int64)
    # Pass 2: 8-bit histogram with the global stretch
    for y0 in range(0, h, tile_size):
        th = min(tile_size, h - y0)
        for x0 in range(0, w, tile_size):
            tw = min(tile_size, w - x0)
            acc = np.zeros((th, tw), dtype=np.float32)
            mask = np.ones((th, tw), dtype=bool)
            for b in selected_bands:
                data, m = raster.read_window(b, x0, y0, tw, th)
                mask &= m
                acc += np.where(m, data, 0.0)
            if not np.any(mask):
                continue
            pan = acc / float(len(selected_bands))
            if np.isclose(vmin, vmax, atol=1e-8):
                scaled = np.full(pan.shape, 128, dtype=np.uint8)
            else:
                scaled = np.clip(np.round((pan - vmin) * (255.0 / (vmax - vmin))), 0, 255).astype(np.uint8)
            hist += np.bincount(scaled[mask].ravel(), minlength=256)[:256]

    scene_mean = sum_vals / float(n_valid)
    return hist, n_valid, scene_mean, np.array([vmin, vmax], dtype=np.float64)


def equalization_lut_from_histogram(hist_before: np.ndarray) -> np.ndarray:
    """Standard 8-bit histogram-equalization LUT from a 256-bin histogram of valid pixels."""
    hist = np.asarray(hist_before, dtype=np.int64).ravel()[:256]
    n_valid = int(hist.sum())
    lut = np.arange(256, dtype=np.uint8)
    if n_valid <= 0:
        return lut
    cdf = hist.cumsum()
    nonzero = cdf[cdf > 0]
    if len(nonzero) <= 1:
        return lut
    cdf_min = int(nonzero[0])
    cdf_range = n_valid - cdf_min
    if cdf_range <= 0:
        return lut
    lut_vals = np.round((cdf - cdf_min) * (255.0 / cdf_range))
    return np.clip(lut_vals, 0, 255).astype(np.uint8)


def median_from_histogram(hist: np.ndarray) -> float:
    """Approximate the median gray level of a 256-bin histogram."""
    hist = np.asarray(hist, dtype=np.int64).ravel()
    total = int(hist.sum())
    if total <= 0:
        return 0.0
    cdf = hist.cumsum()
    return float(np.mean(np.searchsorted(cdf, [(total + 1) // 2, total // 2 + 1])))


def pan_and_equalized_window(
    raster: LoadedRaster,
    x: int,
    y: int,
    width: int,
    height: int,
    selected_bands: List[int],
    stretch_vmin: float,
    stretch_vmax: float,
    eq_lut: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Windowed panchromatic patch: native pan, 8-bit stretched pan, optional equalized
    8-bit, and valid mask. Uses Rasterio windows only.
    """
    acc = np.zeros((height, width), dtype=np.float32)
    mask = np.ones((height, width), dtype=bool)
    for b in selected_bands:
        data, m = raster.read_window(b, x, y, width, height)
        mask &= m
        acc += np.where(m, data, 0.0)
    pan = np.zeros((height, width), dtype=np.float32)
    if np.any(mask):
        pan[mask] = acc[mask] / float(len(selected_bands))
    if np.isclose(stretch_vmin, stretch_vmax, atol=1e-8):
        pan_8 = np.zeros((height, width), dtype=np.uint8)
        pan_8[mask] = 128
    else:
        scaled = (pan - stretch_vmin) * (255.0 / (stretch_vmax - stretch_vmin))
        pan_8 = np.zeros((height, width), dtype=np.uint8)
        pan_8[mask] = np.clip(np.round(scaled[mask]), 0, 255).astype(np.uint8)
    if eq_lut is None:
        eq_8 = pan_8.copy()
    else:
        eq_8 = eq_lut[pan_8]
        eq_8[~mask] = 0
    return pan, pan_8, eq_8, mask


def rgb_window(
    raster: LoadedRaster,
    x: int,
    y: int,
    width: int,
    height: int,
    r_idx: int = 0,
    g_idx: int = 1,
    b_idx: int = 2,
) -> np.ndarray:
    """Read a True-Colour uint8 window (percentile stretch is NOT applied)."""
    bands = []
    for idx in (r_idx, g_idx, b_idx):
        data, _ = raster.read_window(min(idx, raster.count - 1), x, y, width, height)
        bands.append(np.clip(np.round(data), 0, 255).astype(np.uint8))
    return np.stack(bands, axis=-1)


def normalize_to_8bit(
    image: np.ndarray,
    valid_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Linearly scale a 2D float or integer image to 8-bit range [0, 255] uint8,
    considering valid pixels only.
    Handles constant images and edge cases safely without division by zero.
    Invalid pixels are assigned 0.
    """
    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape[:2]

    if valid_mask is None:
        valid_mask = np.isfinite(img)
    else:
        if valid_mask.shape != img.shape:
            raise ValueError(
                f"valid_mask shape {valid_mask.shape} does not match image shape {img.shape}. "
                "Pass valid_mask=None to ignore masking."
            )
        else:
            valid_mask = valid_mask & np.isfinite(img)

    out = np.zeros((h, w), dtype=np.uint8)
    valid_vals = img[valid_mask]

    if len(valid_vals) == 0:
        return out

    v_min = float(np.min(valid_vals))
    v_max = float(np.max(valid_vals))

    if np.isclose(v_min, v_max, atol=1e-8):
        # Constant valid image: map all valid pixels to mid-gray (128)
        out[valid_mask] = 128
        return out

    # Linear min-max stretch to 0..255
    scaled = (valid_vals - v_min) * (255.0 / (v_max - v_min))
    out[valid_mask] = np.clip(np.round(scaled), 0, 255).astype(np.uint8)

    return out


def histogram_equalization(
    pan_image: np.ndarray,
    valid_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply global histogram equalization to the full-resolution panchromatic image.

    Requirements satisfied:
      - Converts float representation into appropriate 8-bit grayscale image.
      - Normalizes intensities using valid pixels only.
      - Excludes nodata/invalid pixels from histogram and CDF computation.
      - Preserves full original spatial dimensions.
      - Handles constant images and empty valid regions safely.
      - Generates before-and-after histograms for inspection.

    Returns:
      equalized_8bit: 2D uint8 equalized image at full original resolution.
      hist_before: 1D int64 array (length 256) of original 8-bit histogram.
      hist_after: 1D int64 array (length 256) of equalized histogram.
      bins: 1D int32 array of bin indices 0..255.
    """
    if pan_image is None or pan_image.size == 0:
        empty = np.zeros((0, 0), dtype=np.uint8)
        h0 = np.zeros(256, dtype=np.int64)
        return empty, h0, h0, np.arange(256, dtype=np.int32)

    h, w = pan_image.shape[:2]

    # Establish valid mask
    if valid_mask is None:
        valid_mask = np.isfinite(pan_image)
    else:
        if valid_mask.shape != pan_image.shape[:2]:
            valid_mask = np.isfinite(pan_image)
        else:
            valid_mask = valid_mask & np.isfinite(pan_image)

    # Convert input to 8-bit representation
    if pan_image.dtype == np.uint8 and np.all(valid_mask):
        image_8bit = pan_image.copy()
    else:
        image_8bit = normalize_to_8bit(pan_image, valid_mask=valid_mask)

    bins = np.arange(256, dtype=np.int32)
    equalized_8bit = np.zeros((h, w), dtype=np.uint8)

    # Fast path if all pixels are valid
    all_valid = bool(np.all(valid_mask))
    if all_valid:
        equalized_8bit = cv2.equalizeHist(image_8bit)
        hist_before = np.bincount(image_8bit.ravel(), minlength=256)[:256].astype(np.int64)
        hist_after = np.bincount(equalized_8bit.ravel(), minlength=256)[:256].astype(np.int64)
        return equalized_8bit, hist_before, hist_after, bins

    valid_pixels = image_8bit[valid_mask]
    n_valid = len(valid_pixels)

    if n_valid == 0:
        hist_before = np.zeros(256, dtype=np.int64)
        hist_after = np.zeros(256, dtype=np.int64)
        return equalized_8bit, hist_before, hist_after, bins

    # Fast histogram on valid pixels only
    hist_before = np.bincount(valid_pixels, minlength=256)[:256].astype(np.int64)

    # Cumulative distribution function (CDF)
    cdf = hist_before.cumsum()
    nonzero_cdf = cdf[cdf > 0]

    if len(nonzero_cdf) <= 1:
        # Constant valid image: preserve 8-bit values
        equalized_8bit[valid_mask] = valid_pixels
        hist_after = hist_before.copy()
        return equalized_8bit, hist_before, hist_after, bins

    cdf_min = int(nonzero_cdf[0])
    cdf_range = n_valid - cdf_min

    if cdf_range <= 0:
        lut = np.arange(256, dtype=np.uint8)
    else:
        # Standard equalization LUT: round( (cdf - cdf_min) / (N - cdf_min) * 255 )
        lut_vals = np.round((cdf - cdf_min) * (255.0 / cdf_range))
        lut = np.clip(lut_vals, 0, 255).astype(np.uint8)

    # Direct LUT indexing
    equalized_8bit = lut[image_8bit]
    equalized_8bit[~valid_mask] = 0
    hist_after = np.bincount(equalized_8bit[valid_mask], minlength=256)[:256].astype(np.int64)

    return equalized_8bit, hist_before, hist_after, bins


def get_image_statistics(
    image: Optional[np.ndarray],
    valid_mask: Optional[np.ndarray] = None
) -> dict:
    """Compute summary statistics for an image considering valid pixels only."""
    if image is None:
        return {
            "total_pixels": 0,
            "valid_pixels": 0,
            "invalid_pixels": 0,
            "valid_percentage": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0
        }

    img = np.asarray(image)
    if img.size == 0 or not np.issubdtype(img.dtype, np.number):
        return {
            "total_pixels": int(img.size),
            "valid_pixels": 0,
            "invalid_pixels": int(img.size),
            "valid_percentage": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0
        }

    if valid_mask is None:
        valid_mask = np.isfinite(img)
    else:
        if valid_mask.shape != img.shape:
            valid_mask = np.isfinite(img)
        else:
            valid_mask = valid_mask & np.isfinite(img)

    vals = img[valid_mask]
    total_pixels = img.size
    valid_count = len(vals)

    if valid_count == 0:
        return {
            "total_pixels": int(total_pixels),
            "valid_pixels": 0,
            "invalid_pixels": int(total_pixels),
            "valid_percentage": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0
        }

    # For massive rasters (> 500,000 pixels), use regular decimation for median to maintain sub-second speed
    if valid_count > 500_000:
        step = max(1, valid_count // 100_000)
        median_val = float(np.median(vals[::step]))
    else:
        median_val = float(np.median(vals))

    return {
        "total_pixels": int(total_pixels),
        "valid_pixels": int(valid_count),
        "invalid_pixels": int(total_pixels - valid_count),
        "valid_percentage": float((valid_count / total_pixels) * 100.0),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": median_val
    }
