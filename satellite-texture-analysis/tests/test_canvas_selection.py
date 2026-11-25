# -*- coding: utf-8 -*-
"""
Tests for the canvas_selection module.
"""

import numpy as np
import pytest
from modules.canvas_selection import (
    prepare_canvas_image,
    canvas_rect_to_original,
    compute_selection_preview,
    LAND_COVER_CATEGORIES,
)


def test_prepare_canvas_image_scaling():
    # Create a 2000x1000 grayscale image
    img = np.arange(2000 * 1000, dtype=np.uint8).reshape((1000, 2000))
    canvas_img, scale_x, scale_y, canvas_w, canvas_h = prepare_canvas_image(img, max_dim=500)
    # Ensure scaling respects max_dim on larger side (2000 -> 500)
    assert canvas_w == 500
    # Height scaled proportionally
    expected_h = int(round(1000 * (500 / 2000)))
    assert canvas_h == expected_h
    # Scale factors
    assert pytest.approx(scale_x, rel=1e-3) == canvas_w / img.shape[1]
    assert pytest.approx(scale_y, rel=1e-3) == canvas_h / img.shape[0]
    # Canvas image should be RGB uint8
    assert canvas_img.shape == (canvas_h, canvas_w, 3)
    assert canvas_img.dtype == np.uint8


def test_canvas_rect_to_original_square_and_clamp():
    # Assume original image 1000x800
    orig_w, orig_h = 1000, 800
    # Canvas scaled to 250x200 (scale 0.25)
    scale_x, scale_y = 0.25, 0.25
    # Rectangle drawn: left=50, top=40, width=100, height=80 (canvas pixels)
    rect = {"left": 50, "top": 40, "width": 100, "height": 80}
    x, y, w, h, valid, msg = canvas_rect_to_original(rect, scale_x, scale_y, orig_w, orig_h, min_size=16)
    # Map back to original: divide by scale
    expected_x = int(50 / scale_x)
    expected_y = int(40 / scale_y)
    expected_w = int(100 / scale_x)
    expected_h = int(80 / scale_y)
    # Enforced square using min dimension
    side = min(expected_w, expected_h)
    assert (w, h) == (side, side)
    assert (x, y) == (expected_x, expected_y)
    assert valid is True

    # Test too small selection
    tiny_rect = {"left": 0, "top": 0, "width": 4, "height": 4}
    x2, y2, w2, h2, valid2, msg2 = canvas_rect_to_original(tiny_rect, scale_x, scale_y, orig_w, orig_h, min_size=16)
    assert valid2 is False
    assert "minimum" in msg2.lower()

def test_compute_selection_preview_thumbnail():
    # Create a simple image 100x100 with a gradient
    img = np.arange(100 * 100, dtype=np.uint8).reshape((100, 100))
    # Extract a region 40x30 at (10,20)
    thumb = compute_selection_preview(img, 10, 20, 40, 30, max_preview=20)
    # Thumbnail max side should be <=20
    assert max(thumb.shape) <= 20
    # Ensure data type
    assert thumb.dtype == np.uint8
    # Verify that region content matches original when not downscaled
    # Since max preview is smaller, expect downscale; just check shape consistency
    assert thumb.shape[0] > 0 and thumb.shape[1] > 0

def test_land_cover_categories_defined():
    # Ensure exactly four categories as required
    assert len(LAND_COVER_CATEGORIES) == 4
    expected = ["Water Bodies", "Vegetation", "Urban / Built-up Area", "Bare Soil"]
    assert LAND_COVER_CATEGORIES == expected
