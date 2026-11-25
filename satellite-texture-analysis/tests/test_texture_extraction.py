"""
Tests for Texture Extraction and Quantization Module.
Validates:
  - Bounding box coordinate validation and boundary enforcement
  - Strict 5-level quantization equation: Q(x, y) = floor( 5 * I(x, y) / 256 )
  - Exact boundary thresholds (0, 51, 102, 153, 204, 255)
  - Guarantee that 255 maps to 4 and never 5
  - TexturePatch extraction and metadata packaging
"""

import numpy as np
import pytest

from modules.texture_extraction import (
    TexturePatch,
    draw_bounding_boxes_preview,
    extract_patch,
    get_default_quadrant_crops,
    quantize_to_5_levels,
    validate_crop
)


def test_validate_crop_boundaries():
    """Verify crop coordinate boundary checks."""
    img_w, img_h = 256, 256

    # Valid crop
    ok, msg = validate_crop(x=10, y=10, width=50, height=50, img_width=img_w, img_height=img_h)
    assert ok is True

    # Negative coordinates
    ok, msg = validate_crop(x=-5, y=10, width=50, height=50, img_width=img_w, img_height=img_h)
    assert ok is False
    assert "negative" in msg

    # Smaller than min_size
    ok, msg = validate_crop(x=0, y=0, width=4, height=50, img_width=img_w, img_height=img_h, min_size=8)
    assert ok is False
    assert "smaller than minimum" in msg

    # Exceeding width boundary
    ok, msg = validate_crop(x=200, y=0, width=60, height=50, img_width=img_w, img_height=img_h)
    assert ok is False
    assert "exceeds" in msg and "width" in msg

    # Exceeding height boundary
    ok, msg = validate_crop(x=0, y=210, width=50, height=50, img_width=img_w, img_height=img_h)
    assert ok is False
    assert "exceeds" in msg and "height" in msg


def test_quantize_to_5_levels_exact_boundaries():
    """
    Test 5-level reduction on specific transition boundaries:
    Equation: Q(x, y) = floor( 5 * I(x, y) / 256 )
    """
    # Key boundary values
    test_inputs = np.array([0, 51, 52, 102, 103, 153, 154, 204, 205, 255], dtype=np.uint8)
    quantized, unique_levels = quantize_to_5_levels(test_inputs)

    assert quantized[0] == 0   # 5 * 0 // 256 = 0
    assert quantized[1] == 0   # 5 * 51 // 256 = 255 // 256 = 0
    assert quantized[2] == 1   # 5 * 52 // 256 = 260 // 256 = 1
    assert quantized[3] == 1   # 5 * 102 // 256 = 510 // 256 = 1
    assert quantized[4] == 2   # 5 * 103 // 256 = 515 // 256 = 2
    assert quantized[5] == 2   # 5 * 153 // 256 = 765 // 256 = 2
    assert quantized[6] == 3   # 5 * 154 // 256 = 770 // 256 = 3
    assert quantized[7] == 3   # 5 * 204 // 256 = 1020 // 256 = 3
    assert quantized[8] == 4   # 5 * 205 // 256 = 1025 // 256 = 4
    assert quantized[9] == 4   # 5 * 255 // 256 = 1275 // 256 = 4 -> GUARANTEE 255 maps to 4, never 5!

    assert set(unique_levels) == {0, 1, 2, 3, 4}


def test_quantize_all_256_intensities():
    """Verify that every integer from 0 to 255 maps strictly to {0, 1, 2, 3, 4}."""
    all_vals = np.arange(256, dtype=np.uint8)
    q_all, _ = quantize_to_5_levels(all_vals)

    # Check minimum and maximum
    assert int(np.min(q_all)) == 0
    assert int(np.max(q_all)) == 4
    # Check that no other values exist
    assert set(np.unique(q_all)) == {0, 1, 2, 3, 4}


def test_extract_patch_success_and_dataclass():
    """Verify patch extraction creates a valid TexturePatch container."""
    img = np.random.RandomState(42).randint(0, 256, size=(100, 100), dtype=np.uint8)
    patch = extract_patch(
        image_8bit=img,
        x=20,
        y=30,
        width=40,
        height=40,
        label="Test Vegetation",
        patch_id="patch_1"
    )

    assert isinstance(patch, TexturePatch)
    assert patch.patch_id == "patch_1"
    assert patch.label == "Test Vegetation"
    assert patch.x == 20
    assert patch.y == 30
    assert patch.width == 40
    assert patch.height == 40
    assert patch.image_8bit.shape == (40, 40)
    assert patch.quantized.shape == (40, 40)
    assert patch.total_pixels == 1600
    assert patch.valid_pixel_count == 1600
    assert set(patch.unique_levels).issubset({0, 1, 2, 3, 4})


def test_extract_patch_invalid_raises():
    """Verify extract_patch raises ValueError on invalid parameters."""
    img = np.zeros((50, 50), dtype=np.uint8)
    # Out of bounds
    with pytest.raises(ValueError):
        extract_patch(img, x=40, y=40, width=20, height=20)

    # All-masked patch
    mask = np.zeros((50, 50), dtype=bool)
    with pytest.raises(ValueError):
        extract_patch(img, x=0, y=0, width=20, height=20, valid_mask=mask)


def test_default_quadrant_crops():
    """Verify default crop generator provides four non-overlapping quadrant crops."""
    crops = get_default_quadrant_crops(img_width=256, img_height=256)
    assert len(crops) == 4
    labels = [c["label"] for c in crops]
    assert "Water" in labels[0]
    assert "Vegetation" in labels[1]
    assert "Agricultural" in labels[2]
    assert "Urban" in labels[3]

    # Verify all crops fit inside 256x256
    for c in crops:
        ok, msg = validate_crop(c["x"], c["y"], c["w"], c["h"], 256, 256)
        assert ok is True, msg


def test_draw_bounding_boxes():
    """Verify bounding box visualization produces valid RGB canvas."""
    img = np.zeros((100, 100), dtype=np.uint8)
    crops = [
        {"id": "p1", "label": "Class 1", "x": 10, "y": 10, "w": 20, "h": 20},
        {"id": "p2", "label": "Class 2", "x": 60, "y": 60, "w": 20, "h": 20}
    ]
    preview, info = draw_bounding_boxes_preview(img, crops)
    assert preview.shape == (100, 100, 3)
    assert preview.dtype == np.uint8
    assert isinstance(info, dict)
    # Canvas should have colored pixels drawn
    assert np.any(preview > 0)
