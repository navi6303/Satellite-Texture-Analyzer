"""
Canvas Selection Module for Interactive Texture Region Selection.

Provides utilities for:
  - Preparing a down-scaled RGB canvas image for display in streamlit-drawable-canvas.
  - Transforming canvas-space rectangle coordinates back to original image coordinates.
  - Drawing existing selections on the canvas background.
  - Generating small preview thumbnails of extracted regions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Land-cover category constants
# ---------------------------------------------------------------------------

LAND_COVER_CATEGORIES: List[str] = [
    "Water Bodies",
    "Vegetation",
    "Urban / Built-up Area",
    "Bare Soil",
]

# Color look-up for each category.
# "hex"  – CSS/HTML hex string used as stroke_color in st_canvas.
# "bgr"  – OpenCV BGR tuple used for cv2.rectangle() calls.
CATEGORY_COLORS: Dict[str, Dict[str, Any]] = {
    "Water Bodies":          {"hex": "#00b4d8", "bgr": (216, 180,   0)},
    "Vegetation":            {"hex": "#2dc653", "bgr": ( 83, 198,  45)},
    "Urban / Built-up Area": {"hex": "#e63946", "bgr": ( 70,  57, 230)},
    "Bare Soil":             {"hex": "#fca311", "bgr": ( 17, 163, 252)},
}

_DEFAULT_COLOR: Dict[str, Any] = {"hex": "#888888", "bgr": (128, 128, 128)}


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------

def get_category_color(category: str) -> Dict[str, Any]:
    """Return the colour dict for *category*, or a neutral grey if unknown."""
    return CATEGORY_COLORS.get(category, _DEFAULT_COLOR)


def prepare_canvas_image(
    equalized_8bit: np.ndarray,
    existing_selections: Optional[List[Dict[str, Any]]] = None,
    max_dim: int = 900,
) -> Tuple[np.ndarray, float, float, int, int]:
    """
    Convert a full-resolution 2-D grayscale uint8 array to a down-scaled
    RGB canvas image, optionally annotating it with *existing_selections*.

    Parameters
    ----------
    equalized_8bit : np.ndarray
        Full-resolution 2-D grayscale array, dtype uint8, shape (H, W).
    existing_selections : list of dicts, optional
        Each dict must have keys: "x", "y", "w", "h", "category", "index".
        Rectangles are drawn in the category's BGR colour.
    max_dim : int
        Maximum side length (pixels) of the canvas image.  If the image is
        already smaller, it is not upscaled.

    Returns
    -------
    canvas_img_rgb : np.ndarray
        uint8 RGB array of shape (canvas_h, canvas_w, 3).
    scale_x : float
        canvas_w / orig_w
    scale_y : float
        canvas_h / orig_h
    canvas_w : int
    canvas_h : int
    """
    orig_h, orig_w = equalized_8bit.shape[:2]

    # Compute scale so the longest side does not exceed max_dim
    scale = min(1.0, float(max_dim) / float(max(orig_h, orig_w)))
    canvas_w = max(1, int(round(orig_w * scale)))
    canvas_h = max(1, int(round(orig_h * scale)))

    # Down-scale using INTER_AREA (best for shrinking)
    if scale < 1.0:
        resized = cv2.resize(equalized_8bit, (canvas_w, canvas_h), interpolation=cv2.INTER_AREA)
    else:
        resized = equalized_8bit.copy()

    # Convert grayscale → RGB
    canvas_img_rgb = cv2.cvtColor(resized.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    scale_x = canvas_w / orig_w
    scale_y = canvas_h / orig_h

    # Annotate existing selections
    if existing_selections:
        for sel in existing_selections:
            cat   = sel.get("category", "")
            color = get_category_color(cat)["bgr"]

            # Scale original coords → canvas coords
            cx  = int(round(sel["x"] * scale_x))
            cy  = int(round(sel["y"] * scale_y))
            cw  = max(2, int(round(sel["w"] * scale_x)))
            ch  = max(2, int(round(sel["h"] * scale_y)))

            cv2.rectangle(
                canvas_img_rgb,
                (cx, cy),
                (cx + cw, cy + ch),
                color,
                2,
            )

            # Draw a small label
            label = f"#{sel.get('index', 0) + 1} {cat}"
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)

            lx1 = cx
            ly1 = max(0, cy - th - 6)
            ly2 = cy
            cv2.rectangle(canvas_img_rgb, (lx1, ly1), (min(canvas_w - 1, lx1 + tw + 6), ly2), color, -1)
            cv2.putText(
                canvas_img_rgb,
                label,
                (lx1 + 3, max(th + 2, cy - 3)),
                font,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return canvas_img_rgb, scale_x, scale_y, canvas_w, canvas_h


def canvas_rect_to_original(
    rect_data: Dict[str, Any],
    scale_x: float,
    scale_y: float,
    orig_w: int,
    orig_h: int,
    min_size: int = 16,
) -> Tuple[int, int, int, int, bool, str]:
    """
    Map a rectangle from streamlit-drawable-canvas JSON coordinates back to
    original image pixel coordinates.

    The canvas result JSON object uses keys "left", "top", "width", "height".
    The result is forced to a **square** (side = min(w_orig, h_orig)) and
    clamped to the image boundaries.

    Parameters
    ----------
    rect_data : dict
        One object entry from ``canvas_result.json_data["objects"]``.
    scale_x, scale_y : float
        Returned by :func:`prepare_canvas_image`.
    orig_w, orig_h : int
        Native image dimensions.
    min_size : int
        Minimum side length (pixels) in original coordinates.

    Returns
    -------
    x, y, w, h : int
        Top-left corner and size in original image coordinates.
    is_valid : bool
    msg : str
        Human-readable status message.
    """
    # Guard against missing keys gracefully
    left   = float(rect_data.get("left",   rect_data.get("x", 0)))
    top    = float(rect_data.get("top",    rect_data.get("y", 0)))
    width  = float(rect_data.get("width",  0))
    height = float(rect_data.get("height", 0))

    # Map to original coords
    orig_x = int(left   / scale_x)
    orig_y = int(top    / scale_y)
    orig_rw = int(width  / scale_x)
    orig_rh = int(height / scale_y)

    # Enforce square: use the smaller dimension
    side = min(orig_rw, orig_rh)

    # Clamp to image boundaries
    orig_x = max(0, min(orig_x, orig_w - 1))
    orig_y = max(0, min(orig_y, orig_h - 1))
    side   = min(side, orig_w - orig_x, orig_h - orig_y)

    if side <= min_size:
        return (
            orig_x, orig_y, side, side,
            False,
            f"Selection too small ({side}×{side} px); minimum is {min_size}×{min_size} px.",
        )

        return (
            orig_x, orig_y, side, side,
            False,
            f"Selection too small ({side}×{side} px); minimum is {min_size}×{min_size} px.",
        )

    return orig_x, orig_y, side, side, True, f"Valid selection: ({orig_x}, {orig_y}), {side}×{side} px."


def compute_selection_preview(
    equalized_8bit: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    max_preview: int = 256,
) -> np.ndarray:
    """
    Extract a small thumbnail of the region ``[y:y+h, x:x+w]`` for preview.

    Parameters
    ----------
    equalized_8bit : np.ndarray
        Full-resolution 2-D grayscale uint8 array.
    x, y, w, h : int
        Region in original image coordinates.
    max_preview : int
        Maximum side length of the returned thumbnail.

    Returns
    -------
    np.ndarray
        2-D uint8 grayscale thumbnail.
    """
    # Clamp slice to valid bounds
    img_h, img_w = equalized_8bit.shape[:2]
    x  = max(0, int(x))
    y  = max(0, int(y))
    w  = max(1, min(int(w), img_w - x))
    h  = max(1, min(int(h), img_h - y))

    crop = equalized_8bit[y : y + h, x : x + w]

    # Down-sample if larger than max_preview
    max_side = max(crop.shape[:2])
    if max_side > max_preview:
        pscale  = float(max_preview) / float(max_side)
        new_w   = max(1, int(round(crop.shape[1] * pscale)))
        new_h   = max(1, int(round(crop.shape[0] * pscale)))
        crop    = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return crop.astype(np.uint8)
