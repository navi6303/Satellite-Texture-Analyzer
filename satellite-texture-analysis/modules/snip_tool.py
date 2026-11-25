"""
Snipping-Tool Module — Bidirectional Custom Streamlit Component
==============================================================
Uses ``streamlit.components.v1.declare_component`` so that JavaScript
returns selection data to Python automatically — no text-box pasting required.

Workflow from Python's perspective:
  1. Call ``snip_canvas(...)`` with the display image (data-URL), dimensions,
     and the active category.
  2. The JS component renders a drag-to-select canvas with a Confirm button.
  3. When the user clicks Confirm, JS calls ``Streamlit.setComponentValue(payload)``
     which immediately makes the payload available as the return value of
     ``snip_canvas(...)`` in the next Python run.
  4. The payload contains ``{x, y, size, confirmed}`` already mapped to the
     original raster coordinate grid (mapping done inside JS).
  5. Python only needs to validate, store, and extract.

Performance notes
-----------------
* The FCC image is encoded once and cached via ``st.cache_data`` keyed on the
  raster file signature + selected display bands.
* Mouse movement events are handled entirely inside the iframe — no Python
  reruns occur during dragging.
* Patch extraction uses NumPy array slicing on the already-loaded equalized
  image; Rasterio windowed-reads are used for new uploads.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components


# ---------------------------------------------------------------------------
# Land-cover category constants
# ---------------------------------------------------------------------------
LAND_COVER_CATEGORIES: List[str] = [
    "Water Bodies",
    "Vegetation",
    "Urban / Built-up Area",
    "Bare Soil",
]

CATEGORY_COLORS: Dict[str, str] = {
    "Water Bodies": "#00b4d8",
    "Vegetation": "#2dc653",
    "Urban / Built-up Area": "#e63946",
    "Bare Soil": "#fca311",
}

# ---------------------------------------------------------------------------
# Declare the bidirectional component (served from local HTML file)
# ---------------------------------------------------------------------------
_COMPONENT_DIR = Path(__file__).parent.parent / "components" / "snip_canvas"

_snip_canvas = components.declare_component(
    "snip_canvas",
    path=str(_COMPONENT_DIR),
)

# ---------------------------------------------------------------------------
# Image encoding helpers (cached)
# ---------------------------------------------------------------------------


def _encode_png_data_url(img_rgb: np.ndarray) -> str:
    """Encode an RGB uint8 NumPy array as a PNG data-URL string."""
    ok, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    if not ok:
        raise RuntimeError("Failed to encode image for canvas display.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _annotate_selections(
    img_rgb: np.ndarray,
    selections: Dict[str, Dict[str, int]],
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """
    Draw confirmed selections (from ORIGINAL coordinates) onto the display image.
    The display image may already be downscaled; we scale the boxes accordingly.
    """
    if not selections:
        return img_rgb
    out = img_rgb.copy()
    dh, dw = out.shape[:2]
    sx = dw / orig_w
    sy = dh / orig_h
    for cat, sel in selections.items():
        boxes = sel if isinstance(sel, list) else [sel]
        hex_col = CATEGORY_COLORS.get(cat, "#ffffff")
        r = int(hex_col[1:3], 16)
        g = int(hex_col[3:5], 16)
        b = int(hex_col[5:7], 16)
        for k, box in enumerate(boxes):
            if not box:
                continue
            cx = int(round(int(box["x"]) * sx))
            cy = int(round(int(box["y"]) * sy))
            width = round(box.get("w", box.get("size", 0)) * sx)
            height = round(box.get("h", box.get("size", 0)) * sy)
            cv2.rectangle(
                out, (cx, cy), (cx + width, cy + height), (r, g, b), 2 if k else 3
            )
            if k == 0:
                idx = LAND_COVER_CATEGORIES.index(cat) + 1
                label = f"#{idx} {cat} n={len(boxes)}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(label, font, 0.46, 1)
                cv2.rectangle(
                    out, (cx, max(0, cy - th - 7)), (cx + tw + 8, cy), (r, g, b), -1
                )
                cv2.putText(
                    out,
                    label,
                    (cx + 4, max(th + 2, cy - 4)),
                    font,
                    0.46,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
    return out


@st.cache_data(show_spinner=False, max_entries=4)
def _build_display_image(
    file_sig: str,  # used only as cache key
    fcc_rgb_bytes: bytes,
    orig_w: int,
    orig_h: int,
    canvas_max_dim: int,
) -> Tuple[np.ndarray, float, int, int]:
    """
    Downscale the FCC image to fit within canvas_max_dim.
    Cached so it is NOT regenerated on every Streamlit rerun.

    Returns (display_rgb, scale, cw, ch).
    """
    img = np.frombuffer(fcc_rgb_bytes, dtype=np.uint8).reshape((orig_h, orig_w, 3))
    scale = min(1.0, canvas_max_dim / max(orig_w, orig_h))
    cw = max(1, int(round(orig_w * scale)))
    ch = max(1, int(round(orig_h * scale)))
    if scale < 1.0:
        disp = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
    else:
        disp = img.copy()
    return disp, scale, cw, ch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def snip_canvas(
    fcc_rgb: np.ndarray,
    orig_w: int,
    orig_h: int,
    category: str,
    confirmed_selections: Dict[str, Any],
    file_sig: str = "",
    canvas_max_dim: int = 900,
    category_colors: Optional[Dict[str, str]] = None,
    key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Render the bidirectional snipping-tool canvas and return a selection dict
    automatically when the user clicks Confirm.

    Parameters
    ----------
    fcc_rgb              : Full-resolution (or display-ready) RGB uint8 array.
    orig_w / orig_h      : Native raster dimensions.
    category             : Active land-cover category name.
    confirmed_selections : Confirmed native rectangles by class, each with x, y, w and h.
    file_sig             : Cache key (raster filename + size).
    canvas_max_dim       : Maximum canvas side in pixels.
    category_colors      : Optional mapping from class names to hex colors.
    key                  : Unique Streamlit widget key.

    Returns
    -------
    Dict ``{x, y, w, h, confirmed: True}`` in ORIGINAL raster pixels, or None.
    The JS component maps canvas→original internally before sending the value.
    """
    colors = dict(CATEGORY_COLORS)
    if category_colors:
        colors.update(category_colors)

    # Build display image (cached)
    rgb_bytes = fcc_rgb.tobytes()
    disp, scale, cw, ch = _build_display_image(
        file_sig, rgb_bytes, fcc_rgb.shape[1], fcc_rgb.shape[0], canvas_max_dim
    )

    # The interactive canvas draws class overlays in native coordinates.
    data_url = _encode_png_data_url(disp)

    # Format confirmed selections for JS component (ensure serializable)
    clean_selections = {}
    if confirmed_selections:
        for cat_k, boxes in confirmed_selections.items():
            if not boxes:
                continue
            box_list = boxes if isinstance(boxes, list) else [boxes]
            clean_selections[cat_k] = [
                {"x": int(b["x"]), "y": int(b["y"]), "w": int(b.get("w", b.get("size", 0))), "h": int(b.get("h", b.get("size", 0)))}
                for b in box_list if b
            ]

    # Call the bidirectional component
    result = _snip_canvas(
        image_data_url=data_url,
        orig_w=orig_w,
        orig_h=orig_h,
        category=category,
        color=colors.get(category, "#00b4d8"),
        confirmed_selections=clean_selections,
        category_colors=colors,
        key=key or f"snip_{category.replace(' ', '_').replace('/', '_')}",
        default=None,
    )

    # Validate and return
    if result and isinstance(result, dict) and result.get("confirmed"):
        return result
    return None


def decode_selection(sel, orig_w, orig_h, min_size=8):
    """Validate the actual native-pixel footprint returned by the canvas."""
    from modules.texture_extraction import validate_crop

    box = dict(
        x=int(sel["x"]),
        y=int(sel["y"]),
        w=int(sel.get("w", sel.get("size", 0))),
        h=int(sel.get("h", sel.get("size", 0))),
    )
    ok, msg = validate_crop(
        box["x"], box["y"], box["w"], box["h"], orig_w, orig_h, min_size=min_size
    )
    return box, ok, msg
