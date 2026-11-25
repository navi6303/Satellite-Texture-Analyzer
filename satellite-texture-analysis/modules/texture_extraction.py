"""
Texture Extraction and 5-Level Quantization Module.
Handles:
  1. Coordinate validation and bounds checking for texture sub-images in original pixel coordinates.
  2. Extraction of rectangular texture patches from the full-resolution enhanced image.
  3. Overlaying the 4 numbered, colored bounding boxes directly onto the full scene preview
     with accurate coordinate scaling.
  4. Strict 5-level gray-level reduction (0, 1, 2, 3, 4) per academic formula:
       Q(x, y) = floor( 5 * I(x, y) / 256 )
  5. Dataclass modeling for texture classes, coordinates, and extracted arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Quantization modes (sidebar labels map to these keys)
QUANT_GLOBAL_EQ_FIXED5 = "global_equalized_fixed5"
QUANT_PER_PATCH_RESCALE = "per_patch_rescale_5bin"
QUANT_RAW_PAN_FIXED5 = "raw_pan_fixed5"
QUANTIZATION_MODES = (
    QUANT_GLOBAL_EQ_FIXED5,
    QUANT_PER_PATCH_RESCALE,
    QUANT_RAW_PAN_FIXED5,
)


@dataclass
class TexturePatch:
    """Container holding an extracted texture patch and its 5-level quantized representation."""
    patch_id: str
    label: str
    x: int
    y: int
    width: int
    height: int
    image_8bit: np.ndarray  # Shape: (height, width), uint8, original extracted pixels
    quantized: np.ndarray   # Shape: (height, width), uint8, values strictly in [0, 4]
    valid_mask: np.ndarray  # Shape: (height, width), bool
    unique_levels: List[int] = field(default_factory=list)
    glcm: Optional[np.ndarray] = None        # Shape: (5, 5), float64
    eigenvalues: Optional[np.ndarray] = None  # Shape: (5,), float64
    transition_eigenvalues: Optional[np.ndarray] = None
    valid_pair_count: int = 0
    horizontal_valid_pairs: int = 0
    occupancy: Optional[np.ndarray] = None  # length-5 probabilities
    rgb_mean: Optional[np.ndarray] = None
    rgb_std: Optional[np.ndarray] = None
    pan_mean: float = 0.0
    pan_std: float = 0.0
    eq_mean: float = 0.0
    eq_std: float = 0.0
    block_id: int = 0
    sanity_warnings: List[str] = field(default_factory=list)
    classic_features: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate_footprint(self):
        """All analytical rasters must occupy exactly the confirmed native window."""
        shape = (self.height, self.width)
        layers = {"equalized image": self.image_8bit, "quantized image": self.quantized,
                  "validity mask": self.valid_mask}
        for key in ("pan_patch", "eq_patch"):
            if self.extra.get(key) is not None:
                layers[key] = self.extra[key]
        for name, data in layers.items():
            if np.shape(data) != shape:
                raise ValueError(f"{self.patch_id}: {name} shape {np.shape(data)} differs "
                                 f"from confirmed box (height, width) {shape}.")

    @property
    def total_pixels(self) -> int:
        return self.width * self.height

    @property
    def valid_pixel_count(self) -> int:
        return int(np.sum(self.valid_mask))

    @property
    def coordinates_str(self) -> str:
        return f"X: {self.x}, Y: {self.y}, W: {self.width}, H: {self.height}"


def validate_crop(
    x: int,
    y: int,
    width: int,
    height: int,
    img_width: int,
    img_height: int,
    min_size: int = 8
) -> Tuple[bool, str]:
    """
    Validate rectangular crop coordinates and dimensions in original image pixels.

    Checks:
      - x and y must be >= 0
      - width and height must be >= min_size
      - x + width <= img_width
      - y + height <= img_height
    """
    if width < min_size or height < min_size:
        return False, f"Patch dimensions ({width}×{height} px) are smaller than minimum allowed ({min_size}×{min_size} px)."

    if x < 0 or y < 0:
        return False, f"Top-left coordinates (x={x}, y={y}) cannot be negative."

    if x + width > img_width:
        return False, f"Patch horizontal span (x={x} + w={width} = {x + width}) exceeds raster width ({img_width})."

    if y + height > img_height:
        return False, f"Patch vertical span (y={y} + h={height} = {y + height}) exceeds raster height ({img_height})."

    return True, "Valid crop parameters."


def quantize_to_5_levels(
    patch_8bit: np.ndarray,
    valid_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Reduce an 8-bit image patch (intensities 0-255) to exactly five gray levels:
    0, 1, 2, 3, and 4, according to the remote-sensing formula:

        Q(x, y) = floor( 5 * I(x, y) / 256 )

    Guarantees:
      - I in [0, 51]   -> 0
      - I in [52, 102] -> 1
      - I in [103, 153]-> 2
      - I in [154, 204]-> 3
      - I in [205, 255]-> 4 (255 maps to 4, never 5!)
      - Masked / invalid pixels are preserved via mask and do not count as a 6th gray level.
    """
    patch = np.asarray(patch_8bit, dtype=np.int32)

    if valid_mask is None:
        valid_mask = np.ones(patch.shape, dtype=bool)
    elif valid_mask.shape != patch.shape:
        raise ValueError("Patch and mask shapes differ.")

    # Integer division implements mathematical floor for positive numbers
    quantized = np.clip((5 * patch) // 256, 0, 4).astype(np.uint8)

    # Unique levels on valid pixels only
    valid_pixels = quantized[valid_mask]
    if len(valid_pixels) > 0:
        unique_levels = sorted([int(lvl) for lvl in np.unique(valid_pixels)])
    else:
        unique_levels = []

    return quantized, unique_levels


def level_occupancy(
    quantized: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    levels: int = 5,
) -> np.ndarray:
    """Fraction of valid pixels in each quantized gray level 0..levels-1."""
    q = np.asarray(quantized).ravel()
    if valid_mask is None:
        vals = q
    else:
        vals = np.asarray(quantized)[np.asarray(valid_mask, dtype=bool)]
    if vals.size == 0:
        return np.zeros(levels, dtype=np.float64)
    counts = np.bincount(vals.astype(np.int64), minlength=levels)[:levels]
    total = float(counts.sum())
    if total <= 0:
        return np.zeros(levels, dtype=np.float64)
    return counts.astype(np.float64) / total


def rescale_patch_to_8bit(
    patch: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    method: str = "percentile",
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> np.ndarray:
    """Per-patch intensity rescale to uint8 [0, 255] via min–max or percentiles."""
    img = np.asarray(patch, dtype=np.float64)
    if valid_mask is None:
        valid_mask = np.ones(img.shape, dtype=bool)
    vals = img[valid_mask]
    out = np.zeros(img.shape, dtype=np.uint8)
    if vals.size == 0:
        return out
    if method == "minmax":
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
    else:
        vmin = float(np.percentile(vals, percentile_low))
        vmax = float(np.percentile(vals, percentile_high))
        if vmax <= vmin:
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
    if vmax <= vmin:
        out[valid_mask] = 128
        return out
    scaled = (img - vmin) * (255.0 / (vmax - vmin))
    out[valid_mask] = np.clip(np.round(scaled[valid_mask]), 0, 255).astype(np.uint8)
    return out


def quantize_by_mode(
    patch_8bit: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    mode: str = QUANT_GLOBAL_EQ_FIXED5,
    rescale_method: str = "percentile",
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """
    5-level quantization under one of the three analysis modes.

    (a) global_equalized_fixed5 — caller already equalized globally; fixed bins.
    (b) per_patch_rescale_5bin — min–max or percentile stretch, then fixed bins.
    (c) raw_pan_fixed5 — fixed bins on the raw panchromatic patch (no HE).
    """
    src = np.asarray(patch_8bit)
    if mode == QUANT_PER_PATCH_RESCALE:
        src = rescale_patch_to_8bit(
            src,
            valid_mask=valid_mask,
            method=rescale_method,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
        )
    if mode == QUANT_RAW_PAN_FIXED5 and np.issubdtype(src.dtype, np.floating):
        quantized = np.clip(np.floor(5 * src / 256), 0, 4).astype(np.uint8)
        unique_levels = np.unique(quantized if valid_mask is None else quantized[valid_mask]).tolist()
    else:
        quantized, unique_levels = quantize_to_5_levels(src, valid_mask=valid_mask)
    occ = level_occupancy(quantized, valid_mask=valid_mask, levels=5)
    return quantized, unique_levels, occ


def evaluate_patch_sanity(
    label: str,
    occupancy: np.ndarray,
    eq_mean: float,
    scene_median_eq: Optional[float],
    unique_levels: List[int],
) -> List[str]:
    """Visible warnings when a crop is radiometrically inconsistent with its class."""
    warnings: List[str] = []
    n_populated = int(np.sum(np.asarray(occupancy) > 0.01))
    if n_populated < 3:
        warnings.append(
            f"Fewer than 3 quantized levels hold >1% of pixels "
            f"(populated={n_populated}, unique={unique_levels})."
        )
    if (
        label == "Water Bodies"
        and scene_median_eq is not None
        and np.isfinite(eq_mean)
        and eq_mean > scene_median_eq
    ):
        warnings.append(
            f"Water patch mean equalized intensity ({eq_mean:.1f}) is above the "
            f"scene median ({scene_median_eq:.1f}). Review the footprint and class label."
        )
    return warnings


def extract_patch(
    image_8bit: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    valid_mask: Optional[np.ndarray] = None,
    label: str = "Texture Patch",
    patch_id: str = "patch_1",
    min_size: int = 8,
    quantization_mode: str = QUANT_GLOBAL_EQ_FIXED5,
    rescale_method: str = "percentile",
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    pan_patch: Optional[np.ndarray] = None,
    eq_patch: Optional[np.ndarray] = None,
    rgb_patch: Optional[np.ndarray] = None,
    scene_median_eq: Optional[float] = None,
    block_id: int = 0,
) -> TexturePatch:
    """
    Extract a sub-image patch directly from the original full-resolution 8-bit image
    and compute its 5-level quantization.
    """
    img_h, img_w = image_8bit.shape[:2]
    is_valid, msg = validate_crop(x, y, width, height, img_w, img_h, min_size=min_size)
    if not is_valid:
        raise ValueError(f"Invalid crop for {label} ({patch_id}): {msg}")

    if valid_mask is None:
        valid_mask = np.ones((img_h, img_w), dtype=bool)
    elif valid_mask.shape != (img_h, img_w):
        raise ValueError(f"{patch_id}: mask shape {valid_mask.shape} differs from the input image {(img_h, img_w)}.")

    # Slice directly from full-resolution array without interpolation or distortion
    sub_img = image_8bit[y : y + height, x : x + width].copy()
    sub_mask = valid_mask[y : y + height, x : x + width].copy()

    if not np.any(sub_mask):
        raise ValueError(f"Extracted patch '{label}' ({patch_id}) contains no valid pixels.")

    quantized, unique_levels, occ = quantize_by_mode(
        sub_img,
        valid_mask=sub_mask,
        mode=quantization_mode,
        rescale_method=rescale_method,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )

    pan_src = pan_patch if pan_patch is not None else sub_img
    eq_src = eq_patch if eq_patch is not None else sub_img
    pan_vals = np.asarray(pan_src, dtype=np.float64)[sub_mask]
    eq_vals = np.asarray(eq_src, dtype=np.float64)[sub_mask]
    pan_mean = float(np.mean(pan_vals)) if pan_vals.size else 0.0
    pan_std = float(np.std(pan_vals)) if pan_vals.size else 0.0
    eq_mean = float(np.mean(eq_vals)) if eq_vals.size else 0.0
    eq_std = float(np.std(eq_vals)) if eq_vals.size else 0.0

    rgb_mean = None
    rgb_std = None
    if rgb_patch is not None:
        rgb = np.asarray(rgb_patch, dtype=np.float64)
        if rgb.ndim == 3:
            rgb_mean = np.array([float(np.nanmean(rgb[:, :, c][sub_mask])) for c in range(min(3, rgb.shape[2]))])
            rgb_std = np.array([float(np.nanstd(rgb[:, :, c][sub_mask])) for c in range(min(3, rgb.shape[2]))])

    warnings = evaluate_patch_sanity(label, occ, eq_mean, scene_median_eq, unique_levels)

    return TexturePatch(
        patch_id=patch_id,
        label=label,
        x=int(x),
        y=int(y),
        width=int(width),
        height=int(height),
        image_8bit=np.asarray(eq_src).copy(),
        quantized=quantized,
        valid_mask=sub_mask,
        unique_levels=unique_levels,
        occupancy=occ,
        rgb_mean=rgb_mean,
        rgb_std=rgb_std,
        pan_mean=pan_mean,
        pan_std=pan_std,
        eq_mean=eq_mean,
        eq_std=eq_std,
        block_id=int(block_id),
        sanity_warnings=warnings,
        extra={
            "quantization_mode": quantization_mode,
            "rescale_method": rescale_method,
            "pan_patch": np.asarray(pan_src).copy(),
            "eq_patch": np.asarray(eq_src).copy(),
            "rgb_patch": np.asarray(rgb_patch).copy() if rgb_patch is not None else None,
        },
    )


def get_default_quadrant_crops(
    img_width: int,
    img_height: int,
    patch_w: Optional[int] = None,
    patch_h: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Calculate 4 well-spaced default crops located in the 4 quadrants of the scene.
    Provides suggested labels that the user can freely rename or confirm.
    """
    if patch_w is None:
        patch_w = max(32, min(512, int(img_width * 0.15)))
    if patch_h is None:
        patch_h = max(32, min(512, int(img_height * 0.15)))

    mid_x = img_width // 2
    mid_y = img_height // 2

    patch_w = min(patch_w, mid_x - 16)
    patch_h = min(patch_h, mid_y - 16)

    # Quadrant 1 (Top-Left)
    x1 = (mid_x - patch_w) // 2
    y1 = (mid_y - patch_h) // 2

    # Quadrant 2 (Top-Right)
    x2 = mid_x + (mid_x - patch_w) // 2
    y2 = (mid_y - patch_h) // 2

    # Quadrant 3 (Bottom-Left)
    x3 = (mid_x - patch_w) // 2
    y3 = mid_y + (mid_y - patch_h) // 2

    # Quadrant 4 (Bottom-Right)
    x4 = mid_x + (mid_x - patch_w) // 2
    y4 = mid_y + (mid_y - patch_h) // 2

    return [
        {"id": "patch_1", "label": "Water Bodies", "x": int(x1), "y": int(y1), "w": int(patch_w), "h": int(patch_h), "color": "#00b4d8"},
        {"id": "patch_2", "label": "Vegetation", "x": int(x2), "y": int(y2), "w": int(patch_w), "h": int(patch_h), "color": "#2dc653"},
        {"id": "patch_3", "label": "Agricultural Land", "x": int(x3), "y": int(y3), "w": int(patch_w), "h": int(patch_h), "color": "#fca311"},
        {"id": "patch_4", "label": "Urban / Built-up Area", "x": int(x4), "y": int(y4), "w": int(patch_w), "h": int(patch_h), "color": "#e63946"},
    ]


def draw_bounding_boxes_preview(
    base_image: np.ndarray,
    crops: List[Union[Dict[str, Any], TexturePatch]],
    line_thickness: int = 2,
    max_preview_dim: int = 1024
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Render clearly numbered, colored bounding boxes and labels for the 4 texture patches
    directly onto the full scene.

    Correct coordinate transformation:
      - Takes original-resolution pixel coordinates (X, Y, W, H).
      - Computes exact preview scale factor S_x = preview_w / orig_w, S_y = preview_h / orig_h.
      - Transforms coordinates: x_prev = X * S_x, y_prev = Y * S_y.
      - Never rearranges or combines patches; renders the true complete scene as the main image.

    Returns:
      canvas: 3D uint8 array with bounding boxes drawn over the scene preview.
      info: dictionary with scaling and dimension metadata.
    """
    orig_h, orig_w = base_image.shape[:2]

    # Generate display preview for the browser
    scale = min(1.0, float(max_preview_dim) / float(max(orig_h, orig_w)))
    preview_w = max(64, int(round(orig_w * scale)))
    preview_h = max(64, int(round(orig_h * scale)))

    if scale < 1.0:
        resized = cv2.resize(base_image, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
    else:
        resized = base_image

    if resized.ndim == 2:
        canvas = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    else:
        canvas = resized.copy()

    scale_x = float(preview_w) / float(orig_w)
    scale_y = float(preview_h) / float(orig_h)

    # Distinct, high-contrast BGR colors for OpenCV
    palette = [
        (216, 180, 0),   # Cyan / Blue
        (83, 198, 45),   # Vibrant Green
        (17, 163, 252),  # Amber / Gold
        (70, 57, 230),   # Crimson / Red
    ]

    for idx, item in enumerate(crops):
        if isinstance(item, TexturePatch):
            x, y, w, h = item.x, item.y, item.width, item.height
            label = f"#{idx + 1}: {item.label}"
        else:
            x, y, w, h = int(item["x"]), int(item["y"]), int(item["w"]), int(item["h"])
            label = f"#{idx + 1}: {item.get('label', f'Texture {idx + 1}')}"

        color = palette[idx % len(palette)]

        # Map original coordinates to preview pixels
        x_prev = int(round(x * scale_x))
        y_prev = int(round(y * scale_y))
        w_prev = max(3, int(round(w * scale_x)))
        h_prev = max(3, int(round(h * scale_y)))

        # Draw outer rectangle
        thickness = max(2, int(round(line_thickness * (1.3 if scale < 0.5 else 1.0))))
        cv2.rectangle(canvas, (x_prev, y_prev), (x_prev + w_prev, y_prev + h_prev), color, thickness)

        # Draw label banner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.44
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)

        box_y1 = max(0, y_prev - text_h - 8)
        box_y2 = y_prev
        cv2.rectangle(canvas, (x_prev, box_y1), (min(canvas.shape[1], x_prev + text_w + 8), box_y2), color, -1)
        cv2.putText(
            canvas,
            label,
            (x_prev + 4, max(text_h + 2, y_prev - 5)),
            font,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    info = {
        "original_width": orig_w,
        "original_height": orig_h,
        "preview_width": preview_w,
        "preview_height": preview_h,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "is_scaled": (scale < 1.0)
    }

    return canvas, info
