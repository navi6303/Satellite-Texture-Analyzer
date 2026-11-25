"""Windowed full-resolution scene products shared by the UI and command-line runner."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.shutil import copy as copy_raster
from rasterio.windows import Window

from modules.subset import checked_box
from modules.neighbors import require_valid_pairs
from modules.preprocessing import equalization_lut_from_histogram, median_from_histogram
from modules.texture_extraction import extract_patch, QUANT_RAW_PAN_FIXED5


@dataclass
class Scene:
    folder: Path
    raster: object
    bands: list
    stats: dict
    hist_before: np.ndarray
    hist_after: np.ndarray | None = None
    provenance: dict = field(default_factory=dict)
    equalization_lut: np.ndarray | None = None

    @property
    def median(self):
        return median_from_histogram(self.hist_after)

    def preview(self, name, max_dim=1024):
        with rasterio.open(self.folder / (name + ".tif")) as src:
            scale = min(1, max_dim / max(src.height, src.width))
            return src.read(
                1,
                out_shape=(
                    max(1, round(src.height * scale)),
                    max(1, round(src.width * scale)),
                ),
            )

    def export_png(self, name, target):
        copy_raster(self.folder / (name + ".tif"), target, driver="PNG", ZLEVEL=3)


LUMA_WEIGHTS = {"B": 0.114, "G": 0.587, "R": 0.299}


def grayscale_weights(raster, bands):
    channels = raster.metadata.extra.get("color_channels")
    if channels and len(bands) == 3 and set(bands) == set(channels.values()):
        return {channels[color]: weight for color, weight in LUMA_WEIGHTS.items()}
    return None


def prepare_scene(raster, bands, folder):
    """Convert native windows to one channel, then stretch valid intensities to 8 bits."""
    if not bands:
        raise ValueError("Select at least one band.")
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    profile = dict(
        driver="GTiff",
        width=raster.width,
        height=raster.height,
        count=1,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        dtype="float32",
        nodata=np.nan,
    )
    if raster.metadata.transform:
        profile.update(transform=raster.metadata.transform)
    if raster.metadata.crs and raster.metadata.crs != "CRS not defined":
        profile.update(crs=raster.metadata.crs)
    weights = grayscale_weights(raster, bands)
    n, total, total2, lo, hi = 0, 0.0, 0.0, np.inf, -np.inf
    with rasterio.open(folder / "native.tif", "w", **profile) as dst:
        for _, win in dst.block_windows(1):
            shape = int(win.height), int(win.width)
            acc = np.zeros(shape, dtype=np.float32)
            mask = np.ones(shape, dtype=bool)
            for band in (weights or bands):
                arr, valid = raster.read_window(
                    band, win.col_off, win.row_off, win.width, win.height
                )
                mask &= valid
                acc += arr if weights is None else arr * weights[band]
            if weights is None:
                acc /= len(bands)
            acc[~mask] = np.nan
            dst.write(acc, 1, window=win)
            vals = acc[mask].astype(np.float64)
            if vals.size:
                lo, hi = min(lo, vals.min()), max(hi, vals.max())
                n += vals.size
                total += vals.sum()
                total2 += np.square(vals).sum()
    if n == 0:
        raise ValueError("The selected bands contain no jointly valid pixels.")
    stats = dict(
        total_pixels=raster.width * raster.height,
        valid_pixels=n,
        min=float(lo),
        max=float(hi),
        mean=total / n,
        std=float(np.sqrt(max(0.0, total2 / n - (total / n) ** 2))),
        valid_percentage=100 * n / (raster.width * raster.height),
    )
    profile.update(dtype="uint8", nodata=None)
    hist = np.zeros(256, dtype=np.int64)
    with (
        rasterio.open(folder / "native.tif") as src,
        rasterio.open(folder / "pan.tif", "w", **profile) as dst,
    ):
        for _, win in src.block_windows(1):
            arr = src.read(1, window=win)
            valid = np.isfinite(arr)
            pixels = np.full(arr.shape, 0, np.uint8)
            pixels[valid] = (
                128
                if hi == lo
                else np.clip(
                    np.rint((arr[valid] - float(lo)) * float(255 / (hi - lo))), 0, 255
                )
            )
            hist += np.bincount(pixels[valid], minlength=256)
            dst.write(pixels, 1, window=win)
    method = "weighted_gray" if weights else ("single_band" if len(bands) == 1 else "band_mean")
    return Scene(folder, raster, list(bands), stats, hist, provenance={"conversion": method})


def equalize_scene(scene):
    lut = equalization_lut_from_histogram(scene.hist_before)
    with (
        rasterio.open(scene.folder / "pan.tif") as src,
        rasterio.open(scene.folder / "native.tif") as native,
    ):
        with rasterio.open(scene.folder / "eq.tif", "w", **src.profile) as dst:
            for _, win in src.block_windows(1):
                eq = lut[src.read(1, window=win)]
                eq[~np.isfinite(native.read(1, window=win))] = 0
                dst.write(eq, 1, window=win)
    scene.hist_after = np.zeros(256, dtype=np.int64)
    np.add.at(scene.hist_after, lut, scene.hist_before)
    scene.equalization_lut = lut
    return scene


def inspect_patch(scene, box, distance, angles):
    """Geometry, mask-based eligibility and a display thumbnail before quantization."""
    box = checked_box(box, scene.raster.width, scene.raster.height)
    win = Window(box["x"], box["y"], box["w"], box["h"])
    with rasterio.open(scene.folder / "native.tif") as native:
        mask = np.isfinite(native.read(1, window=win))
    counts = require_valid_pairs(mask, distance, angles)
    scale = min(1, 144 / max(win.width, win.height))
    shape = (max(1, round(win.height * scale)), max(1, round(win.width * scale)))
    with rasterio.open(scene.folder / "eq.tif") as eq:
        thumbnail = eq.read(1, window=win, out_shape=shape)
    return thumbnail, counts


def read_patch(scene, box, label, patch_id, mode, method, rgb_bands=None, *, distance=1, angles=None):
    box = checked_box(box, scene.raster.width, scene.raster.height)
    x, y, w, h = (box[k] for k in ("x", "y", "w", "h"))
    win = Window(x, y, w, h)
    with rasterio.open(scene.folder / "eq.tif") as src:
        eq = src.read(1, window=win)
    with rasterio.open(scene.folder / "native.tif") as src:
        pan = src.read(1, window=win)
    mask = np.isfinite(pan)
    if angles is not None:
        require_valid_pairs(mask, distance, angles, patch_id)
    source = eq
    if mode == QUANT_RAW_PAN_FIXED5:
        if any(scene.raster.metadata.dtypes[b] != "uint8" for b in scene.bands):
            with rasterio.open(scene.folder / "pan.tif") as src:
                source = src.read(1, window=win)
        else:
            source = np.where(mask, pan, 0)
    rgb = None
    if rgb_bands is not None and scene.raster.count >= 3:
        # Keep native DN values; composite display contrast stretching is never used here.
        channels = [scene.raster.read_window(b, x, y, w, h) for b in rgb_bands]
        rgb = np.stack(
            [np.where(valid, arr, np.nan) for arr, valid in channels], axis=-1
        )
    patch = extract_patch(
        source,
        0,
        0,
        w,
        h,
        min_size=1,
        valid_mask=mask,
        label=label,
        patch_id=patch_id,
        quantization_mode=mode,
        rescale_method=method,
        pan_patch=pan,
        eq_patch=eq,
        rgb_patch=rgb,
        scene_median_eq=scene.median,
    )
    if rgb is not None:
        from PIL import Image

        values = np.nan_to_num(rgb)
        if any(scene.raster.metadata.dtypes[b] != "uint8" for b in rgb_bands):
            values = (values - np.nanmin(values)) * 255 / max(1e-12, np.ptp(values))
        thumb = Image.fromarray(np.clip(values, 0, 255).astype(np.uint8))
        thumb.thumbnail((144, 144))
        patch.extra["thumbnail"] = np.asarray(thumb)
    patch.extra["bands_one_based"] = [i + 1 for i in scene.bands]
    patch.x, patch.y = x, y
    origin = scene.raster.metadata.extra.get("source_window", {})
    patch.extra.update(original_x=x + origin.get("x", 0), original_y=y + origin.get("y", 0))
    patch.validate_footprint()
    return patch
