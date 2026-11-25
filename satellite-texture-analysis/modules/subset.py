"""Crop native source samples before any arithmetic or display scaling."""

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window

from modules.image_loader import load_geotiff_from_path
from modules.texture_extraction import validate_crop


def checked_box(box, width, height):
    values = [box[k] for k in ("x", "y", "w", "h")]
    if any(int(v) != v for v in values):
        raise ValueError("Pixel coordinates must be integers.")
    x, y, w, h = map(int, values)
    ok, reason = validate_crop(x, y, w, h, width, height, min_size=1)
    if not ok:
        raise ValueError(reason)
    return dict(x=x, y=y, w=w, h=h)


def crop_original(source, selection, destination):
    """Materialize all source bands without resampling. PAN uses jointly valid pixels."""
    box = checked_box(selection, source.width, source.height)
    x, y, w, h = box.values()
    meta = source.metadata
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.result_type(*[np.dtype(dt) for dt in meta.dtypes])
    transform = (meta.transform or Affine.identity()) * Affine.translation(x, y)
    profile = dict(driver="GTiff", width=w, height=h, count=source.count,
                   dtype=dtype, transform=transform, tiled=True, blockxsize=256,
                   blockysize=256, BIGTIFF="IF_SAFER", nodata=meta.nodata)
    if meta.crs and meta.crs != "CRS not defined":
        profile["crs"] = meta.crs
    memory = source._in_memory_data
    reader = rasterio.open(source.file_path) if memory is None else nullcontext(None)
    with reader as src, rasterio.open(path, "w", **profile) as dst:
        for band, name in enumerate(source.band_names, 1):
            dst.set_band_description(band, name)
        for _, win in dst.block_windows(1):
            sx, sy = x + int(win.col_off), y + int(win.row_off)
            ww, hh = int(win.width), int(win.height)
            original = Window(sx, sy, ww, hh)
            valid = np.ones((hh, ww), dtype=bool)
            for b in range(source.count):
                if src is not None:
                    arr = src.read(b + 1, window=original, masked=True)
                    valid &= ~np.ma.getmaskarray(arr) & np.isfinite(arr.data)
                    pixels = arr.data
                else:
                    pixels = memory[b, sy:sy+hh, sx:sx+ww]
                    _, band_valid = source.read_window(b, sx, sy, ww, hh)
                    valid &= band_valid
                dst.write(pixels.astype(dtype, copy=False), b + 1, window=win)
            dst.write_mask(valid, window=win)
    subset = load_geotiff_from_path(path, meta.filename)
    subset.metadata.extra.update(source_window=box, original_width=source.width,
                                 original_height=source.height, original_path=source.file_path)
    if "color_channels" in meta.extra:
        subset.metadata.extra["color_channels"] = meta.extra["color_channels"].copy()
    return subset
