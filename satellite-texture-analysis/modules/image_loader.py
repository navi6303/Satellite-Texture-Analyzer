"""
Satellite and Multispectral Image Loader Module.
Provides high-performance, memory-efficient loading of real multi-band GeoTIFF
and standard raster imagery using Rasterio without artificial downsampling.
Preserves original spatial resolution and full geospatial metadata.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure rasterio uses its own bundled PROJ database on Windows if another application
# (such as PostgreSQL / PostGIS) set an incompatible system PROJ_LIB
for _proj_env in ("PROJ_LIB", "PROJ_DATA"):
    _val = os.environ.get(_proj_env)
    if _val and ("postgre" in _val.lower() or not Path(_val).exists()):
        os.environ.pop(_proj_env, None)

import cv2
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine


@dataclass
class RasterMetadata:
    """Comprehensive geospatial and raster metadata container."""
    width: int
    height: int
    count: int
    dtypes: Tuple[str, ...]
    crs: str
    crs_epsg: Optional[str] = None
    transform: Optional[Affine] = None
    nodata: Optional[float] = None
    pixel_size: Tuple[float, float] = (1.0, 1.0)
    bounds: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # (left, bottom, right, top)
    driver: str = "GTiff"
    filename: str = "unknown"
    file_size_bytes: int = 0
    band_descriptions: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def file_size_mb(self) -> float:
        return round(self.file_size_bytes / (1024 * 1024), 2)

    @property
    def dimensions_str(self) -> str:
        return f"{self.width} × {self.height}"

    @property
    def pixel_size_str(self) -> str:
        x_res, y_res = abs(self.pixel_size[0]), abs(self.pixel_size[1])
        if x_res == 1.0 and y_res == 1.0 and self.crs == "CRS not defined":
            return "1.0 × 1.0 (Pixel units)"
        return f"{x_res:.2f} × {y_res:.2f} (Map units / meters)"

    @property
    def bounds_str(self) -> str:
        l, b, r, t = self.bounds
        if l == 0.0 and b == 0.0 and r == 0.0 and t == 0.0:
            return "Unprojected / Local coordinate space"
        return f"Left: {l:.2f}, Bottom: {b:.2f}, Right: {r:.2f}, Top: {t:.2f}"


class LoadedRaster:
    """
    Representation of a multi-band satellite raster.
    Holds reference to dataset source, metadata, and handles band reading
    without holding redundant multi-gigabyte arrays simultaneously in RAM.
    """
    def __init__(
        self,
        file_path: Optional[str] = None,
        metadata: Optional[RasterMetadata] = None,
        band_names: Optional[List[str]] = None,
        in_memory_data: Optional[np.ndarray] = None,
        # Backward-compatible kwargs for test helpers
        data: Optional[np.ndarray] = None,
        valid_mask: Optional[np.ndarray] = None,
    ):
        self.file_path = file_path
        self.metadata = metadata
        # Support old test-helper calling convention: LoadedRaster(data=..., band_names=..., valid_mask=..., metadata=...)
        if data is not None and in_memory_data is None:
            in_memory_data = data
        if metadata is not None:
            self.band_names = band_names if band_names is not None else [f"Band {i+1}" for i in range(metadata.count)]
        else:
            self.band_names = band_names or []
        self._in_memory_data = in_memory_data  # Optional array: (count, H, W)
        # Store optional pre-computed valid_mask for in-memory arrays
        self._valid_mask: Optional[np.ndarray] = valid_mask

    @property
    def width(self) -> int:
        return self.metadata.width

    @property
    def height(self) -> int:
        return self.metadata.height

    @property
    def count(self) -> int:
        return self.metadata.count

    def read_band(self, band_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read a single band at 100% full original resolution.
        Returns:
          data: 2D float32 array (height, width)
          valid_mask: 2D bool array (height, width) indicating valid (non-nodata) pixels.
        """
        return self.read_window(band_idx, 0, 0, self.width, self.height)

    def read_window(
        self,
        band_idx: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read one band over a Rasterio window (column x, row y) at native resolution.
        Avoids loading the full scene when only a patch is required.
        """
        if band_idx < 0 or band_idx >= self.count:
            raise IndexError(f"Band index {band_idx} out of range for {self.count}-band raster.")
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)
        if x < 0 or y < 0 or x + width > self.width or y + height > self.height:
            raise ValueError(
                f"Window (x={x}, y={y}, w={width}, h={height}) is outside "
                f"{self.width}×{self.height}."
            )

        if self._in_memory_data is not None:
            data = self._in_memory_data[band_idx, y : y + height, x : x + width].astype(np.float32)
            if self._valid_mask is not None:
                if self._valid_mask.ndim == 3 and self._valid_mask.shape[0] == self.count:
                    mask = np.isfinite(data) & self._valid_mask[band_idx, y : y + height, x : x + width]
                else:
                    mask = np.isfinite(data) & self._valid_mask[y : y + height, x : x + width]
                return data, mask.astype(bool)
            mask = np.isfinite(data)
            if self.metadata.nodata is not None and not np.isnan(self.metadata.nodata):
                mask &= data != self.metadata.nodata
            return data, mask

        if not self.file_path or not Path(self.file_path).exists():
            raise FileNotFoundError(f"Raster source '{self.file_path}' not accessible.")

        from rasterio.windows import Window

        with rasterio.open(self.file_path) as ds:
            data = ds.read(
                band_idx + 1,
                window=Window(col_off=x, row_off=y, width=width, height=height), masked=True,
            ).astype(np.float32)

        values = data.filled(np.nan)
        return values, np.isfinite(values)

    def read_preview_rgb(
        self,
        r_idx: int = 0,
        g_idx: int = 1,
        b_idx: int = 2,
        max_dim: int = 1024,
        p_low: float = 2.0,
        p_high: float = 98.0
    ) -> np.ndarray:
        """
        Generate a fast, downsampled 8-bit RGB preview array for web display ONLY.
        Uses Rasterio's bilinear overview reader so massive files are read in milliseconds.
        The underlying original resolution is NEVER modified.
        """
        orig_h, orig_w = self.height, self.width
        scale = min(1.0, float(max_dim) / float(max(orig_h, orig_w)))
        target_w = max(1, int(round(orig_w * scale)))
        target_h = max(1, int(round(orig_h * scale)))

        indices = [min(max(0, idx), self.count - 1) for idx in [r_idx, g_idx, b_idx]]
        preview_rgb = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        for c_pos, b_idx in enumerate(indices):
            if self._in_memory_data is not None:
                band_full = self._in_memory_data[b_idx]
                band_prev = cv2.resize(band_full, (target_w, target_h), interpolation=cv2.INTER_AREA)
            elif self.file_path and Path(self.file_path).exists():
                with rasterio.open(self.file_path) as ds:
                    band_prev = ds.read(
                        b_idx + 1,
                        out_shape=(target_h, target_w),
                        resampling=Resampling.bilinear, masked=True
                    ).astype(np.float32).filled(np.nan)
            else:
                band_prev = np.zeros((target_h, target_w), dtype=np.float32)

            # 2% - 98% percentile linear stretch on valid pixels
            valid_vals = band_prev[np.isfinite(band_prev)]
            if self.metadata.nodata is not None and not np.isnan(self.metadata.nodata):
                valid_vals = valid_vals[valid_vals != self.metadata.nodata]

            if len(valid_vals) > 0 and not np.isclose(np.min(valid_vals), np.max(valid_vals)):
                vmin = float(np.percentile(valid_vals, p_low))
                vmax = float(np.percentile(valid_vals, p_high))
                denom = vmax - vmin if vmax > vmin else 1.0
                scaled = np.clip((band_prev - vmin) * (255.0 / denom), 0, 255)
                stretched = np.nan_to_num(scaled).astype(np.uint8)
            else:
                stretched = np.zeros((target_h, target_w), dtype=np.uint8)

            preview_rgb[:, :, c_pos] = stretched

        return preview_rgb


def detect_satellite_band_mapping(
    count: int,
    descriptions: List[str]
) -> Dict[str, Any]:
    """
    Intelligently suggest natural-colour RGB band indices based on band count,
    wavelengths, and standard satellite sensor configurations (e.g. Sentinel-2).
    """
    desc_lower = [d.lower() for d in descriptions]
    r_idx, g_idx, b_idx = 0, 0, 0
    is_tcc = False
    composite_label = "Custom Colour Composite"

    # Check for explicit named descriptions
    r_found = [i for i, d in enumerate(desc_lower) if "red" in d and "edge" not in d]
    g_found = [i for i, d in enumerate(desc_lower) if "green" in d]
    b_found = [i for i, d in enumerate(desc_lower) if "blue" in d]

    if r_found and g_found and b_found:
        r_idx = r_found[0]
        g_idx = g_found[0]
        b_idx = b_found[0]
        is_tcc = True
        composite_label = "True Colour Composite (TCC) — Wavelength Matched"
    elif count >= 4:
        # Standard Sentinel-2 Multi-spectral Level-2A/1C:
        # Band 1: Coastal, Band 2: Blue, Band 3: Green, Band 4: Red, Band 8: NIR
        # Natural Colour Composite (TCC) = B4 (Red), B3 (Green), B2 (Blue)
        r_idx = 3  # 0-based for Band 4
        g_idx = 2  # 0-based for Band 3
        b_idx = 1  # 0-based for Band 2
        is_tcc = True
        composite_label = "True Colour Composite (TCC) — Sentinel-2 Standard (B4-Red, B3-Green, B2-Blue)"
    elif count == 3:
        # Standard 3-band raster: usually Red, Green, Blue
        r_idx = 0
        g_idx = 1
        b_idx = 2
        is_tcc = True
        composite_label = "True Colour Composite (TCC) — Channels 1, 2, 3"
    else:
        r_idx = 0
        g_idx = 0
        b_idx = 0
        is_tcc = False
        composite_label = "Single Channel Grayscale Preview"

    return {
        "r_idx": r_idx,
        "g_idx": g_idx,
        "b_idx": b_idx,
        "is_tcc": is_tcc,
        "label": composite_label
    }


def load_geotiff_from_path(file_path: Union[str, Path], filename: str) -> LoadedRaster:
    """
    Read metadata and initialize a LoadedRaster directly from a GeoTIFF on disk.
    Preserves 100% full original dimensions with zero forced downsampling.
    """
    f_path = str(Path(file_path).resolve())
    file_size = os.path.getsize(f_path)

    with rasterio.open(f_path) as ds:
        width = ds.width
        height = ds.height
        count = ds.count
        driver = ds.driver
        dtypes = tuple(str(dt) for dt in ds.dtypes)
        nodata = ds.nodata
        transform = ds.transform
        bounds = (float(ds.bounds.left), float(ds.bounds.bottom), float(ds.bounds.right), float(ds.bounds.top))
        pixel_size = (float(ds.res[0]), float(ds.res[1]))

        # Extract CRS string and EPSG code
        if ds.crs:
            epsg_code = ds.crs.to_epsg()
            crs_epsg = f"EPSG:{epsg_code}" if epsg_code else None
            crs_str = crs_epsg if crs_epsg else str(ds.crs)
        else:
            crs_epsg = None
            crs_str = "CRS not defined"

        # Band descriptions
        band_descriptions = [ds.descriptions[i] or f"Band {i + 1}" for i in range(count)]

    # Generate friendly band names
    band_names = []
    for i in range(count):
        desc = band_descriptions[i]
        if count >= 4 and i == 1 and desc == f"Band {i + 1}":
            band_names.append(f"Band 2 — Blue (~490 nm)")
        elif count >= 4 and i == 2 and desc == f"Band {i + 1}":
            band_names.append(f"Band 3 — Green (~560 nm)")
        elif count >= 4 and i == 3 and desc == f"Band {i + 1}":
            band_names.append(f"Band 4 — Red (~665 nm)")
        elif count >= 4 and i == 7 and desc == f"Band {i + 1}":
            band_names.append(f"Band 8 — NIR (~842 nm)")
        else:
            band_names.append(desc if desc != f"Band {i + 1}" else f"Spectral Band {i + 1}")

    meta = RasterMetadata(
        width=width,
        height=height,
        count=count,
        dtypes=dtypes,
        crs=crs_str,
        crs_epsg=crs_epsg,
        transform=transform,
        nodata=nodata,
        pixel_size=pixel_size,
        bounds=bounds,
        driver=driver,
        filename=filename,
        file_size_bytes=file_size,
        band_descriptions=band_descriptions
    )

    return LoadedRaster(
        file_path=f_path,
        metadata=meta,
        band_names=band_names
    )


def load_standard_image_from_path(file_path: Union[str, Path], filename: str) -> LoadedRaster:
    """
    Load a standard image (PNG, JPEG, WebP) directly from disk at original resolution.
    """
    f_path = str(Path(file_path).resolve())
    file_size = os.path.getsize(f_path)

    with Image.open(f_path) as img:
        img_mode = img.mode
        width, height = img.size

        if img_mode in ("1", "L", "I", "F") or img_mode.startswith("I;16"):
            count = 1
            band_names = ["Grayscale Intensity Channel"]
            gray = np.asarray(img.convert("L") if img_mode == "1" else img)
            arr = gray[np.newaxis, :, :]
            dtypes = (gray.dtype.name,)
        elif img_mode in ("RGB", "P"):
            rgb = img.convert("RGB")
            count = 3
            band_names = ["Channel 1 (Red)", "Channel 2 (Green)", "Channel 3 (Blue)"]
            dtypes = ("uint8", "uint8", "uint8")
            arr = np.transpose(np.array(rgb, dtype=np.float32), (2, 0, 1))
        elif img_mode == "RGBA":
            rgba = np.array(img, dtype=np.float32)
            count = 3
            band_names = ["Channel 1 (Red)", "Channel 2 (Green)", "Channel 3 (Blue)"]
            dtypes = ("uint8", "uint8", "uint8")
            arr = np.transpose(rgba[:, :, :3], (2, 0, 1))
        else:
            rgb = img.convert("RGB")
            count = 3
            band_names = ["Channel 1 (Red)", "Channel 2 (Green)", "Channel 3 (Blue)"]
            dtypes = ("uint8", "uint8", "uint8")
            arr = np.transpose(np.array(rgb, dtype=np.float32), (2, 0, 1))

    meta = RasterMetadata(
        width=width,
        height=height,
        count=count,
        dtypes=dtypes,
        crs="CRS not defined",
        crs_epsg=None,
        transform=None,
        nodata=None,
        pixel_size=(1.0, 1.0),
        bounds=(0.0, 0.0, float(width), float(height)),
        driver="PIL",
        filename=filename,
        file_size_bytes=file_size,
        band_descriptions=band_names
    )

    if count == 3:
        # Pillow stores ordinary color images in RGB order.
        meta.extra["color_channels"] = {"R": 0, "G": 1, "B": 2}

    return LoadedRaster(
        file_path=f_path,
        metadata=meta,
        band_names=band_names,
        in_memory_data=arr
    )


def save_upload_to_working_file(uploaded_file, target_dir: Union[str, Path]) -> Tuple[Path, str]:
    """
    Safely stream an uploaded satellite image from Streamlit into a dedicated local
    working file. Enables direct Rasterio file-backed reading without storing multi-gigabyte
    duplicates in RAM.
    """
    t_dir = Path(target_dir)
    t_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(uploaded_file.name).name
    target_path = t_dir / filename

    uploaded_file.seek(0)
    with open(target_path, "wb") as f_out:
        shutil.copyfileobj(uploaded_file, f_out)

    return target_path, filename


def load_raster(
    source: Union[str, Path, Any],
    filename: str = "uploaded_raster.tif",
    cache_dir: Optional[Union[str, Path]] = None
) -> LoadedRaster:
    """
    Universal satellite raster loader.
    If given an uploaded Streamlit file object, streams it cleanly to a local working file
    before reading with Rasterio.
    Preserves 100% full original resolution with zero forced downsampling.
    """
    if hasattr(source, "read") and hasattr(source, "name"):
        # Streamlit UploadedFile
        default_dir = Path(__file__).resolve().parent.parent / "outputs" / "uploads"
        out_dir = Path(cache_dir) if cache_dir else default_dir
        f_path, fn = save_upload_to_working_file(source, out_dir)
        filename = fn
    elif isinstance(source, (str, Path)):
        f_path = Path(source)
    else:
        raise ValueError(f"Unsupported raster source: {type(source)}")

    lower_fn = str(filename).lower()
    is_tiff = lower_fn.endswith((".tif", ".tiff"))

    if lower_fn.endswith((".hdr", ".img", ".bil", ".bip", ".bsq", ".dat", ".raw")):
        try:
            return load_geotiff_from_path(f_path, filename=filename)
        except rasterio.errors.RasterioIOError:
            if not lower_fn.endswith(".hdr"):
                raise
            bgr = cv2.imread(str(f_path), cv2.IMREAD_UNCHANGED)
            if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or not np.issubdtype(bgr.dtype, np.floating):
                raise
            # Reverse the channel order as a view of the decoded buffer.
            data = np.moveaxis(bgr[:, :, ::-1], 2, 0)
            n, h, w = data.shape
            meta = RasterMetadata(w, h, n, (str(data.dtype),)*n, "CRS not defined",
                                  filename=filename, file_size_bytes=f_path.stat().st_size,
                                  driver="Radiance HDR", band_descriptions=["Red","Green","Blue"])
            return LoadedRaster(str(f_path), meta, list(meta.band_descriptions), data)

    if is_tiff:
        try:
            return load_geotiff_from_path(f_path, filename=filename)
        except Exception as e_geo:
            try:
                return load_standard_image_from_path(f_path, filename=filename)
            except Exception:
                raise ValueError(f"Failed to open GeoTIFF '{filename}': {e_geo}")
    else:
        try:
            return load_standard_image_from_path(f_path, filename=filename)
        except Exception:
            return load_geotiff_from_path(f_path, filename=filename)

def load_envi_upload(header, binary, folder):
    """Keep matching ENVI header/data files together and open the binary via GDAL."""
    data_path, name = save_upload_to_working_file(binary, folder)
    if data_path.suffix.lower() == ".hdr":
        raise ValueError("Choose the ENVI binary pixel file as the companion.")
    hdr_path = data_path.with_suffix(".hdr")
    header.seek(0)
    with hdr_path.open("wb") as f:
        shutil.copyfileobj(header, f)
    return load_geotiff_from_path(data_path, name)
