"""Shared pixel-offset geometry for patch collection, GLCMs and the worked demo."""

import numpy as np

# A pragmatic floor for an empirical 5-level GLCM, before symmetrization.
# One pair is mathematically sufficient; ten observations still imply coarse estimates.
MIN_NEIGHBOR_PAIRS = 10


def pixel_offset(distance, angle_rad):
    """Raster row and column steps. Positive rows point down the displayed image."""
    dr = int(np.rint(distance * np.sin(angle_rad)))
    dc = int(np.rint(distance * np.cos(angle_rad)))
    return dr, dc


def pair_slices(shape, distance, angle_rad):
    h, w = shape
    dr, dc = pixel_offset(distance, angle_rad)
    if abs(dr) >= h or abs(dc) >= w:
        return np.s_[0:0, 0:0], np.s_[0:0, 0:0]
    src = np.s_[max(0, -dr):h-max(0, dr), max(0, -dc):w-max(0, dc)]
    dst = np.s_[max(0, dr):h-max(0, -dr), max(0, dc):w-max(0, -dc)]
    return src, dst


def require_valid_pairs(mask, distance, angles, patch_name="Selected patch"):
    if not angles:
        raise ValueError("Select at least one GLCM direction.")
    counts = {}
    for angle in angles:
        a, b = pair_slices(mask.shape, distance, np.radians(angle))
        count = int(np.count_nonzero(mask[a] & mask[b]))
        if count < MIN_NEIGHBOR_PAIRS:
            h, w = mask.shape
            raise ValueError(
                f"{patch_name} ({w}×{h} px, {np.count_nonzero(mask):,} valid pixels) "
                f"contains {count} valid neighboring pixel pairs at {angle:g}°, d={distance}. "
                f"At least {MIN_NEIGHBOR_PAIRS} are required per selected offset. "
                "Choose a larger region or one with more adjacent valid pixels."
            )
        counts[angle] = count
    return counts
