"""Box-to-class provenance and the two distinct matrix definitions in the handbook."""

import json

import numpy as np
import pandas as pd

from modules.neighbors import pixel_offset

PCA_FEATURES = ["contrast", "entropy", "homogeneity", "correlation", "asm"]


def matrix_audit(patches):
    rows = []
    for p in patches:
        P = p.glcm
        peak = int(np.argmax(P.diagonal()))
        rows.append(
            {
                "Patch ID": p.patch_id,
                "Class": p.label,
                "x": p.x,
                "y": p.y,
                "width": p.width,
                "height": p.height,
                "P shape": f"{P.shape[0]}×{P.shape[1]}",
                "P sum": float(P.sum()),
                "Symmetry error": float(np.abs(P - P.T).max()),
                "Peak diagonal level": peak,
                "Largest diagonal": float(P[peak, peak]),
                "Pixel occupancy at peak": float(p.occupancy[peak]),
                "Trace": float(np.trace(P)),
                "λ1": float(p.eigenvalues[0]),
                "λ1 minus largest diagonal": float(p.eigenvalues[0] - P[peak, peak]),
                "Eigenvalue sum minus trace": float(p.eigenvalues.sum() - np.trace(P)),
            }
        )
    return pd.DataFrame(rows)


def directional_table(patches):
    """One auditable count record for one patch and one offset; never pool counts."""
    rows = []
    for patch in patches:
        valid_pixels = patch.valid_pixel_count
        for offset in patch.extra.get("directional", []):
            raw = offset["raw_counts"]
            symmetric = raw + raw.T
            dy, dx = pixel_offset(offset["distance"], np.radians(offset["angle_deg"]))
            rows.append(dict(
                patch_id=patch.patch_id, label=patch.label, x=patch.x, y=patch.y,
                original_x=patch.extra.get("original_x", patch.x),
                original_y=patch.extra.get("original_y", patch.y),
                width=patch.width, height=patch.height, total_pixels=patch.total_pixels,
                valid_pixels=valid_pixels, delta_row=dy, delta_column=dx,
                possible_pairs=max(0, patch.height-abs(dy)) * max(0, patch.width-abs(dx)),
                raw_matrix_sum=int(raw.sum()), symmetric_matrix_sum=int(symmetric.sum()),
                matrix_sum=float(offset["probability"].sum()),
                raw_counts=json.dumps(raw.tolist()), symmetric_counts=json.dumps(symmetric.tolist()),
                **{k: v for k, v in offset.items() if k not in ("raw_counts", "probability")},
            ))
    return pd.DataFrame(rows)


def directional_covariance(patches):
    """One observation per selected offset within this same native window."""
    rows = []
    for p in patches:
        directions = p.extra.get("directional", [])
        if len(directions) < 2:
            continue
        X = np.array([[d[k] for k in PCA_FEATURES] for d in directions])
        covariance = np.cov(X, rowvar=False, ddof=1)
        # Covariance is PSD. Round-off may leave numerically negative zeros.
        spectrum = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0)
        rows.append(
            {
                "Patch ID": p.patch_id,
                "Class": p.label,
                "Offset observations": len(X),
                **{f"PC{i}": float(v) for i, v in enumerate(spectrum, 1)},
            }
        )
    return pd.DataFrame(rows)
