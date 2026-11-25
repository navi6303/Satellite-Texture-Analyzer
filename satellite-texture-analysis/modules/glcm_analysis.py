"""
GLCM Analysis and Eigenvalue Computation Module.
Implements:
  1. Mask-aware Gray-Level Co-occurrence Matrix (GLCM) calculation with 5 levels.
  2. Eigenvalue calculation using numpy.linalg.eigvalsh for symmetric matrices.
  3. Feature table construction and pairwise Euclidean distance metrics.
  4. Dynamic, mathematically rigorous scientific interpretation of separability.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    from skimage.feature import graycomatrix
except ModuleNotFoundError as exc:
    if exc.name != "skimage":
        raise
    graycomatrix = None

from modules.neighbors import pair_slices, pixel_offset
from modules.texture_extraction import TexturePatch


def _pair_chunks(q, mask, src, dst):
    """Row-major pairs with both endpoints inside one patch and both valid."""
    left, right = q[src], q[dst]
    if not left.size:
        return
    rows_per_chunk = max(1, 262144 // left.shape[1])
    for start in range(0, left.shape[0], rows_per_chunk):
        rows = slice(start, start + rows_per_chunk)
        if mask is None:
            yield left[rows].ravel(), right[rows].ravel()
        else:
            valid = mask[src][rows] & mask[dst][rows]
            yield left[rows][valid], right[rows][valid]


def _directed_counts(q, mask, distance, theta, levels):
    """Integer histogram before symmetrization. Receives a single quantized patch."""
    q = np.asarray(q)
    if q.ndim != 2 or not q.size or distance < 1 or not np.issubdtype(q.dtype, np.integer):
        raise ValueError("Provide a nonempty 2D integer patch and a positive GLCM distance.")
    if q.min() < 0 or q.max() >= levels:
        raise ValueError("Invalid gray level in the quantized patch.")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != q.shape:
            raise ValueError(f"Validity mask shape {mask.shape} differs from patch shape {q.shape}.")
    src, dst = pair_slices(q.shape, distance, theta)
    # graycomatrix internally uses uint32 bins. Use the int64 path for larger images.
    if (graycomatrix is not None and q.size <= np.iinfo(np.uint32).max
            and (mask is None or mask.all())):
        return graycomatrix(q, [distance], [theta], levels=levels,
                            symmetric=False, normed=False)[:, :, 0, 0].astype(np.int64)
    counts = np.zeros(levels ** 2, dtype=np.int64)
    for left, right in _pair_chunks(q, mask, src, dst):
        codes = left.astype(np.int64) * levels + right
        counts += np.bincount(codes, minlength=levels ** 2)
    return counts.reshape(levels, levels)


def _matrix_from_counts(raw, symmetric, normed):
    pairs = int(raw.sum())
    if not pairs:
        raise ValueError("No valid pixel pairs found for the selected offset.")
    counts = raw + raw.T if symmetric else raw
    matrix = counts.astype(np.float64)
    if normed:
        matrix /= int(counts.sum())
    return matrix, pairs


def compute_glcm(quantized_patch, valid_mask=None, distance=1, angle_rad=0.0,
                 symmetric=True, normed=True, levels=5):
    """Return one patch's GLCM plus its directed pair count before symmetrization."""
    raw = _directed_counts(quantized_patch, valid_mask, distance, angle_rad, levels)
    return _matrix_from_counts(raw, symmetric, normed)


def compute_eigenvalues(glcm: np.ndarray, check_symmetry: bool = True) -> np.ndarray:
    """
    Calculate the eigenvalues of a symmetric 5x5 GLCM using numpy.linalg.eigvalsh.

    Returns:
      1D np.ndarray of shape (5,) with signed eigenvalues sorted by descending absolute value:
      [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5].
    """
    if glcm.shape != (5, 5):
        raise ValueError(f"Expected 5x5 GLCM, got shape {glcm.shape}")

    if check_symmetry and not np.allclose(glcm, glcm.T, atol=1e-7):
        raise ValueError(
            "GLCM must be symmetric to calculate eigenvalues with eigvalsh."
        )

    # numpy.linalg.eigvalsh returns eigenvalues in ascending order
    raw_eigenvalues = np.linalg.eigvalsh(glcm)

    if not np.all(np.isfinite(raw_eigenvalues)):
        raise ValueError("GLCM eigenvalues contain NaN or Infinite values.")

    # Sort by magnitude while retaining each algebraic sign.
    values = raw_eigenvalues[::-1]
    magnitude = np.abs(values)
    # Perron root is positive. Stabilize a possible ±Perron tie at floating precision.
    if np.all(glcm >= 0):
        magnitude[
            np.isclose(magnitude, values[0], rtol=8 * np.finfo(float).eps, atol=0)
        ] = values[0]
    eigenvalues = values[np.argsort(-magnitude, kind="stable")]

    return eigenvalues


DEFAULT_ANGLES_DEG = (0, 45, 90, 135)


def classic_glcm_features(glcm: np.ndarray) -> Dict[str, float]:
    """Haralick-style scalars plus trace and diagonal share of a normalized GLCM."""
    p = np.asarray(glcm, dtype=np.float64)
    n = p.shape[0]
    i, j = np.ogrid[0:n, 0:n]
    diff2 = (i - j).astype(np.float64) ** 2
    contrast = float(np.sum(p * diff2))
    homogeneity = float(np.sum(p / (1.0 + diff2)))
    asm = float(np.sum(p * p))
    pos = p[p > 0]
    marginal = p.sum(axis=1)
    levels = np.arange(n)
    mean = float(levels @ marginal)
    var = float((levels - mean) ** 2 @ marginal)
    my = float(levels @ p.sum(axis=0))
    vy = float((levels - my) ** 2 @ p.sum(axis=0))
    # A constant patch has no measurable linear association. Use the standard
    # constant-window convention (1) so a constant state remains usable in CV.
    corr = (
        float(np.sum((i - mean) * (j - my) * p) / np.sqrt(var * vy))
        if var > 0 and vy > 0
        else 1.0
    )
    return dict(
        contrast=contrast,
        dissimilarity=float(np.sum(np.abs(i - j) * p)),
        homogeneity=homogeneity,
        asm=asm,
        energy=float(np.sqrt(asm)),
        entropy=float(-pos @ np.log2(pos)),
        correlation=float(np.clip(corr, -1, 1)),
        glcm_mean=mean,
        glcm_variance=var,
        max_probability=float(p.max()),
        glcm_trace=float(np.trace(p)),
        diagonal_share=float(np.trace(p)),
    )


def row_normalized_transition_eigenvalues(glcm: np.ndarray) -> np.ndarray:
    """
    Eigenvalues of the row-normalized transition matrix T_ij = P_ij / sum_j P_ij.

    This is a stochastic matrix; for a symmetric GLCM the chain is reversible
    and the spectrum is real. Values are sorted descending and are more robust
    to gray-level occupancy (tone) than eigenvalues of P itself.
    """
    p = np.asarray(glcm, dtype=np.float64)
    masses = p.sum(axis=1)
    roots = np.sqrt(masses[:, None] * masses[None, :])
    # Similar symmetric matrix on occupied states. Unoccupied states have zero rows.
    similar = np.divide(p, roots, out=np.zeros_like(p), where=roots > 0)
    return np.linalg.eigvalsh(similar)[::-1]


def expected_horizontal_pairs(height: int, width: int, distance: int = 1) -> int:
    """Unmasked pair count for θ=0° (horizontal), d pixels apart: h × (w − d)."""
    return int(height) * max(0, int(width) - int(distance))


def compute_averaged_glcm(
    quantized_patch: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    distances: Optional[List[int]] = None,
    angles_deg: Optional[List[float]] = None,
    symmetric: bool = True,
    normed: bool = True,
    levels: int = 5,
    directional: Optional[list] = None,
) -> Tuple[np.ndarray, int, Dict[str, int]]:
    """
    Average normalized GLCMs over the selected distances and angles.

    Returns the mean GLCM, the θ=0° pair count at the first distance (for the
    h×(w−1) check), and a dict of pair counts keyed by ``d{d}_a{angle}``.
    """
    if distances is None:
        distances = [1]
    if angles_deg is None:
        angles_deg = [0.0]
    mats = []
    pair_map: Dict[str, int] = {}
    horiz_pairs = 0
    first_d = int(distances[0])
    for d in distances:
        for ang in angles_deg:
            ang_f = float(ang)
            raw = _directed_counts(quantized_patch, valid_mask, int(d), np.radians(ang_f), levels)
            glcm, n_pairs = _matrix_from_counts(raw, symmetric, normed)
            mats.append(glcm)
            if directional is not None:
                directional.append(
                    dict(
                        distance=int(d),
                        angle_deg=ang_f,
                        raw_counts=raw,
                        pairs=n_pairs,
                        symmetric_pair_total=2 * n_pairs,
                        probability=glcm.copy(),
                        **classic_glcm_features(glcm),
                    )
                )
            pair_map[f"d{int(d)}_a{int(round(ang_f))}"] = int(n_pairs)
            if int(d) == first_d and abs(ang_f) < 1e-9:
                horiz_pairs = int(n_pairs)
    mean_glcm = np.mean(np.stack(mats, axis=0), axis=0)
    if normed:
        s = float(np.sum(mean_glcm))
        if s > 0:
            mean_glcm = mean_glcm / s
    return mean_glcm, horiz_pairs, pair_map


def analyze_patch_glcm(
    patch: TexturePatch,
    distance: int = 1,
    angle_rad: float = 0.0,
    symmetric: bool = True,
    normed: bool = True,
    angles_deg: Optional[List[float]] = None,
    distances: Optional[List[int]] = None,
    compute_transition: bool = True,
) -> TexturePatch:
    """
    Compute GLCM and eigenvalues for a TexturePatch and update its fields in-place.

    If ``angles_deg`` is provided, GLCMs are averaged over those directions
    (rotation-invariant protocol). ``distance`` is used when ``distances`` is None.
    """
    if distances is None:
        distances = [int(distance)]
    if angles_deg is None:
        # Backward compatible: single angle from angle_rad
        angles_deg = [float(np.degrees(angle_rad))]

    patch.validate_footprint()
    directional = []
    glcm, horiz_pairs, pair_map = compute_averaged_glcm(
        patch.quantized,
        valid_mask=patch.valid_mask,
        distances=[int(d) for d in distances],
        angles_deg=[float(a) for a in angles_deg],
        symmetric=symmetric,
        normed=normed,
        levels=5,
        directional=directional if symmetric and normed else None,
    )
    patch.extra["directional"] = directional

    eigenvalues = compute_eigenvalues(glcm, check_symmetry=symmetric)
    patch.glcm = glcm
    patch.eigenvalues = eigenvalues
    patch.valid_pair_count = sum(pair_map.values())
    # Always expose the d=1 horizontal audit, even when excluded from the spectrum.
    mask = patch.valid_mask
    patch.horizontal_valid_pairs = pair_map.get(
        "d1_a0", int(np.count_nonzero(mask[:, :-1] & mask[:, 1:]))
    )
    patch.classic_features = classic_glcm_features(glcm)
    patch.transition_eigenvalues = None
    if compute_transition:
        patch.transition_eigenvalues = row_normalized_transition_eigenvalues(glcm)
    patch.extra["distance"] = (
        int(distances[0]) if len(distances) == 1 else list(distances)
    )
    patch.extra["distances"] = [int(d) for d in distances]
    patch.extra["angles_deg"] = [float(a) for a in angles_deg]
    patch.extra["angle_rad"] = angle_rad
    patch.extra["symmetric"] = symmetric
    patch.extra["normed"] = normed
    patch.extra["pair_counts"] = pair_map
    return patch


EIG_COLUMNS = ["λ1 (Dominant)", "λ2", "λ3", "λ4", "λ5"]
CLASSIC_COLUMNS = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "asm",
    "energy",
    "entropy",
    "correlation",
    "glcm_mean",
    "glcm_variance",
    "max_probability",
    "glcm_trace",
    "diagonal_share",
]
TRANS_COLUMNS = ["Tλ1", "Tλ2", "Tλ3", "Tλ4", "Tλ5"]


def _patch_feature_vector(
    patch: TexturePatch,
    feature_set: str = "combined",
    include_transition: bool = False,
) -> Tuple[List[str], np.ndarray]:
    names: List[str] = []
    vals: List[float] = []
    if feature_set in ("eigenvalues", "combined") and patch.eigenvalues is not None:
        names.extend(EIG_COLUMNS)
        vals.extend([float(v) for v in patch.eigenvalues])
    if feature_set in ("classic", "combined"):
        for k in CLASSIC_COLUMNS[:-1]:  # diagonal_share is identical to trace
            names.append(k)
            vals.append(float(patch.classic_features.get(k, np.nan)))
    if (
        feature_set == "transition"
        and include_transition
        and patch.transition_eigenvalues is not None
    ):
        names.extend(TRANS_COLUMNS)
        te = patch.transition_eigenvalues
        vals.extend([float(te[i]) if i < len(te) else np.nan for i in range(5)])
    return names, np.asarray(vals, dtype=np.float64)


def build_feature_matrix(
    patches: List[TexturePatch],
    feature_set: str = "combined",
    include_transition: bool = False,
) -> Tuple[np.ndarray, List[str], List[str], List[int]]:
    valid = [p for p in patches if p.eigenvalues is not None]
    if not valid:
        return np.zeros((0, 0)), [], [], []
    names, first = _patch_feature_vector(valid[0], feature_set, include_transition)
    rows = [first]
    for p in valid[1:]:
        _, v = _patch_feature_vector(p, feature_set, include_transition)
        rows.append(v)
    X = np.vstack(rows)
    labels = [p.label for p in valid]
    blocks = [int(p.block_id) for p in valid]
    return X, names, labels, blocks


def zscore_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pooled z-score over all patches. Zero-variance columns are left at 0."""
    if X.size == 0:
        return X, np.array([]), np.array([])
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=0)
    sd_safe = np.where(sd < 1e-15, 1.0, sd)
    Z = (X - mu) / sd_safe
    Z[:, sd < 1e-15] = 0.0
    return Z, mu, sd


def pairwise_distance_matrix(X: np.ndarray, labels: List[str]) -> pd.DataFrame:
    n = len(labels)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dist[i, j] = float(np.linalg.norm(X[i] - X[j]))
    return pd.DataFrame(dist, index=labels, columns=labels)


def class_mean_distance_matrix(X: np.ndarray, labels: List[str]) -> pd.DataFrame:
    """Euclidean distances between class centroids."""
    uniq = []
    for lab in labels:
        if lab not in uniq:
            uniq.append(lab)
    means = []
    for lab in uniq:
        idx = [i for i, L in enumerate(labels) if L == lab]
        means.append(np.mean(X[idx], axis=0))
    M = np.vstack(means)
    return pairwise_distance_matrix(M, uniq)


def fisher_ratios(
    X: np.ndarray, labels: List[str], feature_names: List[str]
) -> pd.DataFrame:
    """
    Per-feature Fisher ratio = between-class SS / within-class SS.
    A tiny epsilon keeps every ratio finite (required by automated checks).
    """
    y = np.asarray(labels)
    records = []
    if min(np.unique(y, return_counts=True)[1]) < 2:
        return pd.DataFrame(
            columns=["feature", "fisher_ratio", "between_ss", "within_ss"]
        )
    n, f = X.shape
    for k in range(f):
        x = X[:, k]
        grand = float(np.mean(x))
        between = 0.0
        within = 0.0
        for lab in np.unique(y):
            xc = x[y == lab]
            mu = float(np.mean(xc))
            between += float(len(xc)) * (mu - grand) ** 2
            within += float(np.sum((xc - mu) ** 2))
        # Regularize only zero/tiny denominators in the units of this feature.
        ratio = between / max(within, np.finfo(float).eps * max(between, 1e-20))
        if np.ptp(x) <= 1e-12:
            ratio = 0.0
        records.append(
            {
                "feature": feature_names[k],
                "fisher_ratio": float(ratio),
                "between_ss": float(between),
                "within_ss": float(within),
            }
        )
    df = (
        pd.DataFrame(records)
        .sort_values("fisher_ratio", ascending=False)
        .reset_index(drop=True)
    )
    return df


def build_classic_table(
    patches: List[TexturePatch], include_transition: bool = True
) -> pd.DataFrame:
    records = []
    for p in patches:
        rec: Dict[str, Any] = {"Class Label": p.label, "Patch ID": p.patch_id}
        rec.update({k: float(v) for k, v in (p.classic_features or {}).items()})
        if include_transition and p.transition_eigenvalues is not None:
            for i, v in enumerate(p.transition_eigenvalues):
                rec[f"Tλ{i + 1}"] = float(v)
        records.append(rec)
    return pd.DataFrame(records)


def class_mean_std_table(patches: List[TexturePatch]) -> pd.DataFrame:
    """Mean ± std of the five GLCM eigenvalues per class."""
    rows = []
    labels = []
    for p in patches:
        if p.label not in labels:
            labels.append(p.label)
    for lab in labels:
        group = [p for p in patches if p.label == lab and p.eigenvalues is not None]
        if not group:
            continue
        eigs = np.vstack([p.eigenvalues for p in group])
        rec: Dict[str, Any] = {"Class Label": lab, "N patches": len(group)}
        for i in range(5):
            rec[f"λ{i + 1} mean"] = float(np.mean(eigs[:, i]))
            rec[f"λ{i + 1} std"] = float(
                np.std(eigs[:, i], ddof=1) if len(group) > 1 else np.nan
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def build_eigenvalue_table(patches: List[TexturePatch]) -> pd.DataFrame:
    """
    Construct a structured Pandas DataFrame summarizing the 5 eigenvalues
    for all texture classes.
    """
    records = []
    for patch in patches:
        if patch.eigenvalues is None:
            continue
        rec = {
            "Class Label": patch.label,
            "Patch ID": patch.patch_id,
            "Dimensions": f"{patch.width}x{patch.height}",
            "Valid Pixels": patch.valid_pixel_count,
            "Horizontal Valid Pairs (θ=0°)": patch.horizontal_valid_pairs,
            "Valid Pairs": patch.valid_pair_count,
            "λ1 (Dominant)": patch.eigenvalues[0],
            "λ2": patch.eigenvalues[1],
            "λ3": patch.eigenvalues[2],
            "λ4": patch.eigenvalues[3],
            "λ5": patch.eigenvalues[4],
            "Sum (Trace)": float(np.sum(patch.eigenvalues)),
        }
        records.append(rec)

    df = pd.DataFrame(records)
    return df


def calculate_feature_distances(patches: List[TexturePatch]) -> pd.DataFrame:
    """
    Calculate pairwise Euclidean distances between the 5D eigenvalue vectors
    of each texture patch:

        d(A, B) = sqrt( sum_{k=1}^5 (λ_k^A - λ_k^B)^2 )
    """
    labels = [p.label for p in patches if p.eigenvalues is not None]
    vectors = [p.eigenvalues for p in patches if p.eigenvalues is not None]
    n = len(labels)

    dist_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = float(np.linalg.norm(vectors[i] - vectors[j]))

    df_dist = pd.DataFrame(dist_matrix, index=labels, columns=labels)
    return df_dist


def generate_separability_interpretation(
    patches: List[TexturePatch],
    glcm_distance: int = 1,
    glcm_angle_deg: float = 0.0,
    df_fisher: Optional[pd.DataFrame] = None,
    df_dist_raw: Optional[pd.DataFrame] = None,
    df_dist_std: Optional[pd.DataFrame] = None,
    cv_results: Optional[Dict[str, Any]] = None,
    quantization_mode: str = "",
    angles_deg: Optional[List[float]] = None,
) -> Dict[str, Union[str, List[str], float]]:
    """
    Data-driven interpretation. No hardcoded physical claims (e.g. 'high λ1 = smooth water').
    Primary discriminator is the feature with the largest Fisher ratio, not the eigenvalue range.
    """
    valid_patches = [p for p in patches if p.eigenvalues is not None]
    if len(valid_patches) < 2:
        return {
            "summary": "At least two texture patches with computed eigenvalues are required for separability analysis.",
            "details": [],
            "max_pair": ("", ""),
            "min_pair": ("", ""),
            "max_distance": 0.0,
            "min_distance": 0.0,
            "dominant_dims": [],
        }

    eigs = np.array([p.eigenvalues for p in valid_patches])
    labels = [p.label for p in valid_patches]
    n_classes = len(set(labels))
    counts = {lab: labels.count(lab) for lab in dict.fromkeys(labels)}

    if df_dist_std is not None and len(df_dist_std) > 0:
        dist_df = df_dist_std
        dist_kind = "standardized eigenvalue"
    elif df_dist_raw is not None and len(df_dist_raw) > 0:
        dist_df = df_dist_raw
        dist_kind = "raw eigenvalue"
    else:
        dist_df = class_mean_distance_matrix(eigs, labels)
        dist_kind = "raw eigenvalue"

    pairs = []
    cols = list(dist_df.columns)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            pairs.append((a, b, float(dist_df.iloc[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    max_pair = (pairs[0][0], pairs[0][1]) if pairs else ("", "")
    max_dist = pairs[0][2] if pairs else 0.0
    min_pair = (pairs[-1][0], pairs[-1][1]) if pairs else ("", "")
    min_dist = pairs[-1][2] if pairs else 0.0

    if df_fisher is not None and len(df_fisher) > 0:
        top_feat = str(df_fisher.iloc[0]["feature"])
        top_fisher = float(df_fisher.iloc[0]["fisher_ratio"])
        ranked = df_fisher["feature"].tolist()
    elif min(counts.values()) > 1:
        # Fallback: Fisher on eigenvalues only
        X = eigs
        fisher_df = fisher_ratios(X, labels, EIG_COLUMNS)
        top_feat = str(fisher_df.iloc[0]["feature"])
        top_fisher = float(fisher_df.iloc[0]["fisher_ratio"])
        ranked = fisher_df["feature"].tolist()

    else:
        top_feat, top_fisher, ranked = None, None, []

    gap = [abs(p.eigenvalues[0] - np.diag(p.glcm).max()) for p in valid_patches]
    shares = [np.trace(p.glcm) for p in valid_patches]
    details = [
        f"Across {len(valid_patches)} patches, mean diagonal share was {np.mean(shares):.4f} "
        f"and the mean absolute gap between λ1 and the largest diagonal entry was {np.mean(gap):.4f}.",
        f"The closest {dist_kind} centroids were {min_pair[0]} and {min_pair[1]} ({min_dist:.3f}). "
        f"The farthest were {max_pair[0]} and {max_pair[1]} ({max_dist:.3f}).",
        f"The highest Fisher ratio was {top_fisher:.3f} for {top_feat}"
        + (
            " (trace and diagonal share measure the same quantity)."
            if top_feat in ("diagonal_share", "glcm_trace")
            else "."
        )
        if min(counts.values()) > 1
        else "Within-class scatter requires replicated patches.",
    ]
    summary = (
        f"{len(valid_patches)} patches represented {n_classes} classes. " + details[-1]
    )
    if "Water Bodies" in labels and "Vegetation" in labels:
        notes = []
        for lab in ("Water Bodies", "Vegetation"):
            group = [p for p in valid_patches if p.label == lab]
            notes.append(
                f"{lab}: λ1 {np.mean([p.eigenvalues[0] for p in group]):.4f}, "
                f"largest diagonal entry {np.mean([p.glcm.diagonal().max() for p in group]):.4f}, "
                f"diagonal share {np.mean([p.glcm.trace() for p in group]):.4f}, "
                f"most populated level {np.mean([max(p.occupancy) for p in group]):.1%}"
            )
        details.append("Measured class averages. " + ". ".join(notes) + ".")
        water = [p for p in valid_patches if p.label == "Water Bodies"]
        vegetation = [p for p in valid_patches if p.label == "Vegetation"]

        def avg(group, getter):
            return float(np.mean([getter(p) for p in group]))

        water_lambda, veg_lambda = (
            avg(group, lambda p: p.eigenvalues[0]) for group in (water, vegetation)
        )
        water_trace, veg_trace = (
            avg(group, lambda p: p.glcm.trace()) for group in (water, vegetation)
        )
        if water_trace > veg_trace and water_lambda < veg_lambda:
            details.append(
                "Water has more same-level neighbor pairs overall, but those pairs are spread over several levels. "
                "Vegetation has a higher probability concentrated in a single level pair. These measurements explain the opposite trace and λ1 orderings."
            )
        summary += f" In Pipeline A, Water λ1 was {water_lambda:.4f} and Vegetation λ1 was {veg_lambda:.4f}."
    if cv_results and cv_results.get("best_feature_set"):
        winners = ", ".join(cv_results["winners"])
        summary += f" Highest blocked accuracy was {cv_results['best_accuracy']:.1%} ({winners})."
        for winner in cv_results["winners"]:
            feature, clf = winner.split(" / ")
            score = cv_results["by_set"][feature][clf]
            names = score["class_names"]
            if "Urban / Built-up Area" in names and "Bare Soil" in names:
                u, b = names.index("Urban / Built-up Area"), names.index("Bare Soil")
                cm = score["confusion_matrix"]
                details.append(
                    f"{winner}: {cm[u, b]}/{cm[u].sum()} Urban patches were predicted as "
                    f"Bare Soil and {cm[b, u]}/{cm[b].sum()} Bare Soil patches as Urban."
                )
    elif cv_results:
        details.append(cv_results["by_set"]["combined"]["note"])
    return dict(
        summary=summary,
        details=details,
        min_pair=min_pair,
        max_pair=max_pair,
        min_distance=min_dist,
        max_distance=max_dist,
        ranked_dimensions=ranked,
        primary_feature=top_feat,
        primary_fisher=top_fisher,
    )


# ---------------------------------------------------------------------------
# Step-by-Step GLCM Calculation Demonstration Helpers
# ---------------------------------------------------------------------------

def extract_directional_pixel_pairs(
    quantized_patch: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    distance: int = 1,
    angle_deg: float = 0.0,
    levels: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Extract reference pixels, neighbor pixels, and raw GLCM count matrix
    for a single spatial displacement (distance, angle_deg).

    Returns
    -------
    ref_vals : 1D np.ndarray of reference pixel gray levels
    neigh_vals : 1D np.ndarray of neighbor pixel gray levels
    raw_counts : 2D np.ndarray of shape (levels, levels)
    valid_pair_count : int total valid pixel pairs
    """
    raw = _directed_counts(quantized_patch, valid_mask, distance, np.radians(angle_deg), levels)
    slices = pair_slices(quantized_patch.shape, distance, np.radians(angle_deg))
    blocks = list(_pair_chunks(quantized_patch, valid_mask, *slices))
    empty = np.array([], dtype=quantized_patch.dtype)
    left = np.concatenate([a for a, _ in blocks]) if blocks else empty
    right = np.concatenate([b for _, b in blocks]) if blocks else empty
    return left, right, raw, int(raw.sum())


def get_glcm_demonstration_data(
    patch: TexturePatch,
    distance: int = 1,
    angles_deg: Sequence[float] = (0, 45, 90, 135),
    levels: int = 5,
    max_sample_pairs: int = 20,
    matrix_display_size: int = 8,
) -> Dict[str, Any]:
    """
    Generate rigorous, verifiable step-by-step intermediate calculation data
    for a given TexturePatch across Steps 1 through 9 and Final Eigenvalues.

    Returns a structured dictionary containing all intermediate data arrays,
    matrices, tables, and explanations without fabricated values.
    """
    patch.validate_footprint()
    # ── Step 1: Extracted patch info
    p_img = patch.image_8bit
    h, w = p_img.shape[:2]
    step1 = {
        "patch_id": patch.patch_id,
        "label": patch.label,
        "x": patch.x,
        "y": patch.y,
        "width": patch.width,
        "height": patch.height,
        "dimensions_str": f"{patch.width} × {patch.height} px",
        "total_pixels": patch.total_pixels,
        "valid_pixels": patch.valid_pixel_count,
        "thumbnail": patch.extra.get("thumbnail"),
        "rgb_patch": patch.extra.get("rgb_patch"),
        "image_8bit": p_img,
    }

    # ── Step 2: Grayscale / Panchromatic patch
    pan_arr = patch.extra.get("pan_patch")
    if pan_arr is None:
        pan_arr = p_img
    pan_arr_np = np.asarray(pan_arr, dtype=np.float64)
    mask = patch.valid_mask if patch.valid_mask is not None else np.ones((h, w), dtype=bool)
    valid_pan = pan_arr_np[mask] if np.any(mask) else pan_arr_np.ravel()

    # Streamlit displays uint8 previews, while the numerical table retains native DN.
    display = np.zeros(pan_arr_np.shape, dtype=np.uint8)
    if valid_pan.size:
        lo, hi = np.nanmin(valid_pan), np.nanmax(valid_pan)
        vals = pan_arr_np[mask]
        if lo < 0 or hi > 255 or 0 < hi <= 1:
            vals = (vals - lo) * (255 / (hi - lo)) if hi > lo else np.zeros_like(vals)
        display[mask] = np.clip(vals, 0, 255).astype(np.uint8)
    step2 = {
        "pan_image": display,
        "mean": float(np.nanmean(valid_pan)) if valid_pan.size else patch.pan_mean,
        "std": float(np.nanstd(valid_pan)) if valid_pan.size else patch.pan_std,
        "min": float(np.nanmin(valid_pan)) if valid_pan.size else 0.0,
        "max": float(np.nanmax(valid_pan)) if valid_pan.size else 0.0,
        "sample_matrix": pan_arr_np[:matrix_display_size, :matrix_display_size],
        "display_size": (min(matrix_display_size, h), min(matrix_display_size, w)),
    }

    # ── Step 3: Histogram-equalized patch
    eq_arr_np = np.asarray(p_img, dtype=np.float64)
    valid_eq = eq_arr_np[mask] if np.any(mask) else eq_arr_np.ravel()
    step3 = {
        "eq_image": np.asarray(p_img, dtype=np.uint8),
        "mean": float(np.mean(valid_eq)) if valid_eq.size else patch.eq_mean,
        "std": float(np.std(valid_eq)) if valid_eq.size else patch.eq_std,
        "min": float(np.min(valid_eq)) if valid_eq.size else 0.0,
        "max": float(np.max(valid_eq)) if valid_eq.size else 255.0,
        "sample_matrix": eq_arr_np[:matrix_display_size, :matrix_display_size].astype(int),
        "display_size": (min(matrix_display_size, h), min(matrix_display_size, w)),
        "explanation": (
            "The PAN histogram of the confirmed subset builds one cumulative-distribution lookup table. "
            "Every valid patch pixel uses that same table."
        ),
    }

    # ── Step 4: 5-Level Quantization (0, 1, 2, 3, 4)
    q_arr = np.asarray(patch.quantized, dtype=np.uint8)
    quant_mode = patch.extra.get("quantization_mode", "global_equalized_fixed5")
    mapping_table = [
        {"range": "0 – 51", "level": 0, "description": "Lowest intensity / Deep dark tones / Shadow"},
        {"range": "52 – 102", "level": 1, "description": "Dark mid-tones / Dense canopy"},
        {"range": "103 – 153", "level": 2, "description": "Mid-tones / Intermediate land-cover"},
        {"range": "154 – 204", "level": 3, "description": "Bright mid-tones / Bare soil / Paved area"},
        {"range": "205 – 255", "level": 4, "description": "Highest intensity / Bright surfaces / High reflectance"},
    ]
    step4 = {
        "quantized_matrix": q_arr,
        "sample_matrix": q_arr[:matrix_display_size, :matrix_display_size],
        "levels": list(range(levels)),
        "unique_levels": patch.unique_levels if patch.unique_levels else sorted(list(np.unique(q_arr))),
        "mapping_table": mapping_table,
        "formula": "Q(x, y) = floor( 5 × I(x, y) / 256 )",
        "quantization_mode": quant_mode,
        "occupancy": patch.occupancy if patch.occupancy is not None else np.zeros(levels),
    }

    # ── Step 5: Directions & Spatial Offset
    direction_labels = {}
    for angle in angles_deg:
        dy, dx = pixel_offset(distance, np.radians(angle))
        direction_labels[angle] = f"{angle:g}° (Δcolumn={dx:+d}, Δrow={dy:+d})"
    step5 = {
        "distance": distance,
        "angles_deg": list(angles_deg),
        "direction_labels": {ang: direction_labels.get(int(ang), f"{ang}° direction") for ang in angles_deg},
    }

    # ── Steps 6, 7, 8, 9: Per-direction Breakdown
    directions_data = {}
    normalized_matrices = []

    # Analysis owns the count matrices. The demo uses the selected patch and offset.
    recorded = {(d["distance"], d["angle_deg"]): d["raw_counts"]
                for d in patch.extra.get("directional", [])}
    for ang in angles_deg:
        ang_f = float(ang)
        ang_int = int(round(ang_f))
        raw_counts = recorded.get((distance, ang_f))
        if raw_counts is None:  # Standalone demonstration of an unanalyzed patch or offset.
            raw_counts = _directed_counts(q_arr, mask, distance, np.radians(ang_f), levels)
        valid_pairs = int(raw_counts.sum())
        src, dst = pair_slices(q_arr.shape, distance, np.radians(ang_f))
        possible_pairs = int(q_arr[src].size)
        sample_pairs = []
        for left, right in _pair_chunks(q_arr, mask, src, dst):
            remaining = max_sample_pairs - len(sample_pairs)
            for i, j in zip(left[:remaining], right[:remaining]):
                sample_pairs.append({"Pair #": len(sample_pairs) + 1, "Reference Pixel": int(i),
                                     "Neighbor Pixel": int(j), "Gray-Level Pair (i, j)": f"({i}, {j})"})
            if len(sample_pairs) >= max_sample_pairs:
                break

        occurrences = []
        for index in np.argsort(-raw_counts.ravel(), kind="stable"):
            i, j = np.unravel_index(index, raw_counts.shape)
            count = int(raw_counts[i, j])
            if count:
                occurrences.append({"Pair (i, j)": f"({i}, {j})", "Occurrences": count,
                                    "Frequency %": f"{100 * count / valid_pairs:.2f}%"})

        # Step 8: Symmetrize
        sym_counts = raw_counts + raw_counts.T
        sym_pair_total = int(sym_counts.sum())

        # Step 9: Normalize
        norm_matrix = sym_counts / sym_counts.sum() if sym_counts.sum() > 0 else np.zeros((levels, levels))
        normalized_matrices.append(norm_matrix)

        directions_data[ang_int] = {
            "angle_deg": ang_f,
            "angle_label": direction_labels.get(ang_int, f"{ang_int}°"),
            "sample_pairs": sample_pairs,
            "top_pairs": occurrences[:10],  # Compatibility for existing callers.
            "occurrences": occurrences,
            "possible_pairs": possible_pairs,
            "raw_counts": raw_counts,
            "valid_pairs": valid_pairs,
            "symmetric_counts": sym_counts,
            "symmetric_pair_total": sym_pair_total,
            "normalized_matrix": norm_matrix,
            "normalized_sum": float(norm_matrix.sum()),
        }

    # Multi-direction Averaging (if multiple angles)
    if len(normalized_matrices) > 1:
        averaged_glcm = np.mean(np.stack(normalized_matrices, axis=0), axis=0)
        averaged_glcm /= averaged_glcm.sum()
    elif normalized_matrices:
        averaged_glcm = normalized_matrices[0]
    else:
        averaged_glcm = np.zeros((levels, levels))

    # Final Eigenvalues (Requirement 5)
    final_eigenvalues = compute_eigenvalues(averaged_glcm, check_symmetry=True)

    return {
        "step1": step1,
        "step2": step2,
        "step3": step3,
        "step4": step4,
        "step5": step5,
        "directions": directions_data,
        "averaged_glcm": averaged_glcm,
        "averaged_glcm_sum": float(averaged_glcm.sum()),
        "final_eigenvalues": final_eigenvalues,
        "eigenvalue_trace": float(np.sum(final_eigenvalues)),
    }
