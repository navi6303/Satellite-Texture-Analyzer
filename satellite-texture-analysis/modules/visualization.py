"""
Visualization Module for Satellite Texture Analysis.
Implements:
  1. Five individual Matplotlib figures (one for each eigenvalue across all 4 classes).
  2. Before/After histogram equalization plots.
  3. Labeled 5x5 GLCM heatmaps.
  4. Combined multi-feature comparison charts and distance heatmaps.
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server/Streamlit reliability
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from modules.texture_extraction import TexturePatch
import cv2

# Consistent, accessible academic color palette — matches LAND_COVER_CATEGORIES order:
# Water Bodies, Vegetation, Urban / Built-up Area, Bare Soil
CLASS_COLORS = ["#00b4d8", "#2dc653", "#e63946", "#fca311"]
CLASS_COLOR_MAP = {
    "Water Bodies": "#00b4d8",
    "Vegetation": "#2dc653",
    "Urban / Built-up Area": "#e63946",
    "Bare Soil": "#fca311",
}
ACCENT_COLOR = "#1d3557"
GRID_COLOR = "#e0e0e0"
LABEL_OFFSET_POINTS = 3


def _color_for_label(label: str, idx: int = 0) -> str:
    return CLASS_COLOR_MAP.get(label, CLASS_COLORS[idx % len(CLASS_COLORS)])


def _class_mean_std(
    patches: List[TexturePatch], getter
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Aggregate a scalar per patch into class mean and sample std."""
    order: List[str] = []
    buckets: Dict[str, List[float]] = {}
    for p in patches:
        if p.label not in buckets:
            order.append(p.label)
            buckets[p.label] = []
        buckets[p.label].append(float(getter(p)))
    means = np.array([np.mean(buckets[k]) for k in order], dtype=np.float64)
    stds = np.array(
        [(np.std(buckets[k], ddof=1) if len(buckets[k]) > 1 else 0.0) for k in order],
        dtype=np.float64,
    )
    return order, means, stds


def _apply_nonneg_ylim(
    ax, values: np.ndarray, errs: Optional[np.ndarray] = None, signed_samples=False
) -> None:
    """Y-axis starts at 0 for non-negative data (fixes Figure 5 negative range)."""
    arr = np.asarray(values, dtype=np.float64)
    top = arr.copy()
    if errs is not None:
        top = top + np.asarray(errs, dtype=np.float64)
    vmax = float(np.max(top)) if top.size else 0.0
    vmin = float(np.min(arr)) if arr.size else 0.0
    if vmin >= -1e-12 and not signed_samples:
        ax.set_ylim(0.0, vmax * 1.22 if vmax > 0 else 1.0)
    else:
        if errs is not None:
            vmin = min(vmin, float(np.min(arr - errs)))
        vmax = max(0.0, vmax)
        span = max(vmax - vmin, 1e-6)
        ax.set_ylim(vmin - 0.08 * span, vmax + 0.12 * span)


def _annotate_bars(
    ax,
    bars,
    values,
    errors=None,
    offset_points: int = LABEL_OFFSET_POINTS,
    compact=False,
) -> None:
    """Place value labels at a fixed small offset above each bar (not in data units)."""
    errors = np.zeros(len(values)) if errors is None else np.nan_to_num(errors)
    for bar, val, err in zip(bars, values, errors):
        ax.annotate(
            f"{float(val):.3f}" if compact else f"{float(val):.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, val + err),
            xytext=(0, offset_points),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.3 if compact else 8.0,
            rotation=0,
            fontweight="bold",
            color="#1b263b",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.1),
            clip_on=False,
        )


def downsample_for_display(image: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """
    Downsample a 2D or 3D image to a max dimension for instant Streamlit UI rendering.
    Avoids transmitting 80MB+ full-scene images over websockets.
    """
    if image is None or image.size == 0:
        return image
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = float(max_dim) / float(max(h, w))
    new_w = max(32, int(w * scale))
    new_h = max(32, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def figure_to_bytes(fig: plt.Figure, dpi: int = 200, format: str = "png") -> bytes:
    """Serialize a Matplotlib Figure into an in-memory byte buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


EIGEN_LABELS = [f"λ{i}" for i in range(1, 6)]
PATCHES_PER_CHART = 3
PATCH_COLORS = ["#176998", "#ce7819", "#478263"]


def _groups(patches):
    groups = {}
    for patch in patches:
        if patch.eigenvalues is not None:
            groups.setdefault(patch.label, []).append(patch)
    if not groups:
        raise ValueError("Analyze at least one patch before plotting eigenvalues.")
    return groups


def _bar_axes(ax, ylabel="Eigenvalue"):
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#dce2e6", linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="#586774", linewidth=.7)


def _finish_chart(fig, caption):
    fig.text(.5, .025, caption, ha="center", va="bottom", fontsize=9, color="#344653")
    fig.tight_layout(rect=(0, .09, 1, 1))
    return fig, figure_to_bytes(fig)


def plot_single_eigenvalue(patches, eig_index=0, title=None):
    if not 0 <= eig_index < 5:
        raise ValueError("Choose an eigenvalue index from 0 to 4.")
    groups = _groups(patches)
    labels, means, stds = _class_mean_std(
        [p for members in groups.values() for p in members], lambda p: p.eigenvalues[eig_index]
    )
    fig, ax = plt.subplots(figsize=(8, 4.1))
    for i, (label, mean, sd) in enumerate(zip(labels, means, stds)):
        bars = ax.bar(i, mean, width=.58, color=_color_for_label(label, i),
                      yerr=sd if len(groups[label]) > 1 else None, capsize=3)
        _annotate_bars(ax, bars, [mean], [sd])
    ax.set_xticks(range(len(labels)), labels)
    ax.set_title(title or f"Mean {EIGEN_LABELS[eig_index]} by texture class")
    _bar_axes(ax)
    _apply_nonneg_ylim(ax, means, stds, any(p.eigenvalues[eig_index] < 0 for p in patches if p.eigenvalues is not None))
    return _finish_chart(fig, "Each bar is a class mean. Whiskers show one sample standard deviation across its patches.")


def plot_all_five_eigenvalues(patches):
    return [(i+1, *plot_single_eigenvalue(patches, i)) for i in range(5)]


def plot_combined_eigenvalue_comparison(patches):
    groups = _groups(patches)
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    x = np.arange(5)
    width = .82 / len(groups)
    means, errors = [], []
    for i, (label, group) in enumerate(groups.items()):
        values = np.stack([p.eigenvalues for p in group])
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1) if len(group) > 1 else np.zeros(5)
        offset = (i - (len(groups) - 1) / 2) * width
        ax.bar(x + offset, mean, width=.92 * width, label=label, color=_color_for_label(label, i),
               yerr=sd if len(group) > 1 else None, capsize=2, error_kw={"elinewidth": .8})
        means.extend(mean)
        errors.extend(sd)
    single = len(groups) == 1
    ax.set_title(f"Mean eigenvalues for {next(iter(groups))}" if single else "Mean eigenvalues by texture class")
    ax.set_xticks(x, EIGEN_LABELS)
    ax.set_xlabel("Eigenvalue index")
    if not single:
        ax.legend(frameon=False, ncol=2, fontsize=9)
    _bar_axes(ax)
    _apply_nonneg_ylim(ax, np.array(means), np.array(errors),
                       any(np.min(p.eigenvalues) < 0 for group in groups.values() for p in group))
    return _finish_chart(fig, "Bars average all patches in each class. Whiskers show how much the patches vary (one sample SD).")


def plot_individual_class_eigenvalues(patches, page=0):
    """Display up to three measured patches per class and keep the class scale across pages."""
    groups = _groups(patches)
    start = page * PATCHES_PER_CHART
    visible = [(name, group) for name, group in groups.items() if start < len(group)]
    if page < 0 or not visible:
        raise ValueError("No patches on this chart page.")
    cols = min(2, len(visible))
    rows = (len(visible) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, squeeze=False,
                             figsize=(8.2 if cols == 1 else 10, 3.7 * rows + .4))
    x = np.arange(5)
    for ax, (name, group) in zip(axes.flat, visible):
        shown = group[start:start+PATCHES_PER_CHART]
        width = .78 / len(shown)
        for i, patch in enumerate(shown):
            offset = (i - (len(shown) - 1) / 2) * width
            ax.bar(x + offset, patch.eigenvalues, width=width*.92, color=PATCH_COLORS[i], label=patch.patch_id)
        ax.set_xticks(x, EIGEN_LABELS)
        ax.set_xlabel("Eigenvalue index")
        ax.set_title(name)
        ax.legend(frameon=False, fontsize=8, ncol=len(shown))
        _bar_axes(ax)
        # Keep the same vertical scale across pages of a class.
        _apply_nonneg_ylim(ax, np.stack([p.eigenvalues for p in group]))
    for ax in list(axes.flat)[len(visible):]:
        ax.set_visible(False)
    return _finish_chart(fig, "Each color shows one patch. Compare bars at the same eigenvalue to see variation within a class.")


def plot_combined_eigenvalue_spectra(patches):
    """Legacy figure entry point for the grouped class-mean bars."""
    return plot_combined_eigenvalue_comparison(patches)


def separation_pairs(distances):
    """Read each unordered class pair once, in ascending distance order."""
    return sorted(((left, right, float(distances.loc[left, right]))
                   for i, left in enumerate(distances.index)
                   for right in distances.index[i+1:]), key=lambda item: item[2])


def plot_class_separation(distances):
    pairs = separation_pairs(distances)
    fig, ax = plt.subplots(figsize=(8.8, 4))
    y = np.arange(len(pairs))
    values = [v for _, _, v in pairs]
    bars = ax.barh(y, values, height=.58, color="#337491")
    ax.set_yticks(y, [f"{left} vs {right}" for left, right, _ in pairs], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, (max(values, default=0) or 1)*1.17)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    ax.set_title("Texture distances from combined class GLCMs")
    ax.set_xlabel("Euclidean distance between class eigenvalue vectors")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x", color="#dce2e6", linewidth=.7)
    ax.set_axisbelow(True)
    if not pairs:
        caption = "At least two texture classes are needed to compare distances."
    elif len(pairs) == 1:
        left, right, distance = pairs[0]
        caption = f"Only pair: {left} vs {right} ({distance:.4f})"
    else:
        closest_left, closest_right, smallest = pairs[0]
        farthest_left, farthest_right, greatest = pairs[-1]
        caption = (
            f"Greatest distance: {farthest_left} vs {farthest_right} ({greatest:.4f})\n"
            f"Smallest distance: {closest_left} vs {closest_right} ({smallest:.4f})"
        )
    return _finish_chart(fig, caption)


def plot_glcm_heatmap(
    glcm: np.ndarray, title: str = "5x5 GLCM Heatmap", cmap: str = "Blues"
) -> Tuple[plt.Figure, bytes]:
    """
    Generate an annotated heatmap of a 5x5 GLCM matrix with axes labeled 0..4.
    """
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=150)

    im = ax.imshow(glcm, cmap=cmap, vmin=0.0, vmax=max(0.01, float(np.max(glcm))))

    # Annotate cells
    for r in range(5):
        for c in range(5):
            val = glcm[r, c]
            text_color = "white" if val > 0.4 * np.max(glcm) else "#1b263b"
            ax.text(
                c,
                r,
                f"{val:.4f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.5,
                fontweight="medium",
            )

    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(["0", "1", "2", "3", "4"], fontsize=9.5, fontweight="bold")
    ax.set_yticklabels(["0", "1", "2", "3", "4"], fontsize=9.5, fontweight="bold")

    ax.set_xlabel("Neighbor Gray Level (j)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Reference Gray Level (i)", fontsize=9.5, fontweight="bold")
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    png_bytes = figure_to_bytes(fig)
    return fig, png_bytes


def plot_quantized_patch_histogram(patch: TexturePatch) -> Tuple[plt.Figure, bytes]:
    """Count valid pixels at each of the five levels in the GLCM input patch."""
    levels = np.arange(5)
    counts = np.bincount(patch.quantized[patch.valid_mask], minlength=len(levels))

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    bars = ax.bar(levels, counts, color=_color_for_label(patch.label))
    ax.bar_label(bars, labels=[f"{count:,}" for count in counts], padding=4)
    ax.set_title(f"{patch.label} · {patch.patch_id}")
    ax.set_xticks(levels)
    ax.set_xlabel("Gray level")
    ax.set_ylabel("Number of pixels")
    ax.set_ylim(0, max(1, int(counts.max())) * 1.2)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return fig, figure_to_bytes(fig)


def plot_equalization_histograms(
    hist_before: np.ndarray, hist_after: np.ndarray
) -> Tuple[plt.Figure, bytes]:
    """
    Plot intensity histograms before and after histogram equalization side by side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    bins = np.arange(256)

    ax1.bar(bins, hist_before, width=1.0, color="#457b9d", edgecolor="none", alpha=0.9)
    ax1.set_title(
        "Before Equalization (Panchromatic)", fontsize=10.5, fontweight="bold"
    )
    ax1.set_xlabel("8-bit Gray Level [0 - 255]", fontsize=9)
    ax1.set_ylabel("Pixel Frequency", fontsize=9)
    ax1.set_xlim(0, 255)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.bar(bins, hist_after, width=1.0, color="#2a9d8f", edgecolor="none", alpha=0.9)
    ax2.set_title(
        "After Global Histogram Equalization", fontsize=10.5, fontweight="bold"
    )
    ax2.set_xlabel("8-bit Gray Level [0 - 255]", fontsize=9)
    ax2.set_ylabel("Pixel Frequency", fontsize=9)
    ax2.set_xlim(0, 255)
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    png_bytes = figure_to_bytes(fig)
    return fig, png_bytes


def plot_distance_heatmap(df_dist: pd.DataFrame) -> Tuple[plt.Figure, bytes]:
    """
    Render a pairwise Euclidean distance matrix as a labeled heatmap.
    """
    labels = df_dist.index.tolist()
    matrix = df_dist.values
    n = len(labels)

    fig, ax = plt.subplots(figsize=(5.5, 4.6), dpi=150)
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=float(np.max(matrix)))

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            text_color = "white" if val > 0.5 * np.max(matrix) else "#1b263b"
            ax.text(
                j,
                i,
                f"{val:.4f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.5,
                fontweight="bold",
            )

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9, fontweight="medium")
    ax.set_yticklabels(labels, fontsize=9, fontweight="medium")

    ax.set_title(
        "Pairwise Euclidean Distance in Feature Space",
        fontsize=10.5,
        fontweight="bold",
        pad=12,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    png_bytes = figure_to_bytes(fig)
    return fig, png_bytes


def plot_classic_features(patches: List[TexturePatch]) -> Tuple[plt.Figure, bytes]:
    """Per-feature axes preserve the range of correlation and the units of entropy."""
    from modules.glcm_analysis import CLASSIC_COLUMNS

    display = [
        "Contrast",
        "Dissimilarity",
        "Homogeneity",
        "ASM",
        "Energy (√ASM)",
        "Entropy (bits)",
        "Correlation",
        "GLCM mean",
        "GLCM variance",
        "Maximum probability",
        "Trace",
        "Diagonal share",
    ]
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    for key, name, ax in zip(CLASSIC_COLUMNS, display, axes.flat):
        labs, values, sd = _class_mean_std(patches, lambda p: p.classic_features[key])
        for i, (lab, val, err) in enumerate(zip(labs, values, sd)):
            error = err if sum(p.label == lab for p in patches) > 1 else None
            bars = ax.bar(i, val, yerr=error, capsize=3, color=_color_for_label(lab, i))
            _annotate_bars(ax, bars, [val], errors=[err], compact=True)
        ax.set_xticks(range(len(labs)), [lab.split()[0] for lab in labs], fontsize=8)
        ax.set_title(name, fontsize=10)
        _apply_nonneg_ylim(
            ax, values, sd, any(p.classic_features[key] < 0 for p in patches)
        )
    fig.suptitle("Texture features of the normalized GLCM P", fontweight="bold")
    fig.tight_layout()
    return fig, figure_to_bytes(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
) -> Tuple[plt.Figure, bytes]:
    fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=150)
    cm = np.asarray(cm, dtype=np.float64)
    im = ax.imshow(cm, cmap="Blues", vmin=0.0)
    vmax = float(np.max(cm)) if cm.size else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j,
                i,
                f"{int(val)}",
                ha="center",
                va="center",
                color="white" if vmax and val > 0.5 * vmax else "#1b263b",
                fontsize=10,
                fontweight="bold",
            )
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=25, ha="right", fontsize=8.5)
    ax.set_yticklabels(class_names, fontsize=8.5)
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("True", fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, figure_to_bytes(fig)


def plot_patch_map_overlay(
    base_rgb: np.ndarray,
    patches: List[TexturePatch],
    orig_w: int,
    orig_h: int,
) -> Tuple[plt.Figure, bytes]:
    """Scene preview with every patch box overlaid."""
    img = np.asarray(base_rgb)
    labels = list(dict.fromkeys(p.label for p in patches))
    fig = plt.figure(figsize=(12, 7.6), layout="constrained")
    grid = fig.add_gridspec(2, 4, width_ratios=[1.35, 1.35, 1, 1])
    overview = fig.add_subplot(grid[:, :2])
    extent = [0, orig_w, orig_h, 0]
    overview.imshow(img, extent=extent)
    overview.set_title("Scene and sampling locations")
    overview.set_xlabel("Column (pixels)")
    overview.set_ylabel("Row (pixels)")
    for i, lab in enumerate(labels[:4]):
        group = [p for p in patches if p.label == lab]
        color = _color_for_label(lab)
        ax = fig.add_subplot(grid[i // 2, 2 + i % 2])
        ax.imshow(img, extent=extent)
        xs = [p.x for p in group] + [p.x + p.width for p in group]
        ys = [p.y for p in group] + [p.y + p.height for p in group]
        pad = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.16
        ax.set_xlim(max(0, min(xs) - pad), min(orig_w, max(xs) + pad))
        ax.set_ylim(min(orig_h, max(ys) + pad), max(0, min(ys) - pad))
        for j, p in enumerate(group, 1):
            for target in (overview, ax):
                target.add_patch(
                    Rectangle(
                        (p.x, p.y),
                        p.width,
                        p.height,
                        fill=False,
                        edgecolor=color,
                        lw=1.4,
                    )
                )
            ax.annotate(
                str(j),
                (p.x, p.y),
                xytext=(1, 2),
                textcoords="offset points",
                color="white",
                fontsize=8,
                bbox=dict(facecolor="#1b263b", edgecolor="none", alpha=0.8, pad=1),
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(lab, fontsize=10)
    return fig, figure_to_bytes(fig)


def plot_thumbnails(patches):
    """Show the observed colour and equalized structure for each labeled sample."""
    labels = list(dict.fromkeys(p.label for p in patches))
    groups = [[p for p in patches if p.label == label] for label in labels]
    n = min(8, max(map(len, groups)))
    rows = [
        (label, group[start : start + n])
        for label, group in zip(labels, groups)
        for start in range(0, len(group), n)
    ]
    fig, axes = plt.subplots(
        len(rows), n, figsize=(max(5, 1.65 * n), 1.9 * len(rows)), squeeze=False
    )
    for i, (label, group) in enumerate(rows):
        for j in range(n):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if j < len(group):
                p = group[j]
                ax.imshow(
                    p.extra.get("thumbnail", p.image_8bit),
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                )
                ax.set_title(
                    p.patch_id + (" *" if p.sanity_warnings else ""), fontsize=9
                )
            if j == 0:
                ax.set_ylabel(label.replace(" / ", "\n"), fontsize=9)
    fig.suptitle("Patch thumbnails   ·   * sanity warning", fontsize=12)
    fig.tight_layout()
    return fig, figure_to_bytes(fig)
