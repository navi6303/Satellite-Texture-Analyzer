"""Configuration, sample extraction, and model fitting for both entry points."""

from dataclasses import dataclass, field
from hashlib import sha256

from modules.glcm_analysis import analyze_patch_glcm, compute_glcm, compute_eigenvalues
from modules.patch_sampling import assign_groups, sample_layout
from modules.pipeline import analyze_all_patches, run_separability_and_cv
from modules.neighbors import require_valid_pairs
from modules.scene import read_patch
from modules.texture_extraction import (
    QUANTIZATION_MODES,
    quantize_to_5_levels,
    QUANT_GLOBAL_EQ_FIXED5,
)

LABELS = ["Water Bodies", "Vegetation", "Urban / Built-up Area", "Bare Soil"]


@dataclass
class Settings:
    n: int = 5
    crop_size: int = 64
    block_size: int = 256
    seed: int = 42
    mode: str = QUANT_GLOBAL_EQ_FIXED5
    method: str = "percentile"
    angles: list = field(default_factory=lambda: [0, 45, 90, 135])
    distance: int = 1
    transition: bool = True
    exact_boxes: bool = True

    def validate(self):
        if (
            self.distance not in (1, 2, 3)
            or not self.angles
            or len(set(self.angles)) != len(self.angles)
            or not set(self.angles) <= {0, 45, 90, 135}
        ):
            raise ValueError(
                "Choose at least one unique direction and a distance of 1, 2 or 3."
            )
        if self.mode not in QUANTIZATION_MODES or self.method not in (
            "minmax",
            "percentile",
        ):
            raise ValueError("Unknown quantization mode or rescale method.")


def extract_samples(scene, settings, explicit, regions, rgb_bands, labels=None):
    if labels is None:
        labels = LABELS
    settings.validate()
    raster = scene.raster
    layout = sample_layout(
        labels,
        explicit,
        regions,
        settings.n,
        settings.crop_size,
        settings.block_size,
        settings.seed,
        raster.width,
        raster.height,
        exact=settings.exact_boxes,
    )
    patches = []
    for i, label in enumerate(labels, 1):
        for j, box in enumerate(layout[label], 1):
            # Preserve the original four basenames for the first sample of each class.
            name = f"patch_{i}" if j == 1 else f"patch_{i}_{j:02d}"
            patches.append(
                read_patch(
                    scene, box, label, name, settings.mode, settings.method, rgb_bands,
                    distance=settings.distance, angles=settings.angles
                )
            )
    assign_groups(patches, settings.block_size, raster.width)
    return patches


def analyze_samples(patches, settings):
    settings.validate()
    for p in patches:
        require_valid_pairs(p.valid_mask, settings.distance, settings.angles, p.patch_id)
    analyze_all_patches(
        patches, [settings.distance], settings.angles, settings.transition
    )
    for p in patches:
        if p.horizontal_valid_pairs:
            q, _ = quantize_to_5_levels(p.image_8bit, p.valid_mask)
            baseline, _ = compute_glcm(q, p.valid_mask, distance=1, angle_rad=0)
            p.extra["baseline"] = {
                "Patch ID": p.patch_id,
                "Class": p.label,
                "Global fixed, 0° λ1": float(compute_eigenvalues(baseline)[0]),
                "Largest diagonal": float(baseline.diagonal().max()),
                "Trace": float(baseline.trace()),
                "Sum": float(baseline.sum()),
            }
    return run_separability_and_cv(
        patches,
        [settings.distance],
        settings.angles,
        settings.mode,
        settings.transition,
    )


def audit_regions(scene, regions, rgb_bands):
    """Independent legacy-scale reference (fixed global bins, d=1 horizontal). Never fitted."""
    rows = []
    boxes = ((label, box) for label, value in regions.items()
             for box in (value if isinstance(value, list) else [value]))
    for label, box in boxes:
        p = read_patch(
            scene,
            box,
            label,
            "original",
            QUANT_GLOBAL_EQ_FIXED5,
            "percentile",
            rgb_bands,
        )
        analyze_patch_glcm(p, compute_transition=False)
        rows.append(
            dict(
                label=label,
                **box,
                equalized_mean=p.eq_mean,
                equalized_std=p.eq_std,
                scene_median=scene.median,
                occupancy=p.occupancy.tolist(),
                lambda1=p.eigenvalues[0],
                largest_diagonal=p.glcm.diagonal().max(),
                diagonal_share=p.glcm.trace(),
                warnings=p.sanity_warnings,
            )
        )
    return rows


def file_digest(path):
    with open(path, "rb") as f:
        digest = sha256()
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def sampled_confirmed_boxes(confirmed, settings, width, height):
    """Distribute the requested samples across every confirmed class region."""
    result = {}
    for class_idx, (label, boxes) in enumerate(confirmed.items()):
        n = max(settings.n, len(boxes))
        result[label] = []
        for j, box in enumerate(boxes):
            count = n // len(boxes) + (j < n % len(boxes))
            layout = sample_layout([label], {}, {label: box}, count, settings.crop_size,
                                   settings.block_size, settings.seed + class_idx + 4*j,
                                   width, height, minimum_n=1)
            result[label].extend(layout[label])
    return result
