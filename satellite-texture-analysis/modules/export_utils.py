"""
Export and Reporting Utility Module.
Handles:
  1. Serializing images, CSV data tables, and figures.
  2. Generating academic Markdown reports from computed results only.
  3. Packaging all analysis outputs into a self-contained ZIP archive.
"""

from __future__ import annotations

import importlib.metadata
import io
import json
import zipfile
import platform
import tempfile
from pathlib import Path
from dataclasses import asdict
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image

from modules.reporting import generate_markdown_report, diagnostics, feature_summary
from modules.validation import cv_tables
from modules.neighbors import MIN_NEIGHBOR_PAIRS
from modules.matrix_audit import PCA_FEATURES, matrix_audit, directional_covariance, directional_table
from modules.visualization import (
    plot_classic_features,
    plot_combined_eigenvalue_comparison,
    plot_confusion_matrix,
    plot_equalization_histograms,
    plot_patch_map_overlay,
)


PACKAGE_VERSION_NAMES = [
    "numpy",
    "pandas",
    "matplotlib",
    "opencv-python",
    "opencv-python-headless",
    "scipy",
    "scikit-image",
    "rasterio",
    "pillow",
    "streamlit",
    "scikit-learn",
    "pytest",
]


def collect_package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in PACKAGE_VERSION_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            alt = {
                "opencv-python": "cv2",
                "scikit-image": "skimage",
                "pillow": "PIL",
                "scikit-learn": "sklearn",
            }
            versions[name] = "not-installed"
            if name in alt:
                try:
                    mod = __import__(alt[name])
                    versions[name] = getattr(mod, "__version__", "present")
                except Exception:
                    pass
    return versions


def array_to_png_bytes(array: np.ndarray) -> bytes:
    """Convert a 2D uint8 grayscale or 3D uint8 RGB array into PNG bytes."""
    buf = io.BytesIO()
    if array.ndim == 2:
        img = Image.fromarray(np.asarray(array, dtype=np.uint8))
    elif array.ndim == 3 and array.shape[2] == 3:
        img = Image.fromarray(np.asarray(array, dtype=np.uint8))
    elif array.ndim == 3 and array.shape[2] == 4:
        img = Image.fromarray(np.asarray(array, dtype=np.uint8))
    else:
        raise ValueError(f"Unsupported array shape for image export: {array.shape}")

    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def glcm_to_csv_string(glcm: np.ndarray, patch_label: str) -> str:
    """Convert a 5x5 GLCM into a clean CSV string with labeled axes."""
    header = ["Reference_Level", "Level_0", "Level_1", "Level_2", "Level_3", "Level_4"]
    lines = [f"# GLCM for Texture Class: {patch_label}", ",".join(header)]
    for r in range(5):
        row_vals = [f"Level_{r}"] + [f"{glcm[r, c]:.17g}" for c in range(5)]
        lines.append(",".join(row_vals))
    return "\n".join(lines)


def write_products(
    folder,
    raster,
    selected_bands,
    patches,
    a,
    pan_stats,
    hist_before,
    hist_after,
    preview,
    options,
    source_metadata=None,
):
    """Write small products sequentially. Close each figure before rendering the next."""
    from modules.visualization import (
        plot_single_eigenvalue,
        plot_thumbnails,
        plot_individual_class_eigenvalues,
        plot_class_separation, PATCHES_PER_CHART,
    )

    for name in ("plots", "tables", "patches", "glcm_matrices", "images"):
        (folder / name).mkdir(exist_ok=True, parents=True)

    def chart(name, plot):
        fig, png = plot
        (folder / "plots" / name).write_bytes(png)
        plt.close(fig)

    for p in patches:
        slug = (
            p.label.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )
        base = f"{p.patch_id}_{slug}"
        (folder / "patches" / (base + "_8bit.png")).write_bytes(
            array_to_png_bytes(p.image_8bit)
        )
        (folder / "patches" / (base + "_quantized_5levels.png")).write_bytes(
            array_to_png_bytes(p.quantized * 63)
        )
        (folder / "glcm_matrices" / (base + "_glcm.csv")).write_text(
            glcm_to_csv_string(p.glcm, p.label), encoding="utf8"
        )
        if not p.valid_mask.all():
            (folder / "patches" / (base + "_valid_mask.png")).write_bytes(
                array_to_png_bytes(p.valid_mask * 255)
            )

    csvs = {
        "eigenvalues_summary": ("df_eigenvalues", False),
        "pairwise_distances": ("df_distances", True),
        "pairwise_distances_standardized": ("df_dist_std", True),
        "classic_features": ("df_classic", False),
        "eigenvalues_class_mean_std": ("df_eigen_mean", False),
        "fisher_ratios": ("df_fisher", False),
        "combined_distances_raw": ("df_dist_raw_combined", True),
        "combined_distances_standardized": ("df_dist_std_combined", True),
    }
    for filename, (key, index) in csvs.items():
        if a.get(key) is not None:
            a[key].to_csv(folder / "tables" / (filename + ".csv"), index=index)
    baseline = [p.extra["baseline"] for p in patches if "baseline" in p.extra]
    if baseline:
        pd.DataFrame(baseline).to_csv(
            folder / "tables/same_box_global_horizontal.csv", index=False
        )
    audit_table = matrix_audit(patches)
    audit_table.to_csv(folder / "tables/matrix_audit.csv", index=False)
    records = directional_table(patches)
    if not records.empty:
        records.to_csv(folder / "tables/directional_glcms.csv", index=False)
    pca_table = directional_covariance(patches)
    if not pca_table.empty:
        pca_table.to_csv(folder / "tables/directional_feature_pca.csv", index=False)
    diagnostics(patches).to_csv(folder / "tables/patch_sanity.csv", index=False)
    feature_summary(patches).to_csv(
        folder / "tables/features_class_mean_std.csv", index=False
    )
    for k in range(5):
        chart(f"figure_{k + 1}_lambda{k + 1}.png", plot_single_eigenvalue(patches, k))
    chart("combined_feature_spectrum.png", plot_combined_eigenvalue_comparison(patches))
    chart("individual_class_eigenvalues.png", plot_individual_class_eigenvalues(patches))
    # Retain both historical names for the same four-class mean figure.
    (folder / "plots/combined_eigenvalue_spectra.png").write_bytes((folder / "plots/combined_feature_spectrum.png").read_bytes())
    class_sizes = pd.Series([p.label for p in patches]).value_counts()
    for start in range(PATCHES_PER_CHART, int(class_sizes.max()), PATCHES_PER_CHART):
        page = start // PATCHES_PER_CHART
        chart(f"individual_class_eigenvalues_page_{page+1}.png", plot_individual_class_eigenvalues(patches, page))
    if a.get("df_class_glcm_distances") is not None:
        chart("class_separation.png", plot_class_separation(a["df_class_glcm_distances"]))
    chart("classic_features.png", plot_classic_features(patches))
    chart("patch_thumbnails.png", plot_thumbnails(patches))
    if preview is not None:
        chart(
            "patch_map_overlay.png",
            plot_patch_map_overlay(preview, patches, raster.width, raster.height),
        )
    if hist_before is not None and hist_after is not None:
        chart(
            "equalization_histograms.png",
            plot_equalization_histograms(hist_before, hist_after),
        )
    cv = a.get("cv_results") or {}
    if cv:
        tables = cv_tables(cv)
        results = tables["summary"]
        if results.empty:
            results = pd.DataFrame(
                [
                    dict(feature_set=name, status="unavailable", reason=ev["note"])
                    for name, ev in cv["by_set"].items()
                ]
            )
        results.to_csv(folder / "tables/cv_summary.csv", index=False)
        for name, matrix in tables["confusion"].items():
            matrix.to_csv(
                folder / "tables" / f"confusion_{name}.csv", index_label="actual"
            )
        preds = [
            dict(feature_set=k, **row)
            for k, ev in cv["by_set"].items()
            for row in ev.get("predictions", [])
        ]
        if preds:
            pd.DataFrame(preds).to_csv(
                folder / "tables/cv_predictions.csv", index=False
            )
        if cv.get("best_feature_set"):
            feature, clf = cv["best_feature_set"].split(" / ")
            score = cv["by_set"][feature][clf]
            chart(
                "confusion_matrix.png",
                plot_confusion_matrix(
                    score["confusion_matrix"],
                    score["class_names"],
                    "Blocked CV · " + cv["best_feature_set"],
                ),
            )
    report_options = dict(options)
    for key in ("df_fisher", "df_dist_std", "df_class_glcm_distances", "cv_results", "check_bundle"):
        report_options[key] = a.get(key)
    report = generate_markdown_report(
        raster,
        selected_bands,
        pan_stats,
        {},
        patches,
        a["df_eigenvalues"],
        a["df_distances"],
        a["interpretation"],
        **report_options,
    )
    (folder / "analysis_report.md").write_text(report, encoding="utf8")
    coords = [
        dict(
            patch_id=p.patch_id,
            label=p.label,
            x=p.x,
            y=p.y,
            original_x=p.x + raster.metadata.extra.get("source_window", {}).get("x", 0),
            original_y=p.y + raster.metadata.extra.get("source_window", {}).get("y", 0),
            width=p.width,
            height=p.height,
            block_id=p.block_id,
            grid_cells=p.extra.get("grid_cells"),
            pair_counts=p.extra.get("pair_counts"),
            d1_horizontal_pairs=p.horizontal_valid_pairs,
            occupancy=p.occupancy.tolist(),
            sanity_warnings=p.sanity_warnings,
        )
        for p in patches
    ]
    mode = options.get("quantization_mode", "global_equalized_fixed5")
    local_method = (
        options.get("rescale_method", "percentile")
        if mode == "per_patch_rescale_5bin"
        else None
    )
    meta = dict(
        project="Satellite Texture Analyzer",
        timestamp=datetime.now().astimezone().isoformat(),
        source_filename=raster.metadata.filename,
        dimensions=dict(width=raster.width, height=raster.height),
        selected_bands=selected_bands,
        band_index_base=0,
        crs=raster.metadata.crs,
        bounds=list(raster.metadata.bounds),
        pixel_size=list(raster.metadata.pixel_size),
        native_pan_statistics=pan_stats,
        package_versions=collect_package_versions(),
        python=platform.python_version(),
        patch_coordinates=coords,
        patch_id_to_class={p.patch_id: p.label for p in patches},
        matrix_audit=audit_table.to_dict("records"),
        coordinate_convention="zero-based native pixel (x=column,y=row), width and height, right/bottom exclusive",
        selection_source="complete user-specified windows"
        if options.get("settings", {}).get("exact_boxes")
        else "coordinates recorded per patch",
        eigenvalue_pipeline="A: signed eigvalsh of mean normalized P, descending absolute magnitude, positive first for ±Perron ties",
        directional_pca=dict(
            pipeline="B: sample covariance of directional features per box",
            features=PCA_FEATURES,
            standardize=False,
            ddof=1,
            observations="selected offsets",
        ),
        feature_definitions=dict(
            energy="sqrt(ASM)",
            asm="sum(P**2)",
            entropy="Shannon bits, zero bins omitted",
            correlation_constant_convention=1,
        ),
        glcm_parameters=dict(
            levels=5,
            minimum_directed_pairs_per_offset=MIN_NEIGHBOR_PAIRS,
            symmetric=True,
            normed=True,
            distances=options.get("distances", [1]),
            angles_deg=options.get("angles_deg", [0]),
            aggregation="unweighted mean of normalized GLCMs",
        ),
        quantization=dict(
            mode=mode,
            rescale_method=local_method,
            percentiles=[2, 98] if local_method == "percentile" else None,
            shared_across_boxes=local_method is None,
            formula="floor(5 * input / 256)",
            bin_edges_half_open=[0, 52, 103, 154, 205, 256]
            if mode != "raw_pan_fixed5"
            else None,
            levels=5,
        ),
        cv_best=dict(
            configurations=cv.get("winners", []), best_accuracy=cv.get("best_accuracy")
        ),
        cv_protocol=dict(
            method="leave-one-occupied-block-group-out",
            grid_origin=[0, 0],
            scaling="training-fold mean and population SD",
            svm_C=1,
            svm_gamma="scale",
            feature_sets={
                k: v["feature_names"] for k, v in cv.get("by_set", {}).items()
            },
        ),
        scene_median_equalized=options.get("scene_median_eq"),
        random_seed=options.get("random_seed", 42),
        parameters=options.get("settings", {}),
        reference_audit=options.get("reference_audit", []),
        automated_checks=a.get("check_bundle"),
        eigenvalue_summary=a["df_eigenvalues"].to_dict("records"),
    )
    if a.get("feature_names"):
        meta["combined_scaling"] = dict(
            features=a["feature_names"],
            mean=a["feature_mean"].tolist(),
            std=a["feature_std"].tolist(),
        )
    meta.update(source_metadata or {})
    (folder / "metadata_summary.json").write_text(
        json.dumps(meta, indent=2, allow_nan=False), encoding="utf8"
    )
    if a.get("check_bundle"):
        (folder / "pytest_output.txt").write_text(
            a["check_bundle"]["output"] + "\n", encoding="utf8"
        )
    if options.get("reference_audit"):
        pd.DataFrame(options["reference_audit"]).to_csv(
            folder / "tables/original_region_audit.csv", index=False
        )
    return report


def zip_folder(folder, destination):
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    path.relative_to(folder),
                    compress_type=(
                        zipfile.ZIP_STORED
                        if path.suffix == ".png"
                        else zipfile.ZIP_DEFLATED
                    ),
                )


def export_run(
    scene,
    settings,
    patches,
    analysis,
    regions,
    explicit,
    rgb_bands,
    output,
    reference_audit=None,
):
    """Atomic output replacement: readers never see a half-written run archive."""
    from modules.workflow import file_digest

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        folder = Path(tmp) / "products"
        r = scene.raster
        params = asdict(settings)
        params.update(
            regions=regions,
            manual_coordinates=explicit,
            rgb_bands=rgb_bands,
            class_seed_rule=None
            if settings.exact_boxes
            else "seed + class index + 4 × region index (indices start at zero)",
            class_order=list(dict.fromkeys(p.label for p in patches)),
            one_region_per_class=all(
                p.label in regions and not explicit.get(p.label)
                and (isinstance(regions[p.label], dict) or len(regions[p.label]) == 1)
                for p in patches
            ),
        )
        options = dict(
            settings=params,
            quantization_mode=settings.mode,
            rescale_method=settings.method,
            angles_deg=settings.angles,
            distances=[settings.distance],
            scene_median_eq=scene.median,
            conversion=scene.provenance.get("conversion"),
            random_seed=None if settings.exact_boxes else settings.seed,
            reference_audit=reference_audit or [],
        )
        preview = r.read_preview_rgb(*(rgb_bands or [0, 0, 0]))
        info = dict(
            source_sha256=file_digest(r.metadata.extra.get("original_path") or r.file_path),
            source_subset=r.metadata.extra.get("source_window"),
            original_dimensions=dict(width=r.metadata.extra.get("original_width", r.width),
                                     height=r.metadata.extra.get("original_height", r.height)),
            patch_coordinate_frame="subset-local, zero-based; add source_subset.x and .y for original pixels",

            driver=r.metadata.driver,
            native_dtypes=list(r.metadata.dtypes),
            intensity_conversion=scene.provenance.get("conversion"),
            source_nodata=(
                "NaN"
                if r.metadata.nodata is not None and np.isnan(r.metadata.nodata)
                else r.metadata.nodata
            ),
            affine=list(r.metadata.transform) if r.metadata.transform else None,
        )
        report = write_products(
            folder,
            r,
            scene.bands,
            patches,
            analysis,
            scene.stats,
            scene.hist_before,
            scene.hist_after,
            preview,
            options,
            info,
        )
        for name, filename in (
            ("pan", "panchromatic_grayscale.png"),
            ("eq", "histogram_equalized.png"),
        ):
            scene.export_png(name, folder / "images" / filename)
        # GDAL PNG sidecars are not needed; spatial parameters are in the metadata.
        for path in (folder / "images").glob("*.aux.xml"):
            path.unlink()
        bundle = Path(tmp) / "complete.zip"
        zip_folder(folder, bundle)
        bundle.replace(output)
    return report


def build_complete_zip_package(
    raster,
    selected_bands,
    pan_display_8bit,
    equalized_8bit,
    hist_before,
    hist_after,
    patches,
    df_eigenvalues,
    df_distances,
    interpretation,
    valid_mask=None,
    **options,
):
    """Compatibility adapter for existing in-memory callers."""
    from modules.preprocessing import get_image_statistics

    a = dict(
        options,
        df_eigenvalues=df_eigenvalues,
        df_distances=df_distances,
        interpretation=interpretation,
    )
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        write_products(
            folder,
            raster,
            selected_bands,
            patches,
            a,
            get_image_statistics(pan_display_8bit, valid_mask),
            hist_before,
            hist_after,
            options.get("preview_rgb"),
            options,
            options.get("extra_metadata"),
        )
        for img, name in (
            (pan_display_8bit, "panchromatic_grayscale"),
            (equalized_8bit, "histogram_equalized"),
        ):
            if img is not None:
                (folder / "images" / (name + ".png")).write_bytes(
                    array_to_png_bytes(img)
                )
        result = io.BytesIO()
        zip_folder(folder, result)
        return result.getvalue()
