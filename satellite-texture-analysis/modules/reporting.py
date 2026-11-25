"""Tables and narrative built from the same results used in the interface and CSVs."""

import json
import numpy as np
import pandas as pd
from modules.analysis_checks import checks_markdown
from modules.matrix_audit import matrix_audit, directional_covariance
from modules.glcm_analysis import EIG_COLUMNS
from modules.validation import cv_tables


def diagnostics(patches):
    rows = []
    for p in patches:
        row = {
            "Patch ID": p.patch_id,
            "Class": p.label,
            "x": p.x,
            "y": p.y,
            "width": p.width,
            "height": p.height,
            "Block": p.block_id,
            "Valid pixels": p.valid_pixel_count,
            "Horizontal pairs (d=1)": p.horizontal_valid_pairs,
            "Pan mean": p.pan_mean,
            "Pan std": p.pan_std,
            "Equalized mean": p.eq_mean,
            "Equalized std": p.eq_std,
            "Warnings": " ".join(p.sanity_warnings),
        }
        for i, channel in enumerate("RGB"):
            row[channel + " mean"] = p.rgb_mean[i] if p.rgb_mean is not None else None
            row[channel + " std"] = p.rgb_std[i] if p.rgb_std is not None else None
        row.update({f"L{i} fraction": value for i, value in enumerate(p.occupancy)})
        rows.append(row)
    return pd.DataFrame(rows)


def feature_summary(patches):
    rows = []
    for p in patches:
        row = {
            "Class": p.label,
            **dict(zip(EIG_COLUMNS, p.eigenvalues)),
            **p.classic_features,
        }
        if p.transition_eigenvalues is not None:
            row.update(
                {f"Tλ{i + 1}": v for i, v in enumerate(p.transition_eigenvalues)}
            )
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("Class")
    table = frame.reset_index().melt(
        id_vars="Class", var_name="Feature", value_name="Value"
    )
    return (
        table.groupby(["Feature", "Class"], sort=False)
        .Value.agg(["count", "mean", "std"])
        .reset_index()
    )


def md(frame, index=False, digits=4):
    if frame is None or frame.empty:
        return "Unavailable for this run."
    frame = frame.reset_index() if index else frame

    def fmt(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    def line(cells):
        return "| " + " | ".join(cells) + " |"

    return "\n".join(
        [
            line(map(str, frame.columns)),
            line(["---"] * len(frame.columns)),
            *(line(map(fmt, row)) for row in frame.itertuples(index=False, name=None)),
        ]
    )


def generate_markdown_report(
    raster,
    selected_bands,
    pan_stats,
    equalized_stats,
    patches,
    df_eigenvalues,
    df_distances,
    interpretation,
    **options,
):
    o = options
    cv = o.get("cv_results") or {}
    bundle = o.get("check_bundle")
    water = [p for p in patches if p.label == "Water Bodies"]
    water_flags = sum(bool(p.sanity_warnings) for p in water)
    median = o.get("scene_median_eq")
    settings = o.get("settings", {})
    d = o.get("distances") or [o.get("glcm_distance", 1)]
    angles = o.get("angles_deg") or [o.get("glcm_angle_deg", 0)]
    summary = feature_summary(patches)
    summary["Mean ± SD"] = [
        f"{m:.5f} ± {s:.5f}" if np.isfinite(s) else f"{m:.5f} (n=1)"
        for m, s in zip(summary["mean"], summary["std"])
    ]
    class_table = summary.pivot(index="Feature", columns="Class", values="Mean ± SD")
    by_class = list(dict.fromkeys(p.label for p in patches))
    tables = cv_tables(cv)
    cv_rows = tables["summary"]
    acc_table = (
        md(cv_rows.iloc[:, :3])
        if not cv_rows.empty
        else cv.get("by_set", {}).get("combined", {}).get("note", "Unavailable.")
    )
    rows = []
    for p in patches:
        occ = " / ".join(f"{100 * v:.1f}" for v in p.occupancy)
        rows.append(
            {
                "ID": p.patch_id,
                "Class": p.label,
                "x, y": f"{p.x}, {p.y}",
                "Size": f"{p.width}×{p.height}",
                "Valid pixels / horizontal pairs (d=1)": f"{p.valid_pixel_count:,} / {p.horizontal_valid_pairs:,}",
                "L0–L4 (%)": occ,
                "Equalized mean ± SD": f"{p.eq_mean:.1f} ± {p.eq_std:.1f}",
                "RGB mean ± SD": ", ".join(
                    f"{m:.1f} ± {s:.1f}" for m, s in zip(p.rgb_mean, p.rgb_std)
                )
                if p.rgb_mean is not None
                else "Unavailable",
                "Sanity": "Review" if p.sanity_warnings else "Pass",
            }
        )
    warnings = "\n".join(
        f"- **{p.patch_id}** · {' '.join(p.sanity_warnings)}"
        for p in patches
        if p.sanity_warnings
    )
    audit = o.get("reference_audit") or []
    audit_section = ""
    if audit:
        ref_rows = [
            {
                "Region": r["label"],
                "Eq. mean": r["equalized_mean"],
                "Scene median": r["scene_median"],
                "L0 (%)": 100 * r["occupancy"][0],
                "L3 + L4 (%)": 100 * sum(r["occupancy"][3:]),
                "λ1": r["lambda1"],
                "Largest diagonal": r["largest_diagonal"],
                "Sanity": "Review" if r["warnings"] else "Pass",
            }
            for r in audit
        ]
        audit_section = (
            "### Original full-crop audit\n\nThe supplied region rectangles were independently measured "
            "using global equalization, fixed bins, d=1 and 0°. Only their subpatches enter CV.\n\n"
            + md(pd.DataFrame(ref_rows))
        )
    signed = sum(p.eigenvalues[-1] < -1e-12 for p in patches)
    sign_note = (
        f"λ5 was negative in {signed} of {len(patches)} patches. Figure 5 retains the signed values."
        if signed
        else "All λ5 observations were non-negative. Figure 5 starts at zero."
    )
    model_note = ""
    if cv.get("winners"):
        model_note = " Highest-scoring configurations share the same evaluation folds."
    single_region = settings.get("one_region_per_class", False)
    limit = (
        "All samples of each class came from one labeled region. The held-out blocks measure variation within "
        "those four regions. Accuracy at new water bodies, vegetation sites, urban areas or soils is unknown."
        if single_region
        else "The score applies to the submitted labels and geographic blocks. Neighboring blocks may remain correlated."
    )
    replicated = min(sum(p.label == label for p in patches) for label in by_class) >= 2
    if not replicated:
        limit = "At least one class has a single box. Between-box SD for that class, four-class Fisher ranking and supervised validation accuracy cannot be estimated."
    matrix_table = matrix_audit(patches)
    pca = directional_covariance(patches)
    audit_view = matrix_table[
        [
            "Patch ID",
            "Class",
            "P shape",
            "P sum",
            "Largest diagonal",
            "Pixel occupancy at peak",
            "Trace",
            "λ1",
            "Symmetry error",
        ]
    ]
    mapping = json.dumps(
        {p.patch_id: p.label for p in patches}, ensure_ascii=False, indent=2
    )
    baseline = pd.DataFrame(
        [p.extra["baseline"] for p in patches if "baseline" in p.extra]
    )
    fisher_text = (
        md(o.get("df_fisher"), digits=5)
        if replicated
        else "No four-class Fisher ranking is reported. Replicate boxes are needed to estimate within-class variance."
    )
    validation_text = (
        f"Each occupied group was held out once. The SVM used an RBF kernel, C=1 and gamma=scale."
        if cv.get("winners")
        else "No held-out classification accuracy is available for this layout."
    )
    source_text = (
        "Each observation uses the entire confirmed bounding box, in native source pixels, with invalid pixels masked."
        if settings.get("exact_boxes")
        else "The coordinates below identify every analyzed window."
    )
    detail = "\n".join("- " + d for d in interpretation.get("details", []))
    metrics_details = []
    for name, models in cv.get("by_set", {}).items():
        for clf in ("nearest_centroid", "svm_rbf"):
            score = models.get(clf)
            if score:
                classes = score["class_names"]
                matrix = pd.DataFrame(
                    score["confusion_matrix"],
                    index=pd.Index(classes, name="Actual / predicted"),
                    columns=classes,
                )
                pr = pd.DataFrame(
                    {
                        "Class": classes,
                        "Precision": [score["precision"][c] for c in classes],
                        "Recall": [score["recall"][c] for c in classes],
                    }
                )
                metrics_details.extend(
                    [f"#### {name} · {clf}", md(matrix, index=True), md(pr)]
                )
    mode = o.get("quantization_mode", "global_equalized_fixed5")
    method = o.get("rescale_method", "percentile")
    quantization_text = {
        "global_equalized_fixed5": "Every box uses the same five intervals on the globally equalized 8-bit PAN: 0–51, 52–102, 103–153, 154–204 and 205–255. No additional patch stretch was applied.",
        "per_patch_rescale_5bin": "Each equalized box is stretched separately using "
        + (
            "its 2nd and 98th percentiles"
            if method == "percentile"
            else "its minimum and maximum"
        )
        + " before five-bin quantization.",
        "raw_pan_fixed5": "Every box uses the unenhanced native mean for uint8 input, or the scene-wide 8-bit stretch for other source types.",
    }[mode]
    box = raster.metadata.extra.get("source_window")
    subset_note = ""
    if box:
        subset_note = (f"The original image is {raster.metadata.extra['original_width']} × "
                       f"{raster.metadata.extra['original_height']} pixels. Its selected subset begins at "
                       f"column {box['x']}, row {box['y']} and measures {box['w']} × {box['h']} pixels. "
                       "All bands were cropped at native resolution before averaging. The histogram and "
                       "patch coordinates below refer to this subset.")
    # A short reading path first; all original technical tables remain in the report.
    mean_table = df_eigenvalues.groupby("Class Label", sort=False)[EIG_COLUMNS].mean()
    mean_table.columns = [f"λ{i}" for i in range(1, 6)]
    mean_table.insert(0, "Patches", df_eigenvalues.groupby("Class Label", sort=False).size())
    mean_table = mean_table.rename_axis("Class").reset_index()
    distance_figure = ("![Distance between each pair of classes](plots/class_separation.png)\n\n"
                       "The chart combines the normalized patch GLCMs with equal weight within each class, "
                       "then extracts five eigenvalues per class. Each bar gives the Euclidean distance "
                       "directly between two class eigenvalue vectors."
                       if o.get("df_class_glcm_distances") is not None else "")
    conversion = {
        "weighted_gray": "The app computes grayscale as 0.114 × B + 0.587 × G + 0.299 × R.",
        "single_band": "The input already has one intensity band, whose valid values are copied.",
    }.get(o.get("conversion"), "The native panchromatic image is the arithmetic mean of the selected bands at each jointly valid pixel.")
    return f"""# Satellite Texture Analyzer · Academic Research Report

## 1. Executive summary

{interpretation.get("summary", "")}
{water_flags} of {len(water)} Water patches triggered a sanity warning. {limit}

## How to present the calculation

1. Select one texture patch inside the equalized image subset.
2. Reduce its brightness values to five levels, numbered 0 to 4, using the selected mode described below.
3. Count pairs of neighboring levels for each selected direction.
4. Add the reverse pairs and divide by the new total. Each direction now has a table of relative frequencies.
5. Average the tables cell by cell, giving each direction equal weight.
6. Read the five eigenvalues from the averaged matrix. Compare those five numbers across patches and classes.

### Compare the measured patches

{md(mean_table, digits=4)}

![Mean eigenvalues for each texture class](plots/combined_feature_spectrum.png)

Each bar is the mean over all measured patches of that class. The whiskers show one sample standard deviation.

![Individual patches grouped by eigenvalue](plots/individual_class_eigenvalues.png)

Each group compares up to three patches of one class. Additional patch pages are included when needed.

{distance_figure}

<details>
<summary>Dataset, exact counts, feature tables and validation</summary>

## 2. Dataset

**{raster.metadata.filename}** · {raster.width:,} × {raster.height:,} pixels · {raster.count} source bands · {raster.metadata.crs} · pixel size {raster.metadata.pixel_size_str}. The analysis used bands {", ".join(str(i + 1) for i in selected_bands)} (one-based). {pan_stats.get("valid_pixels", 0):,} source pixels were valid.

{subset_note}

## 3. Panchromatic-like Image Synthesis, Histogram Equalization and Five-Level Gray Level Reduction

{conversion} The resulting intensity mean is {pan_stats.get("mean", float("nan")):.3f} DN and its standard deviation is {pan_stats.get("std", float("nan")):.3f} DN. The valid scene minimum and maximum set the 8-bit display stretch. A global CDF over that image produces the equalized values. The equalized scene median is {median if median is not None else "unavailable"}.

Quantization mode **{mode}**. {quantization_text} Five bins use floor(5 × intensity / 256). Masked pixels are excluded.

## 4. Patch locations and sanity checks

{source_text} Coordinates use a zero-based top-left origin. Each window spans columns x through x+w−1 and rows y through y+h−1.

{md(pd.DataFrame(rows))}

{warnings or "All sampled patches passed both sanity rules."}

![Patch locations](plots/patch_map_overlay.png)

![Patch thumbnails](plots/patch_thumbnails.png)

{audit_section}

## 5. GLCM Computation Parameters

Distance {d} px, angles {angles} degrees. Each direction uses its own valid-pair denominator after symmetrization. The analyzed GLCM is the equally weighted average of these probability matrices. Four directions reduce orientation sensitivity on the pixel lattice. Horizontal audit counts always use d=1, including when the analysis uses other offsets.

### Pipeline A matrix and label audit

Pipeline A diagonalizes the symmetric normalized joint probability matrix P. The signed eigenvalues are ordered by decreasing absolute magnitude. Their signed sum equals the trace. The offset-by-offset symmetric integer counts and denominators are in `tables/directional_glcms.csv`.

```json
{mapping}
```

{md(audit_view, digits=6)}

![Leading normalized GLCM eigenvalue for the measured boxes](plots/figure_1_lambda1.png)

### Same-window comparison with the original settings

These values use the identical box pixels with global fixed five-bin quantization and a horizontal one-pixel offset.

{md(baseline, digits=6)}

## 6. Numerical Results: GLCM Eigenvalues and classic features

Per-class means appear below with sample SD when multiple boxes exist. ASM equals the sum of squared probabilities. Energy is its square root. Entropy is in bits, with zero-probability terms omitted. Correlation is assigned 1 for a constant window (zero marginal variance). Trace and diagonal share are identical and contribute one column to a model. The transition spectrum uses a row-normalized GLCM with zero rows for absent levels.

{md(class_table[by_class], index=True, digits=5)}

Each individual λ figure shows the class means. Error bars show one between-box sample SD only for replicated classes. {sign_note} Per-patch spectra and matrices accompany the report.

### Pipeline B · eigenvalues of directional GLCM feature covariance

For each box, each selected offset contributes one row of five features: contrast, entropy in bits, homogeneity, correlation, and ASM. The table gives the eigenvalues of their 5 × 5 sample covariance matrix in native feature units (denominator m−1). At least two offsets are required. With four offsets the covariance has rank at most three. PC1 measures feature variation across offsets inside the box. These diagnostics are separate from the five Pipeline A figures and the classifiers.

{md(pca, digits=7)}

![Signed eigenvalues of the normalized GLCM for the measured boxes](plots/combined_feature_spectrum.png)

## 7. Pairwise Euclidean Distances

Both tables compare the same five-eigenvalue class centroids. The second uses pooled patch means and population SD for standardization. Additional CSVs give the corresponding distances for the combined feature set.

### Raw

{md(df_distances, index=True)}

### Standardized

{md(o.get("df_dist_std"), index=True)}

## 8. Scientific Interpretation & Separability Discussion

### 8.1 Observed separability

{detail}

### 8.2 Occupancy and the spectrum

The handbook’s typical class order is an empirical expectation. The definition of P imposes no Water–Vegetation ordering. For example, two matrices with diagonal probabilities (1,0,0,0,0) and (0.2,0.2,0.2,0.2,0.2) both have trace 1, yet their λ1 values are 1 and 0.2. An equalized GLCM whose mass concentrates on the diagonal largely reflects gray-level occupancy. In the diagonal limit its eigenvalues equal the sorted level probabilities. The observed λ1-to-diagonal gap above quantifies the approximation for this run. A Water label cannot be verified from the eigenvalue ordering. Transition eigenvalues describe conditional neighbor changes and can still depend on the quantization.

### 8.3 Which features distinguish these samples?

{fisher_text}

Fisher ratios divide the variance of the class means (weighted by class size) by the pooled within-class variance, both normalized by the total patch count. Constant features score zero. A machine-precision denominator floor keeps zero-scatter ratios finite. Ranks describe these samples.

### 8.4 Methodological Limitations

{limit} Sanity flags prompt inspection of the original imagery. They do not establish the cause of a bright Water crop.

## 9. Spatially blocked validation

Grid width {settings.get("block_size", "unspecified")} pixels, anchored at the processed image origin. {len(set(p.block_id for p in patches))} occupied block groups. Cells touched by the same patch stay together. {validation_text} Feasible folds use training-only scaling.{model_note}

{acc_table}

{chr(10).join([chr(10) + item + chr(10) for item in metrics_details])}

## 10. Automated checks

{checks_markdown(bundle) if bundle else "No run check was supplied."}

## 11. Reproduction

See `metadata_summary.json` for the source identity, coordinates, parameter values and package versions. The CSVs include per-box features, counts and verified coordinates. Held-out predictions and confusion matrices are included for eligible replicated runs. Quantized PNG display values 0, 63, 126, 189 and 252 correspond to levels 0 through 4.

</details>
"""
