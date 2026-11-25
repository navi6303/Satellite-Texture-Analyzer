"""Streamlit views for explaining the same patch measurements used by the export."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from modules.glcm_analysis import get_glcm_demonstration_data
from modules.matrix_audit import directional_table
from modules.reporting import diagnostics
from modules.validation import cv_tables
from modules.visualization import (
    EIGEN_LABELS, PATCHES_PER_CHART, plot_class_separation, plot_combined_eigenvalue_comparison,
    plot_individual_class_eigenvalues, plot_patch_map_overlay, plot_single_eigenvalue,
    plot_quantized_patch_histogram,
)


def show_chart(plot):
    fig, data = plot
    try:
        st.image(data, width="stretch")
    finally:
        plt.close(fig)
    # The explanation is embedded in the PNG so it also accompanies downloads.
    return data


def matrix_frame(matrix):
    frame = pd.DataFrame(matrix, index=range(5), columns=map(str, range(5)))
    frame.index.name = "Gray level"
    return frame


def preview_patch(image, valid):
    # Gray levels stay discrete when a large patch is reduced for the screen.
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    rgb[~valid] = [190, 200, 211]
    preview = Image.fromarray(rgb)
    preview.thumbnail((420, 240), Image.Resampling.NEAREST)
    return preview


def walkthrough(patches, settings):
    labels = list(dict.fromkeys(p.label for p in patches))
    c1, c2 = st.columns(2)
    label = c1.selectbox("Texture class", labels, key="walk_class")
    members = [p for p in patches if p.label == label]
    ids = [p.patch_id for p in members]
    pid = c2.selectbox("Texture patch", ids, key=f"walk_patch_{label}")
    p = members[ids.index(pid)]
    demo = get_glcm_demonstration_data(p, distance=settings.distance, angles_deg=settings.angles)

    left, right = st.columns(2)
    left.markdown("#### 1. Select a patch")
    left.image(preview_patch(p.image_8bit, p.valid_mask), caption="Equalized image inside the selected box")
    left.caption(f"{p.width} × {p.height} native pixels · subset origin ({p.x}, {p.y}).")
    right.markdown("#### 2. Reduce to five gray levels")
    right.image(preview_patch(np.rint(p.quantized * (255/4)).astype(np.uint8), p.valid_mask),
                caption="Five possible levels, from 0 (dark) to 4 (bright)")
    notes = {
        "global_equalized_fixed5": "All patches use the same five brightness ranges from the equalized image.",
        "per_patch_rescale_5bin": "The brightness range is stretched inside this patch before dividing it into five levels.",
        "raw_pan_fixed5": "The selected raw PAN mode uses panchromatic brightness before equalization for the five levels.",
    }
    right.caption(notes[settings.mode])
    with right:
        st.caption("Five-level histogram of valid patch pixels")
        show_chart(plot_quantized_patch_histogram(p))
    if p.sanity_warnings:
        st.warning("Patch check · " + " ".join(p.sanity_warnings))

    angle = st.selectbox("Direction to inspect (degrees)", list(settings.angles), key="walk_angle")
    info = demo["directions"][angle]
    n = info["valid_pairs"]
    left, right = st.columns(2)
    left.markdown("#### 3. Count neighboring pixels")
    left.caption(f"The gray-level co-occurrence matrix (GLCM) counts neighboring gray-level pairs inside this patch. "
                 f"Rows give the first level and columns give the neighbor’s level. At {angle}° and distance "
                 f"{settings.distance}, there are {n:,} valid pairs.")
    left.dataframe(matrix_frame(info["raw_counts"]), width="stretch")
    right.markdown("#### 4. Convert counts to frequencies")
    right.caption(f"Count each pair in both orders, then divide by the new total of {2*n:,}. "
                  "The resulting frequencies add up to 1 before rounding.")
    right.dataframe(matrix_frame(info["normalized_matrix"]).style.format("{:.4f}"), width="stretch")
    with st.expander("Inspect exact counts for this patch and direction"):
        st.caption(f"{label} · {pid} · {p.valid_pixel_count:,} valid pixels of {p.total_pixels:,} · "
                   f"{info['possible_pairs']:,} possible pairs · raw sum {n:,} · symmetric sum {info['symmetric_pair_total']:,}.")
        st.dataframe(pd.DataFrame(info["occurrences"]), hide_index=True)
        st.caption("Counts after adding the transpose")
        st.dataframe(matrix_frame(info["symmetric_counts"]))

    st.markdown("#### 5. Average the directions")
    angles = ", ".join(f"{a}°" for a in settings.angles)
    if len(settings.angles) == 1:
        st.caption(f"Only {angles} was selected, so its normalized matrix is the final matrix.")
    else:
        st.caption(f"Average each cell across {angles}. Every selected direction contributes equally.")
    st.dataframe(matrix_frame(p.glcm).style.format("{:.4f}"))
    st.markdown("#### 6. Read the five eigenvalues")
    st.caption("These five numbers come from the final matrix. Together they describe the patch and let us compare textures.")
    st.dataframe(pd.DataFrame([p.eigenvalues], columns=EIGEN_LABELS).style.format("{:.4g}"), hide_index=True)


def compare(patches, analysis):
    st.markdown("#### Compare the four class means")
    grouped = plot_combined_eigenvalue_comparison(patches)
    png = show_chart(grouped)
    st.download_button("Download class comparison", png, "plot2_combined_eigenvalue_spectra.png", "image/png")
    means = analysis["df_eigen_mean"]
    columns = {"Class Label": "Class", "N patches": "Patches", **{f"{k} mean": k for k in EIGEN_LABELS}}
    table = means[list(columns)].rename(columns=columns)
    st.dataframe(table.style.format({k: "{:.4g}" for k in EIGEN_LABELS}), hide_index=True)

    st.markdown("#### Look inside one texture class")
    c1, c2 = st.columns(2)
    label = c1.selectbox("Class to compare", list(dict.fromkeys(p.label for p in patches)), key="chart_class")
    view = c2.radio("Show", ["Individual patches", "Class mean"], horizontal=True)
    group = [p for p in patches if p.label == label]
    if view == "Class mean":
        show_chart(plot_combined_eigenvalue_comparison(group))
    else:
        pages = [f"{i+1}–{min(i+PATCHES_PER_CHART, len(group))}" for i in range(0, len(group), PATCHES_PER_CHART)]
        page = pages.index(st.selectbox("Patches to compare", pages, key=f"patch_page_{label}")) if len(pages) > 1 else 0
        png = show_chart(plot_individual_class_eigenvalues(group, page))
        st.download_button("Download displayed patch bars", png, "plot1_individual_class_eigenvalues.png", "image/png")

    st.markdown("#### Compare separation between classes")
    distance = analysis["df_class_glcm_distances"]
    show_chart(plot_class_separation(distance))
    st.caption(
        "Each class matrix is the average of its normalized patch GLCMs. "
        "The chart measures Euclidean distance directly between the five-eigenvalue class vectors. "
        "Longer bars indicate greater differences between the classes’ texture profiles."
    )
    with st.expander("Inspect one eigenvalue across classes"):
        k = EIGEN_LABELS.index(st.selectbox("Eigenvalue", EIGEN_LABELS))
        png = show_chart(plot_single_eigenvalue(patches, k))
        st.download_button("Download this eigenvalue chart", png, f"figure_{k+1}_lambda{k+1}.png", "image/png")


def checks_and_details(analysis, patches, preview, raster):
    checks = analysis["check_bundle"]
    (st.success if checks["passed"] else st.error)("Automated calculation checks · " + ("PASS" if checks["passed"] else "FAIL"))
    with st.expander("Test output"):
        st.code(checks["output"])
    warnings = [p for p in patches if p.sanity_warnings]
    if warnings:
        st.warning(f"{len(warnings)} of {len(patches)} patches triggered a brightness or gray-level warning. See patch checks below.")
    with st.expander("Patch locations, brightness and gray-level occupancy"):
        show_chart(plot_patch_map_overlay(preview, patches, raster.width, raster.height))
        st.caption("Each box marks the native image area used for one patch.")
        st.dataframe(diagnostics(patches), hide_index=True)
    with st.expander("Exact counts, coordinates and eigenvalues"):
        columns = {
            "label":"Class","patch_id":"Patch ID","x":"Subset x","y":"Subset y","original_x":"Original x",
            "original_y":"Original y","width":"Width","height":"Height","total_pixels":"Total pixels",
            "valid_pixels":"Valid pixels","angle_deg":"θ (°)","distance":"d (px)","delta_row":"Δrow",
            "delta_column":"Δcolumn","possible_pairs":"Pairs before masking","pairs":"Valid pairs",
            "raw_matrix_sum":"Raw sum","symmetric_matrix_sum":"Symmetric sum","matrix_sum":"Normalized sum",
        }
        st.dataframe(directional_table(patches)[list(columns)].rename(columns=columns), hide_index=True)
        st.dataframe(analysis["df_eigenvalues"], hide_index=True)
    with st.expander("Feature rankings and held-out classification"):
        st.caption("Each geographic block is tested separately. Accuracy is the share of held-out patches assigned the correct class.")
        cv = analysis["cv_results"]
        tables = cv_tables(cv)
        if tables["confusion"]:
            st.dataframe(tables["summary"], hide_index=True)
            model = st.selectbox("Confusion matrix", list(tables["confusion"]))
            st.caption("Rows are known classes. Columns are predicted classes.")
            st.dataframe(tables["confusion"][model])
        else:
            st.info(cv["by_set"]["combined"]["note"])
        st.caption("Larger Fisher ratios indicate greater differences between these class means relative to variation within a class.")
        st.dataframe(analysis["df_fisher"], hide_index=True)
        st.caption("Raw distances between class means")
        st.dataframe(analysis["df_distances"])
        st.caption("Distances after putting all five eigenvalues on a common scale")
        st.dataframe(analysis["df_dist_std"])
