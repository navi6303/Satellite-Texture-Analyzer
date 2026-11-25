"""
Satellite Texture Analyzer - Professional Remote Sensing Application.
Postgraduate academic tool for analyzing land-cover textures using Gray-Level
Co-occurrence Matrix (GLCM) eigenvalues.

Workflow:
  1. Upload Real Satellite Image
  2. Inspect original bands, confirm a native-resolution subset and crop all bands
  3. Generate grayscale or average the raster bands
  4. Histogram Equalization
  5. Collect texture patches for four classes
  6. Five-Level Quantization
  7. GLCM Generation
  8. Eigenvalue Calculation
  9. Texture Separability Analysis
  10. Download Results
"""

from __future__ import annotations

import tempfile
import json
import os
import sys
from pathlib import Path
from dataclasses import asdict, replace

# Ensure rasterio uses its own bundled PROJ database on Windows if another application
# (such as PostgreSQL / PostGIS) set an incompatible system PROJ_LIB
for _proj_env in ("PROJ_LIB", "PROJ_DATA"):
    _val = os.environ.get(_proj_env)
    if _val and ("postgre" in _val.lower() or not Path(_val).exists()):
        os.environ.pop(_proj_env, None)

# Ensure modules directory is importable
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import cv2
import pandas as pd
import streamlit as st

from modules.export_utils import export_run
from modules.image_loader import detect_satellite_band_mapping, load_raster
from modules.patch_sampling import parse_coordinate_list, boxes_overlap
from modules.workflow import Settings, extract_samples, analyze_samples, LABELS
from modules.scene import prepare_scene, equalize_scene, grayscale_weights, inspect_patch
from modules.subset import crop_original, checked_box
from modules.workflow import sampled_confirmed_boxes
from modules.snip_tool import snip_canvas
from modules.texture_extraction import (
    QUANT_GLOBAL_EQ_FIXED5,
    QUANT_PER_PATCH_RESCALE,
    QUANT_RAW_PAN_FIXED5,
)
from modules.presentation import walkthrough, compare, checks_and_details
from modules.preprocessing_view import subset_matrix, conversion_matrices, equalization_matrices

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Satellite Texture Analyzer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
state = st.session_state
for key, initial in dict(
    raster=None,
    subset=None,
    subset_rev=0,
    loaded_file_sig=None,
    selected_bands=[],
    scene=None,
    user_classes=list(LABELS),
    confirmed_boxes={c: [] for c in LABELS},
    regions={},
    drawn={},
    texture_patches=[],
    analysis=None,
    export=None,
    selection_rev=0,
    input_signature=None,
    canvas_view_layer="Enhanced Panchromatic Image",
).items():
    state.setdefault(key, initial)
if "workspace" not in state:
    state.workspace = tempfile.TemporaryDirectory(prefix="satellite-texture-")
work = Path(state.workspace.name)


def clear_results():
    state.texture_patches = []
    state.analysis = None
    state.export = None


def reset_downstream_stages():
    state.scene = None
    state.subset = None
    state.subset_rev += 1
    state.confirmed_boxes = {c: [] for c in state.user_classes}
    state.regions, state.drawn = {}, {}
    state.selection_rev += 1
    state.input_signature = None
    clear_results()


with st.sidebar:
    st.header("Analysis settings")
    with st.expander("Directions and distance"):
        distance = st.slider("Inter-pixel Distance (d)", 1, 3, 1)
        angles = st.multiselect(
            "Displacement Angles (θ)", [0, 45, 90, 135], default=[0, 45, 90, 135]
        )
        st.checkbox("Symmetric GLCM", value=True, disabled=True)
        st.checkbox("Normalized Matrix", value=True, disabled=True)
        transition = st.checkbox(
            "Compare row-normalized transition eigenvalues", value=True
        )
    with st.expander("Sampling and gray levels"):
        sampling = st.radio(
            "Pixels to analyze",
            ["Complete user bounding boxes", "One crop per class", "Sample subpatches within regions"],
        )
        exact = sampling != "Sample subpatches within regions"
        one_crop = sampling == "One crop per class"
        n = st.number_input("Patches per class", 3, 100, 5, disabled=exact)
        size = st.number_input("Random crop side (pixels)", 8, 2048, 64, disabled=exact)
        block = st.number_input("Spatial block side (pixels)", 8, 65536, 256)
        seed = st.number_input("Sampling seed", 0, 2**31 - 1, 42, disabled=exact)
        modes = {
            QUANT_PER_PATCH_RESCALE: "Per-patch rescale → 5 bins",
            QUANT_GLOBAL_EQ_FIXED5: "Global equalization → fixed 5 bins",
            QUANT_RAW_PAN_FIXED5: "Raw panchromatic → fixed 5 bins",
        }
        mode = st.selectbox(
            "Quantization",
            list(modes),
            index=list(modes).index(Settings().mode),
            format_func=modes.get,
        )
        method = st.selectbox(
            "Local rescale method",
            ["percentile", "minmax"],
            format_func=lambda x: (
                "2nd–98th percentile" if x == "percentile" else "Min–max"
            ),
            disabled=mode != QUANT_PER_PATCH_RESCALE,
        )
    settings = Settings(
        n, size, block, seed, mode, method, angles, distance, transition, exact
    )
    with st.expander("Satellite imagery sources"):
        st.markdown("[Copernicus Data Space Browser](https://dataspace.copernicus.eu/browser/)")
        st.markdown("[USGS EarthExplorer](https://earthexplorer.usgs.gov/)")

st.title("Satellite Texture Analyzer")
st.caption("Compare patterns in four land-cover classes using small boxes drawn on a satellite image.")
had_results = state.analysis is not None
setup = st.expander("Edit the image, subset or patch boxes") if had_results else st.container()
with setup:
    st.markdown("### 1. Upload Real Satellite Image")
    upload_limit = st.get_option("server.maxUploadSize")
    st.caption(
        "Please upload a real remotely sensed satellite image containing diverse land-use and land-cover types. "
        f"Supports multispectral GeoTIFF, ENVI header with its binary file, Radiance HDR, RGB imagery, and single-band rasters up to **{upload_limit} MB** per file."
    )

    uploaded_file = st.file_uploader(
        "Select Satellite Image File",
        type=["tif", "tiff", "png", "jpg", "jpeg", "bmp", "hdr"],
        help=f"Upload your Sentinel-2, Landsat, or geospatial raster. The per-file upload limit is {upload_limit} MB.",
    )

    # ENVI stores the band layout in the .hdr and pixel values in a companion file.
    envi = False
    if uploaded_file is not None and uploaded_file.name.lower().endswith(".hdr"):
        uploaded_file.seek(0)
        envi = uploaded_file.read(4).upper() == b"ENVI"
        uploaded_file.seek(0)
    payload = st.file_uploader("ENVI binary data file for this header", key="envi_payload") if envi else None
    file_sig = ((uploaded_file.file_id, uploaded_file.name, uploaded_file.size,
                 payload.file_id if payload else None) if uploaded_file is not None else None)
    if state.loaded_file_sig != file_sig:
        reset_downstream_stages()
        state.raster = None
        state.loaded_file_sig = file_sig
        if uploaded_file is not None and (not envi or payload is not None):
            try:
                from modules.image_loader import load_envi_upload
                folder = work / "uploads" / str(state.subset_rev)
                state.raster = (load_envi_upload(uploaded_file, payload, folder) if envi
                                else load_raster(uploaded_file, uploaded_file.name, cache_dir=folder))
            except Exception as exc:
                st.error(f"Error loading satellite image: {exc}")
    if state.raster is None:
        st.info("Upload the matching ENVI binary data file." if envi else "Upload the original satellite image to begin.")
        st.stop()

    # =============================================================================
    # STAGE 2: IMAGE METADATA & BAND INSPECTION
    # =============================================================================
    if st.session_state.raster is not None:
        raster = st.session_state.raster
        meta = raster.metadata

        st.markdown("---")
        st.markdown("### 2. Image Metadata & Band Inspection")
        st.caption(
            "Inspect verified geospatial raster parameters obtained directly from the dataset headers."
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Original Dimensions", meta.dimensions_str)
        m2.metric("Spectral Band Count", f"{meta.count} band(s)")
        m3.metric("Native Data Type", str(meta.dtypes[0]) if meta.dtypes else "unknown")
        m4.metric("File Size", f"{meta.file_size_mb} MB")

        with st.expander("🔍 Complete Geospatial Metadata Details", expanded=False):
            col_m_left, col_m_right = st.columns(2)
            with col_m_left:
                st.markdown(f"**CRS:** `{meta.crs}`")
                st.markdown(f"**Spatial Bounds:** `{meta.bounds_str}`")
                st.markdown(f"**Pixel Size:** `{meta.pixel_size_str}`")
            with col_m_right:
                st.markdown(
                    f"**Nodata:** `{meta.nodata if meta.nodata is not None else 'None'}`"
                )
                st.markdown(f"**Driver:** `{meta.driver}`")
                st.markdown(f"**Transform:** `{meta.transform}`")

        col_bands, col_composite = st.columns([1, 1.2])

        with col_bands:
            st.markdown("#### Original spectral bands")
            st.caption("Every band is retained in the selected subset.")
            st.dataframe(pd.DataFrame({"Band": list(range(1, raster.count + 1)),
                                       "Description": raster.band_names, "Data type": meta.dtypes}),
                         hide_index=True)
        with col_composite:
            st.markdown("#### Colour Composite Visualization")
            st.caption(
                "Configure display channels for visual interpretation. "
                "These channel choices affect only the preview. Conversion uses the original channel values."
            )
            band_suggestions = detect_satellite_band_mapping(
                raster.count, meta.band_descriptions
            )

            if raster.count >= 3:
                c_r, c_g, c_b = st.columns(3)
                with c_r:
                    r_channel = st.selectbox(
                        "Red Channel",
                        options=list(range(raster.count)),
                        index=min(band_suggestions["r_idx"], raster.count - 1),
                        format_func=lambda x: f"B{x + 1}: {raster.band_names[x][:18]}",
                        key="rgb_r",
                    )
                with c_g:
                    g_channel = st.selectbox(
                        "Green Channel",
                        options=list(range(raster.count)),
                        index=min(band_suggestions["g_idx"], raster.count - 1),
                        format_func=lambda x: f"B{x + 1}: {raster.band_names[x][:18]}",
                        key="rgb_g",
                    )
                with c_b:
                    b_channel = st.selectbox(
                        "Blue Channel",
                        options=list(range(raster.count)),
                        index=min(band_suggestions["b_idx"], raster.count - 1),
                        format_func=lambda x: f"B{x + 1}: {raster.band_names[x][:18]}",
                        key="rgb_b",
                    )

                if (
                    raster.count >= 4
                    and r_channel == 3
                    and g_channel == 2
                    and b_channel == 1
                ):
                    comp_tag = "True Colour Composite (TCC) — B4-Red, B3-Green, B2-Blue"
                elif (
                    r_channel == 0
                    and g_channel == 1
                    and b_channel == 2
                    and raster.count == 3
                ):
                    comp_tag = "True Colour Composite (TCC) — Channels 1, 2, 3"
                elif raster.count >= 4 and g_channel == 3 and b_channel == 2:
                    comp_tag = "False Colour Composite (FCC) — Vegetation in Red"
                else:
                    comp_tag = f"Custom Composite (R: B{r_channel + 1}, G: B{g_channel + 1}, B: B{b_channel + 1})"

                st.info(f"🎨 **Visualization Mode:** {comp_tag}")
                rgb_preview = raster.read_preview_rgb(
                    r_channel, g_channel, b_channel, max_dim=1024
                )
                st.image(
                    rgb_preview,
                    caption=(
                        f"Composite Preview (Display: {rgb_preview.shape[1]}×{rgb_preview.shape[0]} px | "
                        f"Native: {raster.width}×{raster.height} px)"
                    ),
                    width="stretch",
                )
                # Store FCC bands in session state for snipping-tool background
                st.session_state["fcc_r"] = r_channel
                st.session_state["fcc_g"] = g_channel
                st.session_state["fcc_b"] = b_channel
            else:
                st.info(f"{raster.count}-band raster. Band 1 is used for the grayscale preview.")
                single_preview = raster.read_preview_rgb(0, 0, 0, max_dim=1024)
                st.image(
                    single_preview,
                    caption=(
                        f"Grayscale Preview (Display: {single_preview.shape[1]}×{single_preview.shape[0]} px | "
                        f"Native: {raster.width}×{raster.height} px)"
                    ),
                    width="stretch",
                )
                st.session_state["fcc_r"] = 0
                st.session_state["fcc_g"] = 0
                st.session_state["fcc_b"] = 0


    # =============================================================================
    # Select on the original RGB preview. The canvas returns native pixel coordinates.
    original = state.raster
    original_preview = rgb_preview if original.count >= 3 else single_preview
    display_bands = [state.fcc_r, state.fcc_g, state.fcc_b]
    st.divider()
    st.markdown("### 2b. Select a subset of the original image")
    if state.subset is None:
        st.caption("Draw a rectangle, move it or drag its handles, then click Confirm Box. Coordinates refer to the original image.")
        candidate = snip_canvas(original_preview, original.width, original.height, "Image subset", {},
                                file_sig=str(file_sig), key=f"subset_{state.subset_rev}")
        if st.button("Use the whole image"):
            candidate = {"x": 0, "y": 0, "w": original.width, "h": original.height}
        with st.expander("Enter subset coordinates instead"):
            subset_text = st.text_input("Original x, y, width, height", key=f"subset_text_{state.subset_rev}")
            typed = st.button("Use these subset coordinates")
        if candidate or typed:
            try:
                rows = parse_coordinate_list(subset_text) if typed else [candidate]
                if len(rows) != 1:
                    raise ValueError("Supply exactly one subset rectangle.")
                state.subset = crop_original(original, rows[0], work / "subset.tif")
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))
        st.stop()

    raster = state.subset
    roi = raster.metadata.extra["source_window"]
    subset_preview = raster.read_preview_rgb(*display_bands)
    left, right = st.columns(2)
    overlay = original_preview.copy()
    sy, sx = overlay.shape[0] / original.height, overlay.shape[1] / original.width
    cv2.rectangle(overlay, (int(roi["x"]*sx), int(roi["y"]*sy)),
                  (round((roi["x"]+roi["w"])*sx)-1, round((roi["y"]+roi["h"])*sy)-1), (255,180,0), 2)
    left.image(overlay, caption="Confirmed subset on the original image", width="stretch")
    right.image(subset_preview, caption="Extracted original-band subset (display composite)", width="stretch")
    st.caption(f"Original pixel origin ({roi['x']}, {roi['y']}) · subset height × width × bands: "
               f"{raster.height} × {raster.width} × {raster.count}.")
    if st.button("Select a different subset"):
        reset_downstream_stages()
        st.rerun()
    window = subset_matrix(raster, f"subset_matrix_{state.subset_rev}")
    bands = list(range(raster.count))
    grayscale = raster.count == 1 or grayscale_weights(raster, bands) is not None
    product = "Grayscale" if grayscale else "Panchromatic"
    st.markdown(f"### 3. Generate {product.lower()} subset")
    if state.scene is None and st.button(f"Confirm Subset and Generate {product} Image", type="primary"):
        try:
            with st.spinner(f"Generating the {product.lower()} image for the full subset…"):
                state.scene = prepare_scene(raster, bands, work / "scene")
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    scene = state.scene
    if scene is not None:
        st.image(scene.preview("pan"), caption=f"{product} subset (8-bit preview)", width="stretch")
        stats = scene.stats
        st.caption(
            f"Valid pixels {stats['valid_pixels']:,} / {stats['total_pixels']:,} · "
            f"Native mean {stats['mean']:.3f} ± {stats['std']:.3f} DN · Range {stats['min']:g}–{stats['max']:g}"
        )
        st.divider()
        conversion_matrices(scene, window)
        st.markdown("### 4. Contrast Enhancement (Histogram Equalization)")
        if st.button("✨ Enhance Contrast", type="primary"):
            with st.spinner("Applying the global equalization lookup table…"):
                clear_results()
                equalize_scene(scene)
        if scene.hist_after is not None:
            c1, c2 = st.columns(2)
            c1.image(scene.preview("pan"), caption="Before equalization", width="stretch")
            c2.image(scene.preview("eq"), caption="After equalization", width="stretch")
            equalization_matrices(scene, window)

    if scene is not None and scene.hist_after is not None:
        rgb_bands = (
            [band_suggestions[k] for k in ("r_idx", "g_idx", "b_idx")]
            if raster.count >= 3
            else None
        )
        preview = subset_preview
        signature = json.dumps([asdict(settings), rgb_bands, sampling], sort_keys=True)
        if signature != state.input_signature:
            clear_results()
            state.input_signature = signature
            if had_results:
                st.rerun()


        # =========================================================================
        # STAGE 4: SELECT FOUR DISTINCT TEXTURE CLASSES
        # =========================================================================
        st.divider()
        st.markdown("### 5. Select Four Distinct Texture Classes")
        st.caption(
            "Define four distinct land-use and land-cover texture classes present in your satellite image. "
            "You can customize the class names below according to your scene."
        )

        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            new_c1 = st.text_input("Texture Class 1", value=state.user_classes[0] if len(state.user_classes) > 0 else "Water Bodies", key="cls_0")
        with col_c2:
            new_c2 = st.text_input("Texture Class 2", value=state.user_classes[1] if len(state.user_classes) > 1 else "Vegetation", key="cls_1")
        with col_c3:
            new_c3 = st.text_input("Texture Class 3", value=state.user_classes[2] if len(state.user_classes) > 2 else "Urban / Built-up Area", key="cls_2")
        with col_c4:
            new_c4 = st.text_input("Texture Class 4", value=state.user_classes[3] if len(state.user_classes) > 3 else "Bare Soil", key="cls_3")

        proposed_classes = [new_c1.strip(), new_c2.strip(), new_c3.strip(), new_c4.strip()]
        if any(not c for c in proposed_classes):
            st.error("⚠️ All four texture class names must be non-empty.")
        elif len(set(proposed_classes)) < 4:
            st.error("⚠️ Texture class names must be unique. Exactly four distinct classes are required.")
        elif proposed_classes != state.user_classes:
            old_classes = list(state.user_classes)
            new_boxes = {}
            new_drawn = {}
            for old_c, new_c in zip(old_classes, proposed_classes):
                boxes = state.confirmed_boxes.get(old_c, [])
                for b in boxes:
                    b["class_name"] = new_c
                    if b.get("patch") is not None:
                        b["patch"].label = new_c
                new_boxes[new_c] = boxes
                new_drawn[new_c] = state.drawn.get(old_c, [])
            state.user_classes = proposed_classes
            state.confirmed_boxes = new_boxes
            state.drawn = new_drawn
            clear_results()
            st.rerun()

        cat_palette = ["#00b4d8", "#2dc653", "#e63946", "#fca311", "#a855f7", "#ec4899", "#14b8a6", "#f97316"]
        category_colors = {c: cat_palette[i % len(cat_palette)] for i, c in enumerate(state.user_classes)}

        # =========================================================================
        # STAGE 5: INTERACTIVE BOUNDING-BOX SELECTION & PATCH COLLECTION
        # =========================================================================
        st.divider()
        st.markdown("### 6. Interactive Bounding-Box Selection & Patch Collection")
        st.caption(
            "Draw rectangular bounding boxes directly over areas of interest. "
            "Use **Zoom & Pan** to inspect different regions. Drag handles to **resize**, or drag inside the box to **reposition**. "
            + ("Collect **one box per class** for a four-crop comparison." if one_crop else
               "Collect **at least 3 separate bounding boxes per class**.")
        )

        col_view, col_cat = st.columns([1.2, 1.8])
        with col_view:
            view_layer = st.radio(
                "Canvas Display Layer:",
                ["Enhanced Panchromatic Image", "Original-band Subset (Composite)"],
                horizontal=True,
                help="Toggle between contrast-enhanced panchromatic raster and RGB composite."
            )
        with col_cat:
            active_cat = st.selectbox(
                "Assign Bounding Box to Texture Class:",
                state.user_classes,
                index=0,
                help="Choose which texture class to assign to the next confirmed bounding box."
            )

        # Determine background image for canvas
        if view_layer == "Enhanced Panchromatic Image":
            canvas_display_img = scene.preview("eq")
            if canvas_display_img.ndim == 2:
                canvas_display_img = cv2.cvtColor(canvas_display_img, cv2.COLOR_GRAY2RGB)
        else:
            canvas_display_img = subset_preview
            if canvas_display_img.ndim == 2:
                canvas_display_img = cv2.cvtColor(canvas_display_img, cv2.COLOR_GRAY2RGB)

        # Gather visible selections to display as colored bounding boxes in canvas
        visible_boxes = {}
        for c in state.user_classes:
            boxes_for_c = state.confirmed_boxes.get(c, [])
            visible_boxes[c] = [b["box"] for b in boxes_for_c]

        canvas_result = snip_canvas(
            canvas_display_img,
            raster.width,
            raster.height,
            active_cat,
            visible_boxes,
            file_sig=f"{state.loaded_file_sig}_{view_layer}",
            category_colors=category_colors,
            key=f"snip_{state.selection_rev}_{active_cat}",
        )

        def add_confirmed(label, candidates):
            # Commit the batch only after every rectangle has passed the same checks.
            if not candidates:
                raise ValueError("Enter at least one rectangle.")
            if one_crop and len(state.confirmed_boxes[label]) + len(candidates) > 1:
                raise ValueError(f"{label}: keep exactly one box in this mode. Delete the existing box to replace it.")
            occupied = [row["box"] for records in state.confirmed_boxes.values() for row in records]
            rows = []
            for candidate in candidates:
                box = checked_box(candidate, raster.width, raster.height)
                if any(boxes_overlap(box, other) for other in occupied):
                    raise ValueError("This rectangle overlaps an existing or newly entered patch.")
                thumb, _ = inspect_patch(scene, box, distance, angles)
                occupied.append(box)
                rows.append(dict(class_name=label, box=box, thumbnail=thumb, **box))
            state.confirmed_boxes[label].extend(rows)
            state.drawn.setdefault(label, []).extend(row["box"] for row in rows)
            state.selection_rev += 1
            clear_results()

        if canvas_result:
            try:
                add_confirmed(active_cat, [canvas_result])
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))

        # ── Review & Manage Confirmed Bounding Boxes
        target_count = 1 if one_crop else 3
        with st.expander("📋 Review & Manage Confirmed Bounding Boxes", expanded=True):
            rev_tabs = st.tabs(state.user_classes)
            for i, c in enumerate(state.user_classes):
                with rev_tabs[i]:
                    boxes = state.confirmed_boxes.get(c, [])
                    st.markdown(f"**{c}** — `{len(boxes)} / {target_count} collected`")
                    if not boxes:
                        st.info(f"No bounding boxes collected for '{c}' yet. Draw a rectangle on the image above and click ✅ Confirm Box.")
                    else:
                        cols_p = st.columns(min(len(boxes), 4))
                        for b_idx, rec in enumerate(boxes):
                            col = cols_p[b_idx % len(cols_p)]
                            col.image(rec["thumbnail"], caption=f"{c} · box {b_idx+1} ({rec['w']}×{rec['h']} px)", width=120)
                            col.caption(f"Subset ({rec['x']}, {rec['y']}) · original ({roi['x']+rec['x']}, {roi['y']+rec['y']})")
                            if col.button("🗑 Delete", key=f"del_box_{c}_{b_idx}"):
                                state.confirmed_boxes[c].pop(b_idx)
                                if c in state.drawn and b_idx < len(state.drawn[c]):
                                    state.drawn[c].pop(b_idx)
                                state.selection_rev += 1
                                clear_results()
                                st.rerun()

        # Optional Native Coordinates Manual Input
        with st.expander("⌨️ Optional: Enter Native Coordinates Directly", expanded=False):
            st.caption("Enter one rectangle per line as x, y, width, height (or x, y, side for a square). Subset-local native pixel coordinates, starting at (0, 0).")
            coord_tabs = st.tabs(state.user_classes)
            for i, c in enumerate(state.user_classes):
                with coord_tabs[i]:
                    txt = st.text_area(f"Coordinates for {c}", key=f"manual_coords_{state.subset_rev}_{i}")
                    if st.button(f"➕ Add Parsed Rectangles to {c}", key=f"btn_add_coords_{i}"):
                        try:
                            add_confirmed(c, parse_coordinate_list(txt))
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Error parsing coordinates: {exc}")

        # =========================================================================
        # STAGE 6: VALIDATE MINIMUM PATCH REQUIREMENT
        # =========================================================================
        st.divider()
        counts = [len(state.confirmed_boxes.get(c, [])) for c in state.user_classes]
        remaining = sum(max(0, target_count - count) for count in counts)
        extra_boxes = one_crop and any(count > 1 for count in counts)
        can_run = not remaining and not extra_boxes
        st.caption(" · ".join(f"{c}: {n} boxes" for c, n in zip(state.user_classes, counts)))
        if remaining:
            st.info(f"Add {remaining} more {'box' if remaining == 1 else 'boxes'} to reach {target_count} per class.")
        if extra_boxes:
            st.info("In the review panel, keep one box per class to run the four-crop comparison.")
        confirmed = {label: [r["box"] for r in state.confirmed_boxes[label]] for label in state.user_classes}
        if st.button("📊 Calculate GLCM & Eigenvalues", type="primary", disabled=not (can_run and angles)):
            clear_results()
            try:
                with st.spinner("Quantizing confirmed patches and computing directional GLCMs…"):
                    coords = (confirmed if settings.exact_boxes else
                              sampled_confirmed_boxes(confirmed, settings, raster.width, raster.height))
                    patches = extract_samples(scene, replace(settings, exact_boxes=True), coords, {}, rgb_bands, state.user_classes)
                    analysis = analyze_samples(patches, settings)
                    state.texture_patches, state.analysis = patches, analysis
                    st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))

patches, a = state.texture_patches, state.analysis
if patches and a:
    st.subheader("Texture results")
    st.caption(f"{len(patches)} patches · {len(state.user_classes)} classes · five gray levels")
    walk, charts, details = st.tabs(["Follow one patch", "Compare textures", "Checks and downloads"])
    with walk:
        walkthrough(patches, settings)
    with charts:
        compare(patches, a)
    with details:
        checks_and_details(a, patches, preview, raster)
        st.markdown("#### Download results")
        if state.export is None:
            with st.spinner("Compiling complete results into academic research ZIP archive…"):
                bundle = work / "satellite_texture_analysis_package.zip"
                report = export_run(
                    scene, settings, patches, a, {} if settings.exact_boxes else confirmed,
                    confirmed if settings.exact_boxes else {}, rgb_bands, bundle
                )
                state.export = (bundle, report)
        bundle, report = state.export
        cols = st.columns(3)
        with bundle.open("rb") as file:
            cols[0].download_button(
                "📦 Download Complete ZIP Package",
                file,
                bundle.name,
                "application/zip",
                type="primary",
            )
        cols[1].download_button(
            "📄 Download Research Report (Markdown)",
            report,
            "satellite_texture_analysis_report.md",
            "text/markdown",
        )
        cols[2].download_button(
            "📊 Download Eigenvalues Table (CSV)",
            a["df_eigenvalues"].to_csv(index=False),
            "glcm_eigenvalues_table.csv",
            "text/csv",
        )
