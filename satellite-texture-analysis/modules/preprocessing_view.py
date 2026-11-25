"""Small native pixel previews and full-subset histogram statistics."""
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from modules.scene import LUMA_WEIGHTS
import streamlit as st


def matrix_window(raster, side):
    return Window(0, 0, min(side, raster.width), min(side, raster.height))


def show_matrix(values, valid):
    frame = pd.DataFrame(values, columns=[str(i) for i in range(values.shape[1])])
    st.dataframe(frame.where(valid), width="stretch")


def subset_matrix(raster, state_key):
    with st.expander("Original subset pixel values", expanded=True):
        side = st.selectbox("Matrix preview size", [10, 20, 30], format_func=lambda n: f"Up to {n} × {n}", key=state_key)
        window = matrix_window(raster, side)
        names = [f"Band {i+1} · {label}" for i, label in enumerate(raster.band_names)]
        band = st.selectbox("Band to display", names, key=f"{state_key}_band") if len(names) > 1 else names[0]
        st.caption(f"Top-left {int(window.height)} × {int(window.width)} pixels at native resolution. "
                   f"The full subset is {raster.height:,} rows × {raster.width:,} columns × {raster.count} bands. "
                   "Processing uses the full subset. Blank cells are outside the valid analysis mask.")
        with rasterio.open(raster.file_path) as src:
            values = src.read(names.index(band) + 1, window=window, masked=True)
        show_matrix(values.data, ~np.ma.getmaskarray(values))
    return window


def conversion_matrices(scene, window):
    raster = scene.raster
    with rasterio.open(scene.folder / "native.tif") as src:
        converted = src.read(1, window=window)
    valid = np.isfinite(converted)
    method = scene.provenance["conversion"]
    with st.expander("How the subset becomes one intensity per pixel", expanded=True):
        if method == "single_band":
            st.caption("The input already has one intensity band. Its values are copied before the 8-bit stretch.")
        else:
            if method == "weighted_gray":
                bands = raster.metadata.extra["color_channels"]
                names = {color: bands[color] for color in ("B", "G", "R")}
                st.markdown("**Gray = 0.114 × B + 0.587 × G + 0.299 × R**")
                st.caption("B, G and R refer to the blue, green and red channels, regardless of their storage order.")
            else:
                names = {f"Band {i+1}": i for i in scene.bands}
                st.caption(f"Add the {len(names)} band values at each valid pixel and divide by {len(names)}.")
            with rasterio.open(raster.file_path) as src:
                if method == "weighted_gray":
                    inputs = src.read([i + 1 for i in names.values()], window=window)
                    for tab, color, values in zip(st.tabs(list(names)), names, inputs):
                        with tab:
                            st.caption(f"{color} channel at native resolution")
                            show_matrix(values, valid)
                else:
                    band = st.selectbox("Input band matrix", list(names))
                    show_matrix(src.read(names[band] + 1, window=window), valid)

                if valid.any():
                    row, col = map(int, np.argwhere(valid)[0])
                    chips = src.read([i + 1 for i in names.values()], window=Window(col, row, 1, 1))
                    example = dict(zip(names, chips[:, 0, 0]))
                    result = converted[row, col]
                    st.caption(f"Actual conversion at subset row {row}, column {col} (zero-based).")
                    if method == "weighted_gray":
                        terms = " + ".join(f"{LUMA_WEIGHTS[color]} × {value:g}" for color, value in example.items())
                        st.markdown(f"**Gray = {terms} ≈ {result:.6g}**")
                    else:
                        total = chips[:, 0, 0].astype(np.float64).sum()
                        st.markdown(f"**Band average = {total:.8g} ÷ {len(names)} ≈ {result:.6g}**")
                    example["Grayscale" if method == "weighted_gray" else "Band average"] = result
                    st.dataframe(pd.DataFrame([example]), hide_index=True)
                else:
                    st.caption("This preview contains no valid pixels for a worked example. Try a larger preview.")
        st.markdown("**Converted intensity matrix before the 8-bit stretch**")
        show_matrix(converted, valid)


def equalization_matrices(scene, window):
    st.caption("Histogram equalization improves image contrast by redistributing intensity values based on their cumulative histogram.")
    with st.expander("Pixel matrices, histograms and equalization mapping", expanded=True):
        with rasterio.open(scene.folder / "native.tif") as src:
            valid = np.isfinite(src.read(1, window=window))
        st.caption("Matrices show the same top-left pixels. The histograms and mapping use every valid pixel in the subset.")
        st.markdown("**8-bit intensity matrix before equalization**")
        st.caption("The full subset’s valid intensity range is stretched to 0–255 before building the histogram.")
        with rasterio.open(scene.folder / "pan.tif") as src:
            before = src.read(1, window=window)
        show_matrix(before, valid)
        st.markdown("**Histogram before equalization**")
        intensity_histogram(scene.hist_before)
        cumulative = scene.hist_before.cumsum()
        st.markdown("**Cumulative histogram**")
        st.caption("At each intensity, this line shows how many valid pixels have that intensity or lower.")
        st.line_chart(pd.DataFrame({"Cumulative count": cumulative}).rename_axis("8-bit intensity"),
                      x_label="8-bit intensity", y_label="Cumulative pixel count")
        st.markdown("**Intensity mapping used for equalization**")
        st.dataframe(pd.DataFrame({
            "Original intensity": np.arange(256),
            "Pixel count": scene.hist_before,
            "Cumulative count": cumulative,
            "Equalized intensity": scene.equalization_lut,
        }), hide_index=True, width="stretch")
        st.markdown("**Equalized intensity matrix**")
        with rasterio.open(scene.folder / "eq.tif") as src:
            after = src.read(1, window=window)
        show_matrix(after, valid)
        if valid.any():
            changed = valid & (before != after)
            row, col = np.argwhere(changed if changed.any() else valid)[0]
            st.caption(f"At row {row}, column {col}, the lookup table maps "
                       f"{before[row, col]} to {after[row, col]}.")
        st.markdown("**Histogram after equalization**")
        intensity_histogram(scene.hist_after)


def intensity_histogram(counts):
    frame = pd.DataFrame({"Intensity": np.arange(256), "Pixel count": counts})
    st.bar_chart(frame, x="Intensity", y="Pixel count", height=220)
