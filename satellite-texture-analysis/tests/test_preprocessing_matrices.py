"""Check the displayed preprocessing against the actual image products."""
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from streamlit.testing.v1 import AppTest


def test_color_matrix_walkthrough_uses_full_subset(tmp_path):
    y, x = np.indices((24, 32))
    rgb = np.stack([9*x % 256, 11*y % 256, (x+7*y) % 256], axis=-1).astype("uint8")
    path = tmp_path / "colors.png"
    Image.fromarray(rgb).save(path)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "app.py")).run()
    app.file_uploader[0].upload(path.name, path.read_bytes()).run()
    next(w for w in app.text_input if w.label.startswith("Original x")).set_value("3,2,24,20")
    next(b for b in app.button if b.label == "Use these subset coordinates").click().run()
    assert not app.exception
    assert app.session_state["scene"] is None
    next(b for b in app.button if "Generate Grayscale Image" in b.label).click().run()
    assert not app.exception
    scene = app.session_state["scene"]
    roi = rgb[2:22, 3:27].astype(np.float32)
    expected = .114*roi[:,:,2] + .587*roi[:,:,1] + .299*roi[:,:,0]
    with rasterio.open(scene.folder / "native.tif") as src:
        np.testing.assert_allclose(src.read(1), expected, atol=1e-5)
    channel_tabs = {tab.label: tab for tab in app.tabs if tab.label in ("B", "G", "R")}
    assert list(channel_tabs) == ["B", "G", "R"]
    for channel, i in [("B", 2), ("G", 1), ("R", 0)]:
        np.testing.assert_array_equal(channel_tabs[channel].dataframe[0].value, roi[:10, :10, i])
    example = next(d.value for d in app.dataframe if list(d.value.columns) == ["B", "G", "R", "Grayscale"])
    np.testing.assert_equal(example.iloc[0][["B","G","R"]].to_numpy(), roi[0,0,::-1])
    np.testing.assert_allclose(example.iloc[0]["Grayscale"], expected[0,0])

    next(b for b in app.button if "Enhance Contrast" in b.label).click().run()
    assert not app.exception
    scene = app.session_state["scene"]
    mapping = next(d.value for d in app.dataframe if "Cumulative count" in d.value.columns)
    assert mapping["Pixel count"].sum() == 24*20
    assert any("the lookup table maps" in item.value for item in app.caption)
    np.testing.assert_array_equal(mapping["Cumulative count"], scene.hist_before.cumsum())
    np.testing.assert_array_equal(mapping["Equalized intensity"], scene.equalization_lut)
    with rasterio.open(scene.folder / "pan.tif") as pan, rasterio.open(scene.folder / "eq.tif") as eq:
        np.testing.assert_array_equal(scene.hist_before, np.bincount(pan.read(1).ravel(), minlength=256))
        np.testing.assert_array_equal(eq.read(1), scene.equalization_lut[pan.read(1)])
        np.testing.assert_array_equal(scene.hist_after, np.bincount(eq.read(1).ravel(), minlength=256))
    next(w for w in app.selectbox if w.label == "Matrix preview size").set_value(30).run()
    assert not app.exception
    assert any(d.value.shape == (20,24) for d in app.dataframe)


def test_single_band_png_keeps_native_intensity(tmp_path):
    from modules.image_loader import load_raster
    from modules.scene import prepare_scene
    from modules.subset import crop_original

    data = np.arange(40, dtype="uint16").reshape(5, 8) * 300
    path = tmp_path / "gray.png"
    Image.fromarray(data).save(path)
    raster = load_raster(path, path.name)
    subset = crop_original(raster, dict(x=2,y=1,w=4,h=3), tmp_path / "subset.tif")
    scene = prepare_scene(subset, [0], tmp_path / "scene")
    assert scene.provenance["conversion"] == "single_band"
    with rasterio.open(scene.folder / "native.tif") as src:
        np.testing.assert_array_equal(src.read(1), data[1:4,2:6])
