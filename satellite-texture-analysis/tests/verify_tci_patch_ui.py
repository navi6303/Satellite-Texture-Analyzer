"""Upload a window of the real TCI and compare the Streamlit occurrence table to source pixels.

Usage: python tests/verify_tci_patch_ui.py "/path/to/TCI (3).tif"
"""
import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.windows import Window
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def verify(tci):
    root = Path(__file__).resolve().parents[2]
    # The upload is a native-sized crop of the real TCI. Copy the source pixels and
    # mask without resizing. Ground truth below comes directly from the original.
    with rasterio.open(tci) as src, TemporaryDirectory() as tmp:
        window = Window(6800, 8100, 1024, 768)
        data = src.read(window=window, masked=True)
        path = Path(tmp) / "real_TCI_window.tif"
        with rasterio.open(path, "w", driver="GTiff", count=3, dtype="uint8", crs=src.crs,
                           width=1024, height=768, transform=src.window_transform(window)) as out:
            out.write(data.data)
            out.write_mask(~np.ma.getmaskarray(data).any(axis=0))
        ref = src.read(window=Window(6817, 8123, 700, 520), masked=True)
        upload = path.read_bytes()

    valid = ~np.ma.getmaskarray(ref).any(axis=0)
    pan = ref.data.mean(axis=0, dtype=np.float32)
    lo, hi = pan[valid].min(), pan[valid].max()
    display = np.clip(np.rint((pan - lo) * float(255 / (hi - lo))), 0, 255).astype("uint8")
    cumulative = np.bincount(display[valid], minlength=256).cumsum()
    first = cumulative[cumulative > 0][0]
    lut = np.clip(np.rint((cumulative-first) * (255 / (cumulative[-1]-first))), 0, 255)
    eq = lut[display].astype("uint8")

    # The separate canvas interaction test exercises the browser-side generation of
    # this same event. AppTest supplies its value to the production confirmation path.
    pending = {"Image subset": (17, 23, 700, 520)}
    def canvas(image, width, height, category, *args, **kwargs):
        if category in pending:
            x, y, w, h = pending.pop(category)
            return dict(x=x, y=y, w=w, h=h, confirmed=True, category=category)

    def check(app):
        assert not app.exception and not app.error, ([e.value for e in app.exception], [e.value for e in app.error])

    def click(app, label):
        next(b for b in app.button if label in b.label).click().run()
        check(app)

    urban = "Urban / Built-up Area"
    with patch("modules.snip_tool.snip_canvas", canvas):
        app = AppTest.from_file(str(root/"app.py"), default_timeout=120).run()
        app.file_uploader[0].upload("real_TCI_window.tif", upload).run()
        click(app, "Confirm Subset and Generate")
        click(app, "Enhance Contrast")
        next(s for s in app.selectbox if "Assign Bounding Box" in s.label).set_value(urban)
        pending[urban] = (472, 320, 12, 8)
        app.run()
        check(app)
        for i, label in enumerate(app.session_state["user_classes"]):
            coords = ("484,320,9,6\n499,320,12,8" if label == urban else
                      "\n".join(f"{20+44*j},{10+48*i},16,12" for j in range(3)))
            next(w for w in app.text_area if w.label == "Coordinates for " + label).set_value(coords)
            click(app, "Add Parsed Rectangles to " + label)
        click(app, "Calculate GLCM & Eigenvalues")
        assert [t.label for t in app.tabs if t.label in ("Follow one patch", "Compare textures", "Checks and downloads")] == ["Follow one patch", "Compare textures", "Checks and downloads"]
        app.selectbox(key="walk_class").set_value(urban).run()
        check(app)
        ptable = lambda: next(d.value for d in app.dataframe if "Raw sum" in d.value.columns)
        summary = ptable()
        assert len(summary) == 48
        for idx, (x, w, h) in enumerate([(472, 12, 8), (484, 9, 6)]):
            selector = next(s for s in app.selectbox if s.label == "Texture patch")
            if idx:
                selector.select_index(idx).run()
                check(app)
            pid = ["patch_3", "patch_3_02"][idx]
            p = next(p for p in app.session_state["texture_patches"] if p.patch_id == pid)
            np.testing.assert_array_equal(p.image_8bit, eq[320:320+h, x:x+w])
            q = np.digitize(eq[320:320+h, x:x+w], [52, 103, 154, 205])
            mask = valid[320:320+h, x:x+w]
            assert mask.all()
            expected = np.zeros((5, 5), dtype=int)
            for row in range(h):
                for col in range(w-1):
                    expected[q[row, col], q[row, col+1]] += 1
            # Only the selected patch and direction appear in the walkthrough.
            tables = [d.value for d in app.dataframe if "Occurrences" in d.value.columns]
            assert len(tables) == 1
            actual = dict(zip(tables[0]["Pair (i, j)"], tables[0]["Occurrences"]))
            assert actual == {f"({i}, {j})": expected[i,j] for i,j in np.ndindex(5,5) if expected[i,j]}
            r = summary[(summary["Patch ID"] == pid) & (summary["θ (°)"] == 0)].iloc[0]
            assert (r["Subset x"], r["Subset y"], r["Original x"], r["Original y"]) == (x,320,x+17,343)
            n = h * (w-1)
            assert r["Valid pairs"] == r["Raw sum"] == expected.sum() == n
            assert r["Symmetric sum"] == 2*n and np.isclose(r["Normalized sum"], 1)
            print(f"PASS {pid}: TCI original ({6817+x}, 8443), {w}×{h}, {mask.sum()} valid pixels, "
                  f"0° d=1, raw {n}, symmetric {2*n}, normalized {r['Normalized sum']:.6f}. "
                  f"All {np.count_nonzero(expected)} nonzero occurrence cells matched.")
        app.selectbox(key="walk_angle").set_value(90).run()
        check(app)
        selected = app.session_state["texture_patches"]
        p = next(p for p in selected if p.patch_id == "patch_3_02")
        counts = next(d.value for d in app.dataframe if "Occurrences" in d.value.columns)
        assert counts["Occurrences"].sum() == (p.height-1) * p.width
        next(r for r in app.radio if r.label == "Show").set_value("Class mean").run()
        check(app)
        assert app.session_state["analysis"]["check_bundle"]["passed"]
        print("PASS: one-patch walkthrough, direction switch, class-mean view and held-out checks.")
        next(w for w in app.multiselect if "Angles" in w.label).set_value([90]).run()
        check(app)
        assert app.session_state["analysis"] is None
        click(app, "Calculate GLCM & Eigenvalues")
        p = app.session_state["texture_patches"][0]
        np.testing.assert_allclose(p.glcm, p.extra["directional"][0]["probability"])
        assert any("Only 90°" in c.value for c in app.caption)
        print("PASS: changing to a single direction recomputes results and explains the one-matrix average.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tci", type=Path)
    verify(parser.parse_args().tci)
