"""Independently reconstruct the supplied real TCI export from the original RGB pixels."""

import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

OFFSETS = [(0, 0, 1), (45, 1, 1), (90, 1, 0), (135, 1, -1)]
EIGEN_COLUMNS = ["λ1 (Dominant)", "λ2", "λ3", "λ4", "λ5"]


def table(archive, name):
    return list(csv.DictReader(io.StringIO(archive.read(f"tables/{name}.csv").decode())))


def equalization_by_rgb_sum(src, roi):
    # Eight-bit RGB has only 766 possible unweighted band sums. Accumulate these
    # directly from the original file, independently of all application code.
    x, y, w, h = (roi[k] for k in ("x", "y", "w", "h"))
    sums = np.zeros(766, dtype=np.int64)
    for row in range(0, h, 128):
        data = src.read(window=Window(x, y + row, w, min(128, h - row)), masked=True)
        valid = ~np.ma.getmaskarray(data).any(axis=0)
        totals = data.data.sum(axis=0, dtype="uint16")
        sums += np.bincount(totals[valid], minlength=766)
    native = np.arange(766, dtype="float32") / 3
    low, high = native[sums > 0][[0, -1]]
    scale = 255.0 / float(high - low)
    display = np.clip(np.rint((native - low) * scale), 0, 255).astype("uint8")
    cumulative = np.bincount(display, weights=sums, minlength=256).cumsum()
    first = cumulative[cumulative > 0][0]
    lut = np.clip(np.rint((cumulative - first) * 255 / (cumulative[-1] - first)), 0, 255)
    return lut.astype("uint8")[display], int(sums.sum())


def counts_for(q, valid, dy, dx):
    h, w = q.shape
    left, right = max(0, -dx), min(w, w - dx)
    a = np.s_[:h-dy, left:right]
    b = np.s_[dy:, left+dx:right+dx]
    paired = valid[a] & valid[b]
    codes = q[a][paired] * 5 + q[b][paired]
    return np.bincount(codes, minlength=25).reshape(5, 5)


def verify(source, archive, boxes):
    with rasterio.open(source) as src, ZipFile(archive) as z:
        assert z.testzip() is None
        meta = json.loads(z.read("metadata_summary.json"))
        roi = meta["source_subset"]
        assert src.dtypes == ("uint8",) * 3
        assert meta["original_dimensions"] == dict(width=src.width, height=src.height)
        assert meta["dimensions"] == dict(width=roi["w"], height=roi["h"])
        lut, valid_count = equalization_by_rgb_sum(src, roi)
        eigen = {r["Patch ID"]: r for r in table(z, "eigenvalues_summary")}
        directions = {(r["patch_id"], float(r["angle_deg"])): r for r in table(z, "directional_glcms")}
        expected = json.loads(Path(boxes).read_text())
        patches = meta["patch_coordinates"]
        assert len(patches) == sum(map(len, expected.values())) == 12
        for item in patches:
            pid = item["patch_id"]
            x, y, w, h = (item[k] for k in ("original_x", "original_y", "width", "height"))
            assert dict(x=x, y=y, w=w, h=h) in expected[item["label"]]
            assert (x, y) == (item["x"] + roi["x"], item["y"] + roi["y"])
            data = src.read(window=Window(x, y, w, h), masked=True)
            valid = ~np.ma.getmaskarray(data).any(axis=0)
            eq = lut[data.data.sum(axis=0, dtype="uint16")]
            eq[~valid] = 0
            label = item["label"].lower().replace(" ", "_").replace("/", "_")
            png = z.read(f"patches/{pid}_{label}_8bit.png")
            np.testing.assert_array_equal(eq, np.asarray(Image.open(io.BytesIO(png))))
            q = np.digitize(eq, [52, 103, 154, 205])
            matrices = []
            for angle, dy, dx in OFFSETS:
                counts = counts_for(q, valid, dy, dx)
                recorded = directions[pid, angle]
                assert (int(recorded["x"]), int(recorded["y"])) == (item["x"], item["y"])
                assert (int(recorded["original_x"]), int(recorded["original_y"])) == (x, y)
                assert (int(recorded["width"]), int(recorded["height"])) == (w, h)
                assert int(recorded["total_pixels"]) == w * h and int(recorded["valid_pixels"]) == valid.sum()
                assert counts.sum() == int(recorded["pairs"]) == int(recorded["raw_matrix_sum"])
                np.testing.assert_array_equal(counts, json.loads(recorded["raw_counts"]))
                assert (h-abs(dy))*(w-abs(dx)) == int(recorded["possible_pairs"])
                assert int(recorded["symmetric_matrix_sum"]) == 2 * counts.sum()
                symmetric = counts + counts.T
                np.testing.assert_array_equal(symmetric, json.loads(recorded["symmetric_counts"]))
                matrices.append(symmetric / symmetric.sum())
            values = np.linalg.eigvalsh(np.mean(matrices, axis=0))
            observed = [float(eigen[pid][column]) for column in EIGEN_COLUMNS]
            np.testing.assert_allclose(np.sort(observed), values, atol=1e-12)
        print(f"PASS: {valid_count:,} valid subset pixels, 12 original-coordinate patch images, "
              "48 directional matrices and pair denominators, and all 60 eigenvalues.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--boxes", type=Path, default=Path(__file__).resolve().parents[1]
                        / "sample_data" / "tci_validation_boxes.json")
    args = parser.parse_args()
    verify(args.source, args.archive, args.boxes)
