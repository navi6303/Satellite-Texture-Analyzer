"""Reproduce the supplied TCI study, or run another raster with a region JSON file."""

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

from modules.export_utils import export_run
from modules.image_loader import load_raster, detect_satellite_band_mapping
from modules.subset import checked_box, crop_original
from modules.scene import prepare_scene, equalize_scene
from modules.workflow import Settings, extract_samples, analyze_samples, audit_regions, sampled_confirmed_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raster", type=Path)
    parser.add_argument(
        "--regions",
        type=Path,
        required=True,
        help="Mapping from class names to one native x,y,w,h rectangle or a list of rectangles.",
    )
    parser.add_argument(
        "--settings", type=Path, help="Optional JSON overrides for Settings."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("satellite_texture_analysis_package.zip")
    )
    parser.add_argument("--subset", nargs=4, type=int, metavar=("X","Y","WIDTH","HEIGHT"),
                        help="Original-image subset in zero-based native pixels. If omitted, use the smallest rectangle enclosing the supplied original-coordinate boxes.")
    args = parser.parse_args()
    regions = json.loads(args.regions.read_text())
    settings = Settings(
        **(json.loads(args.settings.read_text()) if args.settings else {})
    )
    r = load_raster(args.raster, args.raster.name)
    if len(regions) != 4 or any(not value for value in regions.values()):
        parser.error("Supply four named classes with at least one source rectangle each.")
    source_boxes = [checked_box(box, r.width, r.height) for value in regions.values()
                    for box in (value if isinstance(value, list) else [value])]
    if not source_boxes:
        parser.error("Supply original-image rectangles for four classes.")
    if args.subset:
        subset_box = checked_box(dict(zip(("x","y","w","h"), args.subset)), r.width, r.height)
    else:
        x = min(b["x"] for b in source_boxes)
        y = min(b["y"] for b in source_boxes)
        subset_box = dict(x=x, y=y, w=max(b["x"]+b["w"] for b in source_boxes)-x,
                          h=max(b["y"]+b["h"] for b in source_boxes)-y)
    def local(b):
        return checked_box(dict(b, x=b["x"]-subset_box["x"], y=b["y"]-subset_box["y"]),
                           subset_box["w"], subset_box["h"])
    regions = {label: [local(b) for b in value] if isinstance(value,list) else local(value)
               for label,value in regions.items()}
    if settings.exact_boxes:
        explicit = {
            lab: value if isinstance(value, list) else [value]
            for lab, value in regions.items()
        }
        if any(len(boxes) < 3 for boxes in explicit.values()):
            parser.error("Provide at least three confirmed patches per class, or use within_box_settings.json for region sampling.")
        sampling_regions = {}
    else:
        explicit, sampling_regions = {}, regions
    mapping = detect_satellite_band_mapping(r.count, r.metadata.band_descriptions)
    rgb = [mapping[k] for k in ("r_idx", "g_idx", "b_idx")] if r.count >= 3 else None
    with tempfile.TemporaryDirectory() as tmp:
        subset = crop_original(r, subset_box, Path(tmp) / "subset.tif")
        scene = equalize_scene(prepare_scene(subset, list(range(subset.count)), tmp))
        labels = list(regions)
        if not settings.exact_boxes and any(isinstance(v, list) for v in regions.values()):
            class_regions = {label: value if isinstance(value, list) else [value]
                             for label, value in regions.items()}
            coords = sampled_confirmed_boxes(class_regions, settings, subset.width, subset.height)
            patches = extract_samples(scene, replace(settings, exact_boxes=True), coords, {}, rgb, labels)
        else:
            patches = extract_samples(scene, settings, explicit, sampling_regions, rgb, labels)
        result = analyze_samples(patches, settings)
        references = audit_regions(scene, sampling_regions, rgb)
        export_run(
            scene,
            settings,
            patches,
            result,
            sampling_regions,
            explicit,
            rgb,
            args.output,
            references,
        )
        print(json.dumps({p.patch_id: p.label for p in patches}, ensure_ascii=True))
        for p in patches:
            print(
                f"{p.patch_id} ({p.x},{p.y},{p.width},{p.height}): P.shape={p.glcm.shape}, sum={p.glcm.sum():.12f}, lambda1={p.eigenvalues[0]:.8f}"
            )
        print(result["check_bundle"]["output"])
        print(result["interpretation"]["summary"])
        print(args.output.resolve())
        return 0 if result["check_bundle"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
