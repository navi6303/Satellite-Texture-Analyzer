"""
Multi-patch sampling helpers: coordinate lists, non-overlapping random crops,
and spectral region suggestions for batch analysis of True-Colour GeoTIFFs.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from modules.texture_extraction import validate_crop


def parse_coordinate_list(text: str) -> List[Dict[str, int]]:
    """
    Parse lines of the form ``x, y, size`` or ``x, y, width, height``.
    Blank lines and comments starting with # are ignored.
    """
    rows: List[Dict[str, int]] = []
    if not text:
        return rows
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.replace(";", ",").split(",") if p.strip()]
        if len(parts) not in (3, 4):
            raise ValueError(f"Need at least x, y, size on each line; got: {raw!r}")
        nums = [int(p) for p in parts]
        x, y = nums[0], nums[1]
        if len(nums) == 3:
            size = nums[2]
            rows.append({"x": x, "y": y, "size": size, "w": size, "h": size})
        else:
            w, h = nums[2], nums[3]
            rows.append({"x": x, "y": y, "size": min(w, h), "w": w, "h": h})
    return rows


def boxes_overlap(a: Dict[str, int], b: Dict[str, int], min_gap: int = 0) -> bool:
    aw = int(a.get("w", a.get("size", 0)))
    ah = int(a.get("h", a.get("size", 0)))
    bw = int(b.get("w", b.get("size", 0)))
    bh = int(b.get("h", b.get("size", 0)))
    return not (
        a["x"] + aw + min_gap <= b["x"]
        or b["x"] + bw + min_gap <= a["x"]
        or a["y"] + ah + min_gap <= b["y"]
        or b["y"] + bh + min_gap <= a["y"]
    )


def touched_blocks(box, block_size, scene_width):
    """Every grid cell intersected by a half-open native pixel rectangle."""
    columns = (scene_width + block_size - 1) // block_size
    x, y = box["x"], box["y"]
    w, h = box.get("w", box.get("size")), box.get("h", box.get("size"))
    return [
        r * columns + c
        for r in range(y // block_size, (y + h - 1) // block_size + 1)
        for c in range(x // block_size, (x + w - 1) // block_size + 1)
    ]


def assign_groups(patches, block_size, scene_width):
    """Join cells touched by a boundary-crossing patch before constructing folds."""
    parent = {}

    def root(cell):
        while parent.setdefault(cell, cell) != cell:
            parent[cell] = parent[parent[cell]]
            cell = parent[cell]
        return cell

    for p in patches:
        cells = touched_blocks(
            dict(x=p.x, y=p.y, w=p.width, h=p.height), block_size, scene_width
        )
        for cell in cells[1:]:
            a, b = root(cells[0]), root(cell)
            parent[max(a, b)] = min(a, b)
        p.extra["grid_cells"] = cells
    for p in patches:
        p.block_id = root(p.extra["grid_cells"][0])


def sample_layout(
    labels, explicit, regions, n, size, block, seed, width, height, *, exact=False, minimum_n=3
):
    """Reserve all manual rectangles first. Sample remaining squares inside the chosen ROI."""
    if block < 1 or (not exact and (n < minimum_n or size < 8)):
        raise ValueError(
            "Use at least three patches, crop side ≥8 and a positive block size."
        )
    layout = {lab: [b.copy() for b in explicit.get(lab, [])] for lab in labels}
    occupied = []

    def validate(box):
        w, h = box.get("w", box.get("size")), box.get("h", box.get("size"))
        ok, msg = validate_crop(box["x"], box["y"], w, h, width, height, min_size=1)
        if not ok:
            raise ValueError(msg)
        box.update(w=w, h=h)

    for lab, boxes in layout.items():
        if exact and not boxes:
            raise ValueError(f"{lab}: draw or enter at least one bounding box.")
        if not exact and len(boxes) > n:
            raise ValueError(f"{lab}: supplied {len(boxes)} coordinates for N={n}.")
        for box in boxes:
            validate(box)
            if any(boxes_overlap(box, other) for other in occupied):
                raise ValueError(
                    f"{lab}: manual patch overlaps a previously specified crop."
                )
            occupied.append(box)

    if exact:
        return layout

    for i, lab in enumerate(labels):
        boxes = layout[lab]
        if len(boxes) == n:
            continue
        if lab not in regions:
            raise ValueError(f"{lab}: supply {n} patches or draw a fallback region.")
        region = regions[lab].copy()
        validate(region)
        x0, y0 = region["x"], region["y"]
        x1, y1 = x0 + region["w"], y0 + region["h"]
        if x1 - x0 < size or y1 - y0 < size:
            raise ValueError(f"{lab}: sampling region is smaller than the crop side.")
        rng = np.random.default_rng(seed + i)
        # Prefer distinct blocks. Crops remain inside cells when geometrically possible.
        cells = []
        for by in range(y0 // block, (y1 - 1) // block + 1) if size <= block else ():
            for bx in range(x0 // block, (x1 - 1) // block + 1):
                lo_x, hi_x = max(x0, bx * block), min(x1, (bx + 1) * block) - size
                lo_y, hi_y = max(y0, by * block), min(y1, (by + 1) * block) - size
                if lo_x <= hi_x and lo_y <= hi_y:
                    cells.append((lo_x, hi_x, lo_y, hi_y))
        candidates = list(cells)
        rng.shuffle(candidates)
        for attempt in range(12000):
            if len(boxes) == n:
                break
            if candidates and attempt < max(1000, len(candidates)):
                lx, hx, ly, hy = candidates[attempt % len(candidates)]
            else:
                lx, hx, ly, hy = x0, x1 - size, y0, y1 - size
            box = dict(
                x=int(rng.integers(lx, hx + 1)),
                y=int(rng.integers(ly, hy + 1)),
                w=size,
                h=size,
            )
            if not any(boxes_overlap(box, other) for other in occupied):
                boxes.append(box)
                occupied.append(box)
        if len(boxes) != n:
            raise ValueError(
                f"{lab}: placed {len(boxes)}/{n} crops. Enlarge the region or reduce the crop side."
            )
    return layout
