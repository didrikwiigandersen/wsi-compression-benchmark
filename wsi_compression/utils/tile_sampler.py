"""
Randomly samples 1000 tiles from a provided WSI in .ndpi format. Upon selecting a tile, it checks the corresponding
mask (.png) whether there is tissue there. If there is tissue, the tile's coordinates are selected and stored. The tile
is marked as selected. Otherwise, another tile is selected.

The sample is passed as an argument to the codec engines for compression. The process of tile sampling is separate
to ensure that the codec-engines work on the same data.
"""

# ---------------- Packages --------------------
from wsi_compression.utils.classes.Tile import Tile
from wsi_compression.utils.helpers import (
    _load_mask_boolean,
    _slide_mask_scales,
    _mask_rect_has_tissue,
    _iou_rect
)

from typing import List
import openslide
import numpy as np

# ---------------- Global variables (for tuning) --------------------
TILE_SIZE = 256
NUM_TILES = 50
SEED = 42
MIN_TISSUE_FRAC = 1
MAX_ATTEMPTS = 1_000_000
MAX_IOU = 0

# ---------------- Main --------------------
def sample_tiles_with_mask(slide_path: str, mask_png_path: str) -> List[Tile]:
    """
    Samples 1000 unique tiles from 'slide_path', using 'mask_png_path' to ensure tissue presence.
    Returns: List[Tile] (level-0 coordinates). Also writes CSV '<slide_path>.tile_coords.csv'.
    """
    slide = openslide.OpenSlide(slide_path)
    try:
        slide_w, slide_h = slide.dimensions
        if TILE_SIZE > slide_w or TILE_SIZE > slide_h:
            raise ValueError(
                f"TILE_SIZE={TILE_SIZE} is larger than slide dimensions {slide_w}x{slide_h}."
            )

        # ---- Load mask & scales ----
        mask_bool = _load_mask_boolean(mask_png_path)
        sx, sy = _slide_mask_scales(slide, mask_bool)

        # Warn if aspect ratios differ a lot
        mask_h, mask_w = mask_bool.shape
        slide_ar = slide_w / max(1.0, float(slide_h))
        mask_ar = mask_w / max(1.0, float(mask_h))
        if abs(slide_ar - mask_ar) > 0.05:  # >5% AR gap
            print("[tile-sampler] Warning: slide and mask aspect ratios differ >5%. "
                  "Ensure mask aligns to slide; scaling is used but may be approximate.")

        tissue_ys, tissue_xs = np.nonzero(mask_bool)
        if tissue_xs.size == 0:
            raise ValueError("Mask has no tissue pixels (all zeros).")

        rng = np.random.default_rng(SEED)
        chosen: List[Tile] = []
        chosen_set = set()

        attempts = 0
        while len(chosen) < NUM_TILES and attempts < MAX_ATTEMPTS:
            attempts += 1

            # Pick a random tissue pixel in mask-space
            idx = rng.integers(0, tissue_xs.size)
            mx = int(tissue_xs[idx])  # mask x (col)
            my = int(tissue_ys[idx])  # mask y (row)

            # Map to slide-space
            px = int(round(mx / sx))
            py = int(round(my / sy))

            # Choose a tile top-left so (px,py) lies inside the tile
            x0_lo = px - TILE_SIZE + 1
            x0_hi = px
            y0_lo = py - TILE_SIZE + 1
            y0_hi = py

            x0 = int(rng.integers(x0_lo, x0_hi + 1))
            y0 = int(rng.integers(y0_lo, y0_hi + 1))
            x0 = max(0, min(x0, slide_w - TILE_SIZE))
            y0 = max(0, min(y0, slide_h - TILE_SIZE))

            key = (x0, y0)
            if key in chosen_set:
                continue

            has_tissue, tissue_count, examined = _mask_rect_has_tissue(
                mask_bool, x0, y0, TILE_SIZE, TILE_SIZE, sx, sy
            )
            coverage = (tissue_count / examined) if examined > 0 else 0.0
            if not has_tissue or coverage < MIN_TISSUE_FRAC:
                continue

            ok_overlap = True
            for t in chosen:  # chosen is your List[Tile]
                if _iou_rect(x0, y0, TILE_SIZE, TILE_SIZE, t.x, t.y, t.w, t.h) > MAX_IOU:
                    ok_overlap = False
                    break
            if not ok_overlap:
                continue  # resample

            # Accept tile
            chosen_set.add(key)
            tile_id = len(chosen)
            chosen.append(Tile(id=tile_id, x=x0, y=y0, w=TILE_SIZE, h=TILE_SIZE, area=coverage))

        if len(chosen) < NUM_TILES:
            raise RuntimeError(
                f"Only found {len(chosen)} unique tiles with tissue after {attempts} attempts. "
                f"Consider reducing TILE_SIZE or NUM_TILES, or verify the mask coverage."
            )
        return chosen

    finally:
        # Ensure file handle is closed even on exceptions
        slide.close()
