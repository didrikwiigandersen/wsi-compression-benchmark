"""
Randomly samples 1000 tiles from a provided WSI in .ndpi format. Upon selecting a tile, it checks the corresponding
mask (.png) whether there is tissue there. If there is tissue, the tile's coordinates are selected and stored. The tile
is marked as selected. Otherwise, another tile is selected.

The sample is passed as an argument to the codec engines for compression. The process of tile sampling is separate
to ensure that the codec-engines work on the same data.
"""
from dataclasses import dataclass
# ---------------- Packages --------------------
from typing import List, Dict, Tuple

import openslide
import numpy as np
import math
import csv
from PIL import Image

# ---------------- Global variables --------------------
TILE_SIZE = 256
NUM_TILES = 1000
SEED = 42
MIN_TISSUE_FRAC = 1
MAX_ATTEMPTS = 1_000_000
MAX_IOU = 0
WRITE_CSV = True


# ---------------- Classes --------------------
@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    w: int
    h: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h
        }

# ---------------- Helper functions --------------------
def _load_mask_boolean(mask_png_path: str) -> np.ndarray:
    """
    Load a mask PNG as a boolean array (True = tissue).
    Any nonzero grayscale value is considered tissue.
    """
    mask_img = Image.open(mask_png_path).convert('L') # grayscale
    mask_arr = np.array(mask_img, dtype=np.uint8)
    tissue = mask_arr > 0
    return tissue

def _slide_mask_scales(slide: openslide.OpenSlide, mask_bool: np.ndarray) -> Tuple[float, float]:
    """
    Compute scale factors to map slide level-0 coords to mask indices.
    mask_x = round(slide_x * sx), mask_y = round(slide_y * sy)
    """
    slide_w, slide_h = slide.dimensions  # level-0 size
    mask_h, mask_w = mask_bool.shape
    sx = mask_w / float(slide_w)
    sy = mask_h / float(slide_h)
    return sx, sy

def _mask_rect_has_tissue(mask_bool: np.ndarray, x0: int, y0: int, w: int, h: int, sx: float, sy: float) -> Tuple[bool, int, int]:
    """
    Given a slide-space rectangle (x0,y0,w,h), check if ANY mask pixel in the corresponding
    mask-space rectangle is tissue. Returns (has_tissue, tissue_count, examined_count).
    """
    # Map slide rect to mask rect
    mx0 = max(0, int(math.floor(x0 * sx)))
    my0 = max(0, int(math.floor(y0 * sy)))
    mx1 = min(mask_bool.shape[1], int(math.ceil((x0 + w) * sx)))
    my1 = min(mask_bool.shape[0], int(math.ceil((y0 + h) * sy)))

    if mx1 <= mx0 or my1 <= my0:  # degenerate mapping (shouldn't happen)
        return False, 0, 0

    patch = mask_bool[my0:my1, mx0:mx1]
    tissue_count = int(patch.sum())
    examined = patch.size
    return (tissue_count > 0), tissue_count, examined

def _iou_rect(ax, ay, aw, ah, bx, by, bw, bh) -> float:
    ax1, ay1 = ax + aw, ay + ah
    bx1, by1 = bx + bw, by + bh
    inter_w = max(0, min(ax1, bx1) - max(ax, bx))
    inter_h = max(0, min(ay1, by1) - max(ay, by))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    union = aw*ah + bw*bh - inter
    return inter / union


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

        # Optional sanity: warn if aspect ratios differ a lot
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
            chosen.append(Tile(x=x0, y=y0, w=TILE_SIZE, h=TILE_SIZE))

        if len(chosen) < NUM_TILES:
            raise RuntimeError(
                f"Only found {len(chosen)} unique tiles with tissue after {attempts} attempts. "
                f"Consider reducing TILE_SIZE or NUM_TILES, or verify the mask coverage."
            )

        if WRITE_CSV:
            csv_path = slide_path + ".tile_coords.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["tile_id", "x", "y", "w", "h"])
                writer.writeheader()
                for i, t in enumerate(chosen):
                    row = {"tile_id": i}
                    row.update(t.as_dict())
                    writer.writerow(row)
            print(f"[tile-sampler] Wrote {len(chosen)} tiles to {csv_path} (seed={SEED}).")

        return chosen

    finally:
        # Ensure file handle is closed even on exceptions
        slide.close()