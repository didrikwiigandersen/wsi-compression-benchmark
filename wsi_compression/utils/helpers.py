"""
Utility file with helper functions used in tile_sampler.py and tile_visualizer.py.
"""

# ---------------- Packages --------------------
from wsi_compression.utils.classes.Tile import Tile
from skimage.metrics import structural_similarity as ssim

import numpy as np
import openslide
from PIL import Image
from typing import Tuple
import math
import io

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

# ------------------- Helpers for engines --------------------- #
def _raw_bytes(w: int, h: int) -> int:
    # 8 bits/channel, 3 channels => 3 bytes per pixel
    return int(w) * int(h) * 3

def _read_tile_rgb(slide: openslide.OpenSlide, t: Tile) -> np.ndarray:
    # OpenSlide returns RGBA; convert to RGB, uint8
    return np.array(slide.read_region((t.x, t.y), 0, (t.w, t.h)).convert("RGB"), dtype=np.uint8)

def _encode_jpeg_to_bytes(arr: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(
        buf,
        format="JPEG",
        quality=quality,
        subsampling=0,   # 4:4:4
        optimize=False,
        progressive=False,
    )
    return buf.getvalue()

def _ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    return ssim(a, b, channel_axis=2, data_range=255)