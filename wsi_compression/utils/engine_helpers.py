"""
Helper functions used in the engines.
"""
# ------------------- Packages --------------------- #
from wsi_compression.utils.classes.Tile import Tile
from skimage.metrics import structural_similarity as ssim
from typing import Tuple

import io
import numpy as np
import openslide
from PIL import Image
import tempfile
import subprocess
import os

# ------------------- General helpers for engines --------------------- #
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

# ------------------- Helpers for jxl_engine.py --------------------- #

def _encode_jxl_bytes_from_rgb(rgb: np.ndarray, distance: float, effort: int=7) -> bytes:
    """
    Encode an RGB uint8 array to JXL bytes using cjxl.
    """
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as pngf:
        Image.fromarray(rgb, mode="RGB").save(pngf.name, format="PNG", optimize=False)
        png_path = pngf.name
    jxl_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".jxl", delete=False) as jxlf:
            jxl_path = jxlf.name

        cmd = [
            "cjxl", png_path, jxl_path,
            "--distance", str(float(distance)),
            "-e", str(int(effort)),
            "--quiet",
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with open(jxl_path, "rb") as f:
            data = f.read()
        return data

    finally:
        try: os.remove(png_path)  # png_path always exists
        except OSError: pass
        if jxl_path and os.path.exists(jxl_path):
            try: os.remove(jxl_path)
            except OSError: pass

def _decode_jxl_bytes_to_rgb(jxl_bytes: bytes) -> np.ndarray:
    """Decode JXL bytes to RGB uint8 via djxl → PNG temp."""
    with tempfile.NamedTemporaryFile(suffix=".jxl", delete=False) as jxlf:
        jxlf.write(jxl_bytes)
        jxlf.flush()
        jxl_path = jxlf.name

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as pngf:
        png_path = pngf.name

    try:
        subprocess.run(
            ["djxl", jxl_path, png_path, "--quiet"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        rec = np.array(Image.open(png_path).convert("RGB"), dtype=np.uint8)
        return rec
    finally:
        try: os.remove(jxl_path)
        except OSError: pass
        try: os.remove(png_path)
        except OSError: pass

def _ssim_for_distance(rgb: np.ndarray, distance: float, effort: int) -> Tuple[float, bytes]:
    """Encode at given distance, decode, compute SSIM vs original. Returns (ssim, jxl_bytes)."""
    jxl_bytes = _encode_jxl_bytes_from_rgb(rgb, distance=distance, effort=effort)
    recon = _decode_jxl_bytes_to_rgb(jxl_bytes)
    ssim_val = float(_ssim_rgb(rgb, recon))
    return ssim_val, jxl_bytes


def _match_ssim_bisection(
    rgb: np.ndarray,
    target_ssim: float,
    tol: float = 1e-3,
    max_iters: int = 12,
    effort: int = 7,
    dist_lo_init: float = 0.0,
    dist_hi_init: float = 3.0,
    dist_hi_max: float = 6.0,
) -> Tuple[float, float, bytes]:
    """
    Find distance so that SSIM(distance) ~= target_ssim within ±tol via bisection.
    Returns (best_distance, best_ssim, best_bytes). If exact tol not achieved,
    returns the closest observed.

    :param rgb:
    :param target_ssim:
    :param tol:
    :param max_iters:
    :param effort:
    :param dist_lo_init:
    :param dist_hi_init:
    :param dist_hi_max:
    :return:
    """
    # Evaluate ends
    lo_d = float(dist_lo_init)
    hi_d = float(dist_hi_init)

    lo_s, lo_b = _ssim_for_distance(rgb, lo_d, effort)  # higher SSIM expected
    hi_s, hi_b = _ssim_for_distance(rgb, hi_d, effort)  # lower SSIM expected

    # If hi side still above target, expand upward (decrease quality)
    while hi_d < dist_hi_max and hi_s > target_ssim + tol:
        hi_d = max(0.5, hi_d * 1.6)  # move away from 0, then expand
        hi_s, hi_b = _ssim_for_distance(rgb, hi_d, effort)

    # Keep the best-so-far (closest to target)
    best_d, best_s, best_b = lo_d, lo_s, lo_b
    if abs(hi_s - target_ssim) < abs(best_s - target_ssim):
        best_d, best_s, best_b = hi_d, hi_s, hi_b
    if abs(lo_s - target_ssim) <= tol:
        return lo_d, lo_s, lo_b
    if abs(hi_s - target_ssim) <= tol:
        return hi_d, hi_s, hi_b

    # Ensure invariant: lo is the higher-SSIM end
    if lo_s < hi_s:
        lo_d, hi_d = hi_d, lo_d
        lo_s, hi_s = hi_s, lo_s
        lo_b, hi_b = hi_b, lo_b

    # Bisection
    for _ in range(max_iters):
        mid_d = 0.5 * (lo_d + hi_d)
        mid_s, mid_b = _ssim_for_distance(rgb, mid_d, effort)

        # Track best
        if abs(mid_s - target_ssim) < abs(best_s - target_ssim):
            best_d, best_s, best_b = mid_d, mid_s, mid_b

        if abs(mid_s - target_ssim) <= tol:
            return mid_d, mid_s, mid_b

        # Decide side (remember: lo_s >= hi_s)
        if mid_s >= target_ssim:
            lo_d, lo_s, lo_b = mid_d, mid_s, mid_b
        else:
            hi_d, hi_s, hi_b = mid_d, mid_s, mid_b

        if abs(hi_d - lo_d) < 1e-6:
            break

    # Not within tol: return closest observed
    return best_d, best_s, best_b