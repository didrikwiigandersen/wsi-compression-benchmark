"""
Helpers for J2K engine.
"""

# ------------------- Packages --------------------- #
import numpy as np
from typing import Tuple
import subprocess
import tempfile
from PIL import Image
import os

from wsi_compression.utils.engines.jpeg_helpers import (
    ssim_rgb
)
# ------------------- Helper Functions --------------------- #
def encode_j2k_bytes_from_rgb(rgb: np.ndarray, rate: float) -> bytes:
    """
    Encode RGB -> JP2 using OpenJPEG CLI with a lossless PPM temp.
    """
    # ------- Assertion and setup --------- #
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    ppm_path = None
    jp2_path = None
    try:
        # Create a .ppm file from the RGB array passed in
        with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as ppmp:
            ppm_path = ppmp.name
        Image.fromarray(rgb, "RGB").save(ppm_path, format="PPM")

        # Create a .jp2 file to store the encoded file
        with tempfile.NamedTemporaryFile(suffix=".jp2", delete=False) as jp2f:
            jp2_path = jp2f.name

        # Create the command to encode the tile
        cmd = [
            "opj_compress",
            "-i",
            ppm_path,
            "-o",
            jp2_path,
            "-r",
            str(float(rate)),
            "-quiet"
        ]

        # Run the command
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Return the raw compressed bytes
        with open(jp2_path, "rb") as f:
            return f.read()
    except FileNotFoundError as e:
        raise RuntimeError("opj_compress not found on PATH. Install OpenJPEG or add it to PATH.") from e
    finally:
        # Remove the created files
        for p in (ppm_path, jp2_path):
            if p and os.path.exists(p):
                try: os.remove(p)
                except OSError: pass

def decode_j2k_bytes_to_rgb(jp2_bytes: bytes) -> np.ndarray:
    """Decode JP2 -> RGB uint8 via OpenJPEG CLI using a PPM temp (lossless)."""

    # Create temporary .jp2 file
    with tempfile.NamedTemporaryFile(suffix=".jp2", delete=False) as jp2f:
        jp2f.write(jp2_bytes); jp2f.flush()
        jp2_path = jp2f.name

    # Create temporary .ppm file
    with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as ppmp:
        ppm_path = ppmp.name

    try:
        # Creating the command to decode the tile
        cmd = [
            "opj_decompress",
            "-i",
            jp2_path,
            "-o",
            ppm_path,
            "-quiet"
        ]

        # Running the command
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Returning the raw bytes of the decoded image
        return np.array(Image.open(ppm_path).convert("RGB"), dtype=np.uint8)
    except FileNotFoundError as e:
        raise RuntimeError("opj_decompress not found on PATH. Install OpenJPEG or add it to PATH.") from e
    finally:
        # Cleaning up created files
        for p in (jp2_path, ppm_path):
            try: os.remove(p)
            except OSError: pass

def ssim_for_rate(rgb: np.ndarray, rate: float) -> Tuple[float, bytes]:
    """Encode at given JP2 `rate`, decode, compute SSIM vs original. Returns (ssim, jp2_bytes)."""
    jp2_bytes = encode_j2k_bytes_from_rgb(rgb, rate=rate)
    recon = decode_j2k_bytes_to_rgb(jp2_bytes)
    return ssim_rgb(rgb, recon), jp2_bytes # structural similarity

def match_ssim_bisection_rate(
    rgb: np.ndarray,
    target_ssim: float,
    tol: float = 1e-3,
    max_iters: int = 12,
    rate_lo_init: float = 1.0,
    rate_hi_init: float = 600.0,
    rate_hi_max: float = 1200.0,
) -> Tuple[float, float, bytes]:
    """
    Find a JP2 `rate` such that SSIM(rate) ~= target_ssim within ±tol via bisection.
    Monotonicity (empirical): rate ↑ ⇒ quality ↓ ⇒ SSIM ↓.

    :param rgb:
    :param target_ssim:
    :param tol:
    :param max_iters:
    :param rate_lo_init:
    :param rate_hi_init:
    :param rate_hi_max:
    :return: (best_rate, best_ssim, best_bytes) — if tol not met, returns closest observed.
    """
    lo_r = float(rate_lo_init)   # higher quality
    hi_r = float(rate_hi_init)   # lower quality

    lo_s, lo_b = ssim_for_rate(rgb, lo_r)
    hi_s, hi_b = ssim_for_rate(rgb, hi_r)

    # Expand upper side if still too good (above target + tol)
    while hi_r < rate_hi_max and hi_s > target_ssim + tol:
        hi_r = max(hi_r * 1.6, hi_r + 1.0)
        hi_s, hi_b = ssim_for_rate(rgb, hi_r)

    # Track best so far
    best_r, best_s, best_b = (lo_r, lo_s, lo_b)
    if abs(hi_s - target_ssim) < abs(best_s - target_ssim):
        best_r, best_s, best_b = (hi_r, hi_s, hi_b)

    if abs(lo_s - target_ssim) <= tol:
        return lo_r, lo_s, lo_b
    if abs(hi_s - target_ssim) <= tol:
        return hi_r, hi_s, hi_b

    # Ensure invariant: lo is higher-SSIM side
    if lo_s < hi_s:
        lo_r, hi_r = hi_r, lo_r
        lo_s, hi_s = hi_s, lo_s
        lo_b, hi_b = hi_b, lo_b

    for _ in range(max_iters):
        mid_r = 0.5 * (lo_r + hi_r)
        mid_s, mid_b = ssim_for_rate(rgb, mid_r)

        if abs(mid_s - target_ssim) < abs(best_s - target_ssim):
            best_r, best_s, best_b = (mid_r, mid_s, mid_b)

        if abs(mid_s - target_ssim) <= tol:
            return mid_r, mid_s, mid_b

        # choose side (lo_s >= hi_s)
        if mid_s >= target_ssim:
            lo_r, lo_s, lo_b = (mid_r, mid_s, mid_b)
        else:
            hi_r, hi_s, hi_b = (mid_r, mid_s, mid_b)

        if abs(hi_r - lo_r) < 1e-6:
            break

    return best_r, best_s, best_b
