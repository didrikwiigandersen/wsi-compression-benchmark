"""
Helpers specific for the JXL Engine.
"""

# ----------------------- Packages ----------------------- #
from typing import Tuple, Sequence
import os
import shutil
import subprocess
import tempfile
import numpy as np
from PIL import Image

from wsi_compression.utils.engines.jpeg_helpers import (
    ssim_rgb
)

# ----------------------- Functions ----------------------- #
def ensure_cli_tools_or_raise(names: Sequence[str]) -> None:
    """Ensure required CLI tools exist on PATH."""
    missing = [n for n in names if shutil.which(n) is None]
    if missing:
        raise RuntimeError(f"Missing required CLI tool(s): {', '.join(missing)}. "
                           f"Install them and ensure they are on PATH.")

def encode_jxl_bytes_from_rgb(rgb: np.ndarray, distance: float, effort: int = 7) -> bytes:
    """
    Encode an RGB uint8 array to JXL bytes via `cjxl`. Uses a lossless temporary PNG for compatibility.
    """
    # Assertion and setup
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    jxl_path = None

    # Writes a temporary PNG from the provided RGB array
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as pngf:
        png_path = pngf.name
        Image.fromarray(rgb, mode="RGB").save(png_path, format="PNG", optimize=False)

    try:
        # Creates a temporary .jxl file
        with tempfile.NamedTemporaryFile(suffix=".jxl", delete=False) as jxlf:
            jxl_path = jxlf.name

        # Constructs the jxl command: https://github.com/libjxl/libjxl?tab=readme-ov-file
        cmd = [
            "cjxl", png_path, jxl_path,
            "--distance", f"{float(distance)}",
            "-e", f"{int(effort)}",
            "--quiet",
        ]

        # Create a subprocess and run the command
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Return JXl byte string s.t. we can do: cr = w*h*3 / len(jxl_bytes)
        with open(jxl_path, "rb") as f:
            return f.read()
    finally:
        # Delete temporary files
        if png_path:
            try: os.remove(png_path)
            except OSError: pass
        if jxl_path and os.path.exists(jxl_path):
            try: os.remove(jxl_path)
            except OSError: pass

def decode_jxl_bytes_to_rgb(jxl_bytes: bytes) -> np.ndarray:
    """Decode JXL bytes to RGB uint8 via `djxl` -> PNG temp (lossless intermediate)."""
    # Create a temporary .jxl file and write in the provided jxl_bytes
    with tempfile.NamedTemporaryFile(suffix=".jxl", delete=False) as jxlf:
        jxl_path = jxlf.name
        jxlf.write(jxl_bytes)
        jxlf.flush()

    # Create a temporary .png file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as pngf:
        png_path = pngf.name

    try:
        # Create a command to decode: https://github.com/libjxl/libjxl?tab=readme-ov-file
        cmd = [
            "djxl",
            jxl_path,
            png_path,
            "--quiet"
        ]

        # Run the command
        subprocess.run(cmd,
                       check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

        # Reads the decoded PNG and return it as a numpy array h*w*3
        rec = Image.open(png_path).convert("RGB")
        return np.asarray(rec, dtype=np.uint8)
    finally:
        # Remove temporary files
        for p in (jxl_path, png_path):
            if p:
                try: os.remove(p)
                except OSError: pass

def _ssim_for_distance_jxl(rgb: np.ndarray, distance: float, effort: int) -> Tuple[float, bytes]:
    """Encode at given JXL distance, decode, compute SSIM vs original. (ssim, jxl_bytes)"""
    jxl_bytes = encode_jxl_bytes_from_rgb(rgb, distance=distance, effort=effort) # encode
    recon = decode_jxl_bytes_to_rgb(jxl_bytes) # decode
    return ssim_rgb(rgb, recon), jxl_bytes # return the structural similarity

def match_ssim_bisection_jxl(
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
    Find JXL distance so SSIM(distance) ~= target_ssim within ±tol using a bracketing+bisection strategy.
    Monotonicity assumption: distance ↑ => quality ↓ => SSIM ↓ (approx holds in practice).
    Returns (best_distance, best_ssim, best_bytes).

    The best distance is the distance value among all tested candidates that minimizes the absolute SSIM error
    |SSIM(distance) − target_ssim|, with a preference to stop early once that error is ≤ tol.
    """
    lo_d = float(dist_lo_init)
    hi_d = float(dist_hi_init)

    lo_s, lo_b = _ssim_for_distance_jxl(rgb, lo_d, effort)
    hi_s, hi_b = _ssim_for_distance_jxl(rgb, hi_d, effort)

    # Expand upper bound if still “too good” (above target)
    while hi_d < dist_hi_max and hi_s > target_ssim + tol:
        hi_d = max(0.5, hi_d * 1.6)
        hi_s, hi_b = _ssim_for_distance_jxl(rgb, hi_d, effort)

    # Track best-of-seen
    best_d, best_s, best_b = (lo_d, lo_s, lo_b)
    if abs(hi_s - target_ssim) < abs(best_s - target_ssim):
        best_d, best_s, best_b = (hi_d, hi_s, hi_b)

    # Early exits
    if abs(lo_s - target_ssim) <= tol:
        return lo_d, lo_s, lo_b
    if abs(hi_s - target_ssim) <= tol:
        return hi_d, hi_s, hi_b

    # Ensure lo side has higher SSIM
    if lo_s < hi_s:
        lo_d, hi_d = hi_d, lo_d
        lo_s, hi_s = hi_s, lo_s
        lo_b, hi_b = hi_b, lo_b

    # Bisection
    for _ in range(max_iters):
        mid_d = 0.5 * (lo_d + hi_d)
        mid_s, mid_b = _ssim_for_distance_jxl(rgb, mid_d, effort)

        if abs(mid_s - target_ssim) < abs(best_s - target_ssim):
            best_d, best_s, best_b = (mid_d, mid_s, mid_b)

        if abs(mid_s - target_ssim) <= tol:
            return mid_d, mid_s, mid_b

        # Choose side (lo_s >= mid_s >= hi_s if monotone)
        if mid_s >= target_ssim:
            lo_d, lo_s, lo_b = (mid_d, mid_s, mid_b)
        else:
            hi_d, hi_s, hi_b = (mid_d, mid_s, mid_b)

        if abs(hi_d - lo_d) < 1e-6:
            break

    return best_d, best_s, best_b
