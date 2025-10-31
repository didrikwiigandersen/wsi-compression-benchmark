"""
Helpers used in JPEG engine.
"""

# ------------------- Packages --------------------- #
import io
import numpy as np
import openslide
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from wsi_compression.utils.classes.Tile import Tile

# ------------------- Codec-agnostic helpers --------------------- #
def _raw_bytes(w: int, h: int) -> int:
    """3 bytes/pixel for 8-bit RGB."""
    return int(w) * int(h) * 3

def _read_tile_rgb(slide: openslide.OpenSlide, t: Tile) -> np.ndarray:
    """OpenSlide returns RGBA; convert to RGB, uint8"""
    return np.array(slide.read_region((t.x, t.y), 0, (t.w, t.h)).convert("RGB"), dtype=np.uint8)

def _ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    """Compare structural similarity"""
    return float(ssim(a, b, channel_axis=2, data_range=255))

def _encode_jpeg_to_bytes(arr: np.ndarray, quality: int) -> bytes:
    """Encode RBG uint8 to JPEG bytes"""
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

def _decode_jpeg_bytes_to_rgb(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes → RGB uint8."""
    return np.array(Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"), dtype=np.uint8)

