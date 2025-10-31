"""
JPEG engine. Encodes tiles passed in, decode back. Measures time, compression ratio. Returns this as a Result object.
"""
# ----------------------- Packages ----------------------- #
from typing import List
import time
import openslide

from wsi_compression.config import Settings
from wsi_compression.utils.classes.Result import Result
from wsi_compression.utils.classes.Tile import Tile
from wsi_compression.utils.engines.jpeg_helpers import (
    _raw_bytes,
    _read_tile_rgb,
    _encode_jpeg_to_bytes,
    _decode_jpeg_bytes_to_rgb,
    _ssim_rgb
)

# ----------------------- Main ----------------------- #
def jpg_run_tiles(
    slide_path: str,
    tiles: List[Tile],
) -> List[Result]:
    """
    Takes in-memory tiles, returns in-memory Results list.
    :param slide_path: The path to the .ndpi slide.
    :param tiles: The coordinates of the selected tiles from tile_sampler.py
    :return: A list of Result objects.
    """

    # ---------------- Setup ----------------- #
    s = Settings()
    q = max(1, min(100, s.JPEG_QUALITY)) # Pillow accepts 1..100

    results: List[Result] = [] # array for storing results
    slide = openslide.OpenSlide(slide_path) # opening slide

    # ---------------- Iteration ----------------- #
    try:
        # Iterate over each tile
        for t in tiles:
            # Read raw file
            raw = _read_tile_rgb(slide, t)

            # Encode to JPEG
            t0 = time.perf_counter()
            jpeg_bytes = _encode_jpeg_to_bytes(raw, quality=q)
            t1 = time.perf_counter()

            # Decode from JPEG
            t2 = time.perf_counter()
            recon = _decode_jpeg_bytes_to_rgb(jpeg_bytes)
            t3 = time.perf_counter()

            # Compute metrics
            rb = _raw_bytes(t.w, t.h) # raw = 3 * W * H
            bo = len(jpeg_bytes)      # compressed bytes
            cr = rb / max(1, bo)      # compressed vs raw bytes
            s = _ssim_rgb(raw, recon) # structural similarity

            # Make a result object
            m = Result(
                id=t.id,
                tile_data=t,
                codec="jpeg",
                raw_bytes=rb,
                cr=cr,
                enc_ms=(t1 - t0) * 1000.0,
                dec_ms=(t3 - t2) * 1000.0,
                ssim=s
            )
            results.append(m)
            print(f"[{time.time()}][JPEG] Appended: {m}")
    finally:
        slide.close()

    return results
