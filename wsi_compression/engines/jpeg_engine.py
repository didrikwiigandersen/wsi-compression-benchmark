"""
JPEG engine. Encodes tiles passed in, decode back. Measures time, compression ratio. Return this as an object.
"""
import time

import numpy as np

# ----------------------- Packages ----------------------- #
from wsi_compression.utils.classes.Result import Result
from wsi_compression.utils.classes.Tile import Tile
from wsi_compression.utils.helpers import (
    _raw_bytes,
    _read_tile_rgb,
    _encode_jpeg_to_bytes,
    _ssim_rgb
)
from typing import List
import openslide
import io
from PIL import Image

# ----------------------- Main ----------------------- #
def run_tiles(
    slide_path: str,
    tiles: List[Tile],
    jpeg_quality: int = 80 # will never be lossless, inherent to JPEG format
) -> List[Result]:
    """
    Takes in-memory tiles, returns in-memory Results list.
    :param tiles:
    :param jpeg_quality:
    :return:
    """

    # ---------------- Setup ----------------- #
    results: List[Result] = [] # array for storing results
    slide = openslide.OpenSlide(slide_path) # opening slide
    iterator = 0
    try:
        # Iterate over each tile
        for t in tiles:
            iterator += 1
            # Read raw file
            raw = _read_tile_rgb(slide, t)

            # Encode to JPEG
            t0 = time.perf_counter()
            jpeg_bytes = _encode_jpeg_to_bytes(raw, quality=jpeg_quality)
            t1 = time.perf_counter()

            # Decode from JPEG
            t2 = time.perf_counter()
            recon = np.array(Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"), dtype=np.uint8)
            t3 = time.perf_counter()

            # Compute metrics
            rb = _raw_bytes(t.w, t.h)
            bo = len(jpeg_bytes)
            cr = rb / max(1, bo)
            s = _ssim_rgb(raw, recon)

            # Make a result object
            m = Result(
                id=iterator,
                tile_data=t,
                codec="jpeg",
                raw_bytes=rb,
                cr=cr,
                enc_ms=(t1 - t0) * 1000.0,
                dec_ms=(t3 - t2) * 1000.0,
                ssim=s
            )
            results.append(m)
            print(m)
    finally:
        slide.close()
        print("[JPEG] Complete")
        return results
