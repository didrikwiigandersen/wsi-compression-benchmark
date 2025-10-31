"""
JXL engine. Takes in a slide, coordinates for tiles, and the results from the JPEG engine run, to ensure quality match.
"""
import time

import openslide

# ----------------------- Packages ----------------------- #
from wsi_compression.utils.classes.Tile import Tile
from wsi_compression.utils.classes.Result import Result
from wsi_compression.utils.engine_helpers import (
    _read_tile_rgb,
    _match_ssim_bisection,
    _encode_jxl_bytes_from_rgb,
    _decode_jxl_bytes_to_rgb,
    _raw_bytes,
    _ssim_rgb
)
from typing import List

# ----------------------- Constants ----------------------- #
EFFORT = 7

# ----------------------- Main ----------------------- #
def jxl_run_tiles(
        slide_path: str,
        tiles: List[Tile],
        jpeg_results: List[Result]
) -> List[Result]:
    """

    :param slide_path:
    :param tiles:
    :param jpeg_results:
    :return:
    """

    # ---------------- Setup ----------------- #
    results: List[Result] = [] # array for storing results
    anchor = {r.tile_data.id: float(r.ssim) for r in jpeg_results}
    slide = openslide.OpenSlide(slide_path) # opening slide

    # ---------------- Iteration ----------------- #
    try:
        # Iterate over each tile
        for t in tiles:
            target_ssim = anchor.get(t.id)
            if target_ssim is None:
                print(f"[JXL] warn: no JPEG anchor for tile_id={t.id}; skipping.")
                continue

            # Read raw file
            raw = _read_tile_rgb(slide, t)

            # --------- Set distance based on SSIM ---------- #
            best_d, best_s, best_bytes = _match_ssim_bisection(raw, target_ssim=target_ssim, effort=EFFORT)

            # --------- re-encode once at the best distance to time encode ---------- #
            t0 = time.perf_counter()
            final_bytes = _encode_jxl_bytes_from_rgb(raw, distance=best_d, effort=EFFORT)
            t1= time.perf_counter()

            # -------------- time decode separately ------------- #
            t2 = time.perf_counter()
            recon_final = _decode_jxl_bytes_to_rgb(final_bytes)
            t3 = time.perf_counter()
            s_final = _ssim_rgb(raw, recon_final)

            # -------------- compute metrics ------------- #
            rb = _raw_bytes(t.w, t.h)          # 8-bit * 3 channels
            bo = len(final_bytes)
            cr = rb / max(1, bo)

            results.append(Result(
                id=t.id,
                tile_data=t,
                codec="jxl",
                raw_bytes=rb,
                cr=cr,
                enc_ms=(t1 - t0) * 1000.0,
                dec_ms=(t3 - t2) * 1000.0,
                ssim=s_final
            ))
            print(f"{t.id}:  added to jxl_result")
    finally:
        slide.close()

    return results
