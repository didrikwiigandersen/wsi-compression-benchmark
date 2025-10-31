"""
Engine for JPEG2000.
"""

# ----------------------- Packages ----------------------- #
from typing import List, Dict

import openslide
import time

from wsi_compression.utils.classes.Tile import Tile
from wsi_compression.utils.classes.Result import Result

from wsi_compression.utils.engine_helpers import (
    _read_tile_rgb,
    _raw_bytes,
    _match_ssim_bisection_rate,
    _encode_j2k_bytes_from_rgb,
    _decode_j2k_bytes_to_rgb,
    _ssim_rgb
)
# ----------------------- Constants ----------------------- #

# ----------------------- Main ----------------------- #

def j2k_run_tiles(
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
    results: List[Result] = []
    target_by_tile: Dict[int, float] = {r.tile_data.id: float(r.ssim) for r in jpeg_results}
    slide = openslide.OpenSlide(slide_path)

    # ---------------- Iteration ----------------- #
    try:
        for t in tiles:
            target = target_by_tile.get(t.id)
            if target is None:
                print(f"[J2K] warn: no JPEG anchor for tile_id={t.id}; skipping.")
                continue

            # Reading the tile
            raw = _read_tile_rgb(slide, t)

            # Search for the params that match the quality level of the jpeg
            best_r, best_s, best_b = _match_ssim_bisection_rate(raw, target_ssim=target)

            # --- time final encode ---
            t0 = time.perf_counter()
            final_bytes = _encode_j2k_bytes_from_rgb(raw, rate=best_r)
            t1 = time.perf_counter()

            # --- time final decode + compute final SSIM ---
            t2 = time.perf_counter()
            recon_final = _decode_j2k_bytes_to_rgb(final_bytes)
            t3 = time.perf_counter()
            s_final = _ssim_rgb(raw, recon_final)

            # --- CR strictly vs RAW (8-bit, 3 channels) ---
            rb = _raw_bytes(t.w, t.h)
            cr = rb / max(1, len(final_bytes))

            results.append(Result(
                id=t.id,
                tile_data=t,
                codec="j2k",
                raw_bytes=rb,
                cr=cr,
                enc_ms=(t1 - t0) * 1000.0,
                dec_ms=(t3 - t2) * 1000.0,
                ssim=float(s_final),
            ))

            # Remove for prod
            print(f"[J2K] tile {t.id} done (rate={best_r:.2f})")
    finally:
        slide.close()

    return results

