"""
Engine for JPEG2000.
"""

# ----------------------- Packages ----------------------- #
from typing import List, Dict
import openslide
import time

from wsi_compression.config import Settings
from wsi_compression.utils.classes.Tile import Tile
from wsi_compression.utils.classes.Result import Result
from wsi_compression.utils.engines.jpeg_helpers import (
    read_tile_rgb,
    raw_bytes,
    ssim_rgb
)

from wsi_compression.utils.engines.j2k_helpers import (
    match_ssim_bisection_rate,
    encode_j2k_bytes_from_rgb,
    decode_j2k_bytes_to_rgb
)

# ----------------------- Main ----------------------- #

def j2k_run_tiles(
        slide_path: str,
        tiles: List[Tile],
        jpeg_results: List[Result]
) -> List[Result]:
    """
    J2K run. Encodes and decodes based on set quality level. Measures CR, times.
    :param slide_path: The path of the slide in memory.
    :param tiles: The tiles identified by the tile_sampler.
    :param jpeg_results: The results of the jpg run. Used to retrieve the SSIM.
    :return: A List of Result objects.
    """

    # ---------------- Setup ----------------- #

    # Creates a lookup table: {id: 1, ssim: 0.995}
    anchor: Dict[int, float] = {r.tile_data.id: float(r.ssim) for r in jpeg_results if r.codec.lower() == "jpeg"}
    results: List[Result] = []  # array for storing results
    slide = openslide.OpenSlide(slide_path)  # opening slide

    # Init a settings object and retrieve relevant fields
    s = Settings()
    ssim_tol = float(s.SSIM_TOL)
    max_iters = int(s.MAX_ITERS)
    j2k_rate_lo = float(s.J2K_RATE_LO)
    j2k_rate_hi = float(s.J2K_RATE_HI)
    j2k_rate_max =  float(s.J2K_RATE_MAX)

    # ---------------- Iteration ----------------- #
    try:
        for t in tiles:
            target = anchor.get(t.id)
            if target is None:
                print(f"[J2K] warn: no JPEG anchor for tile_id={t.id}; skipping.")
                continue

            # Reading the tile
            raw = read_tile_rgb(slide, t)

            # Search for the params that match the quality level of the jpeg
            best_r, best_s, best_b = match_ssim_bisection_rate(
                raw,
                target_ssim=target,
                tol=ssim_tol,
                max_iters=max_iters,
                rate_lo_init=j2k_rate_lo,
                rate_hi_init=j2k_rate_hi,
                rate_hi_max=j2k_rate_max,
            )

            # --- Time an encode at the chosen rate ---
            t0 = time.perf_counter()
            jp2_bytes = encode_j2k_bytes_from_rgb(
                raw,
                rate=best_r)
            t1 = time.perf_counter()

            # --- Time decode ---
            t2 = time.perf_counter()
            recon = decode_j2k_bytes_to_rgb(jp2_bytes)
            t3 = time.perf_counter()

            # Compute the structural similarity
            s_final = float(ssim_rgb(raw, recon))

            # ------- Compute metrics ------------
            rb = raw_bytes(t.w, t.h)
            bo = len(jp2_bytes)
            cr = rb / max(1, bo)

            # ----- Create a Result object ------- #
            m = Result(
                    id=t.id,
                    tile_data=t,
                    codec="j2k",
                    raw_bytes=rb,
                    cr=cr,
                    enc_ms=(t1 - t0) * 1000.0,
                    dec_ms=(t3 - t2) * 1000.0,
                    ssim=s_final,
                )
            results.append(m)
            print(f"[{time.time()}][J2K] Appended: {m}")
    finally:
        slide.close()

    return results
