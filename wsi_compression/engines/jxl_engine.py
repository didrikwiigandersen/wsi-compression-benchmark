"""
JXL engine. Takes in a slide, coordinates for tiles, and the results from the JPEG engine run, to ensure quality match.
"""

# ----------------------- Packages ----------------------- #
from wsi_compression.config import Settings
from wsi_compression.utils.classes.Tile import Tile
from wsi_compression.utils.classes.Result import Result
from wsi_compression.utils.engines.jpeg_helpers import (
    raw_bytes,
    read_tile_rgb,
    ssim_rgb
)

from wsi_compression.utils.engines.jxl_helpers import (
    ensure_cli_tools_or_raise,
    match_ssim_bisection_jxl,
    encode_jxl_bytes_from_rgb,
    decode_jxl_bytes_to_rgb
)

from typing import List, Dict
import openslide
import time

# ----------------------- Main ----------------------- #
def jxl_run_tiles(
        slide_path: str,
        tiles: List[Tile],
        jpeg_results: List[Result]
) -> List[Result]:
    """
    Encode each tile with JXL at a distance tuned to match the JPEG anchor SSIM.

    :param slide_path: The slide path in memory.
    :param tiles: The selected tiles from tile_sampler.py
    :param jpeg_results: the List of Results from jpeg_engine. Used to compare SSIM.
    :return: Returns a List of Result objects.
    """

    # ---------------- Ensure tools ----------------- #
    ensure_cli_tools_or_raise(("cjxl", "djxl"))

    # ---------------- Setup ----------------- #

    # Creates a lookup table: {id: 1, ssim: 0.995}
    anchor: Dict[int, float] = {r.tile_data.id: float(r.ssim) for r in jpeg_results if r.codec.lower() == "jpeg"}
    results: List[Result] = []  # array for storing results
    slide = openslide.OpenSlide(slide_path)  # opening slide

    # Init a settings object and retrieve relevant fields
    s = Settings()
    ssim_tol = float(s.SSIM_TOL)
    max_iters = int(s.MAX_ITERS)
    jxl_effort = max(1, min(9, int(s.JXL_EFFORT))) # Accepts effort between 1..9
    jxl_dist_lo = float(s.JXL_DIST_LO)
    jxl_dist_hi = float(s.JXL_DIST_HI)
    jxl_dist_max = float(s.JXL_DIST_MAX)

    # ---------------- Iteration ----------------- #
    try:
        # Iterate over each tile
        for t in tiles:
            target_ssim = anchor.get(t.id) # retrieve the ssim for that id
            if target_ssim is None:
                print(f"[JXL] warn: no JPEG anchor for tile_id={t.id}; skipping.")
                continue # skip

            # Read raw file
            raw = read_tile_rgb(slide, t)

            # --------- Set distance based on SSIM ---------- #
            best_d, best_s, best_bytes = match_ssim_bisection_jxl(
                raw,
                target_ssim=target_ssim,
                tol=ssim_tol,
                max_iters=max_iters,
                effort=jxl_effort,
                dist_lo_init=jxl_dist_lo,
                dist_hi_init=jxl_dist_hi,
                dist_hi_max=jxl_dist_max
            )

            # --- Time an encode at the chosen distance ---
            t0 = time.perf_counter()
            final_bytes = encode_jxl_bytes_from_rgb(
                raw,
                distance=best_d,
                effort=jxl_effort)
            t1 = time.perf_counter()

            # --- Time decode ---
            t2 = time.perf_counter()
            recon = decode_jxl_bytes_to_rgb(final_bytes)
            t3 = time.perf_counter()

            # Compute the structural similarity
            s_final = ssim_rgb(raw, recon)

            # ------ Compute metrics --------
            rb = raw_bytes(t.w, t.h)       # raw RGB reference
            bo = len(final_bytes)          # encoded size
            cr = rb / max(1, bo)           # compression ratio

            # Append results to the results list
            m = Result(
                id=t.id,
                tile_data=t,
                codec="jxl",
                raw_bytes=rb,
                cr=cr,
                enc_ms=(t1 - t0) * 1000.0,
                dec_ms=(t3 - t2) * 1000.0,
                ssim=s_final
            )
            results.append(m)
            print(f"[{time.time()}][JXL] Appended: {m}")

    finally:
        slide.close()

    return results
