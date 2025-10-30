"""

"""
from utils.tile_sampler import sample_tiles_with_mask
from utils.tile_visualizer import visualize_tiles_on_overview
from engines.jpeg_engine import run_tiles

from typing import List
import pandas as pd
import matplotlib.pyplot as plt

from wsi_compression.utils.classes.Result import Result

def results_to_df(results: List[Result]) -> pd.DataFrame:
    rows = []
    for r in results:
        t = r.tile_data
        rows.append({
            "tile_id": t.id,            # use tile's id directly
            "codec": r.codec,
            "raw_bytes": r.raw_bytes,
            "cr": r.cr,
            "enc_ms": r.enc_ms,
            "dec_ms": r.dec_ms,
            "ssim": r.ssim,
        })
    return pd.DataFrame(rows)



def plot_cr_hist(df: pd.DataFrame, bins: int = 40, title: str = "JPEG Q=80 — Compression Ratio"):
    plt.figure()
    df["cr"].plot(kind="hist", bins=bins)
    plt.xlabel("Compression Ratio (raw_bytes / compressed_bytes)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def main():
    # ---------------------- Defining paths ---------------------- #
    slide_path = "/Users/didrikwiig-andersen/palette-research/projects/pathology-compression/data/lab_sample.ndpi"
    mask_path = "/Users/didrikwiig-andersen/palette-research/projects/pathology-compression/data/mask.png"

    # -------------------- Sampling tiles ----------------------- #
    tiles = sample_tiles_with_mask(slide_path, mask_path)
    visualize_tiles_on_overview(slide_path, mask_path, [t.as_dict() for t in tiles])

    # -------------------- Running codec engines ----------------------- #
    jpeg_result = run_tiles(slide_path, tiles)

    # -------------------- Quicklook ---------------------------------- #
    df = results_to_df(jpeg_result)
    plot_cr_hist(df)




    #j2k_result = run_tiles(tiles, jpeg_result)
    #jxl_result = run_tiles(tiles, jpeg_result)




if __name__ == '__main__':
    main()