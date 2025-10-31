"""

"""
from utils.tile_sampler import sample_tiles_with_mask
from utils.tile_visualizer import visualize_tiles_on_overview
from engines.jpeg_engine import jpg_run_tiles
from engines.jxl_engine import jxl_run_tiles

import pandas as pd
import matplotlib.pyplot as plt

from wsi_compression.utils.classes.Result import Result
from wsi_compression.utils.analysis_helpers import (
    results_to_df,
    merge_on_tile,
    plot_scatter_cr,
    plot_ecdf_overlay,
    plot_gain_hist,
    plot_box,
    print_summary
)
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
    jpeg_result = jpg_run_tiles(slide_path, tiles)
    jxl_result = jxl_run_tiles(slide_path, tiles, jpeg_result)

    # -------------------- Analysis ---------------------------------- #
    df_jpg = results_to_df(jpeg_result)
    df_jxl = results_to_df(jxl_result)

    # after you build df_jpg / df_jxl from results_to_df(...)
    df_pairs = merge_on_tile(df_jpg, df_jxl)

    # 1) Paired scatter with y=x (log axes)
    plot_scatter_cr(df_pairs)

    # 2) ECDF overlay (global distribution)
    plot_ecdf_overlay(df_jpg, df_jxl)

    # 3) Histogram of per-tile savings (% smaller bytes than JPEG)
    plot_gain_hist(df_pairs)

    # 4) Simple boxplot comparison
    plot_box(df_jpg, df_jxl)

    # 5) Text summary
    print_summary(df_pairs)

    #j2k_result = run_tiles(tiles, jpeg_result)




if __name__ == '__main__':
    main()