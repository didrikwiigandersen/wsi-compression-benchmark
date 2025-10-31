"""

"""
from utils.tile_sampler import sample_tiles_with_mask
from utils.tile_visualizer import visualize_tiles_on_overview
from engines.jpeg_engine import jpg_run_tiles
from engines.jxl_engine import jxl_run_tiles

import pandas as pd
import matplotlib.pyplot as plt

from wsi_compression.engines.j2k_engine import j2k_run_tiles
from wsi_compression.utils.classes.Result import Result
from wsi_compression.utils.analysis_helpers import (
    results_to_df,
    merge_on_tile3,
    plot_scatter_pair,
    plot_ecdf_overlay3,
    plot_gain_hist_pair,
    plot_box3,
    print_summary_pairs
)


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
    #j2k_result = j2k_run_tiles(slide_path, tiles, jpeg_result)

    # -------------------- Analysis ---------------------------------- #
    #df_jpg = results_to_df(jpeg_result)
    #df_jxl = results_to_df(jxl_result)
    #df_j2k = results_to_df(j2k_result)


if __name__ == '__main__':
    main()