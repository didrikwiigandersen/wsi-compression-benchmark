"""
Main module for wsi_compression.
"""
from wsi_compression.utils.sampling.tile_sampler import sample_tiles_with_mask
from wsi_compression.utils.visualization.tile_visualizer import visualize_tiles
from engines.jpeg_engine import jpg_run_tiles
from engines.jxl_engine import jxl_run_tiles

from wsi_compression.engines.j2k_engine import j2k_run_tiles
from wsi_compression.utils.analysis.analysis_helpers import (
    plot_cr_scatter_by_codec,
    jxl_superiority_tests
)
from wsi_compression.config import Settings

def main():
    # --------------------- Setup -------------------- #
    s = Settings()

    # -------------------- Sampling tiles ----------------------- #
    tiles = sample_tiles_with_mask(s.SLIDE_PATH, s.MASK_PATH, s.RNG_SEED)
    visualize_tiles(s.SLIDE_PATH, s.MASK_PATH, tiles, out_path=s.VIZ_OUT_PATH)

    # -------------------- Running codec engines ----------------------- #
    jpeg_result = jpg_run_tiles(s.SLIDE_PATH, tiles)
    jxl_result = jxl_run_tiles(s.SLIDE_PATH, tiles, jpeg_result)
    j2k_result = j2k_run_tiles(s.SLIDE_PATH, tiles, jpeg_result)

    # -------------------- Analysis ---------------------------------- #
    plot_cr_scatter_by_codec(
        jpeg_result, jxl_result, j2k_result,
        savepath=s.SCATTER_PUT_PATH,
        show=True
    )

    df_long = plot_cr_scatter_by_codec(jpeg_result, jxl_result, j2k_result, show=False)
    stats_out = jxl_superiority_tests(df_long)
    print(stats_out)


if __name__ == '__main__':
    main()