"""

"""
from wsi_compression.utils.sampling.tile_sampler import sample_tiles_with_mask
from wsi_compression.utils.visualization.tile_visualizer import visualize_tiles_on_overview
from engines.jpeg_engine import jpg_run_tiles
from engines.jxl_engine import jxl_run_tiles

from wsi_compression.engines.j2k_engine import j2k_run_tiles
from wsi_compression.utils.analysis.analysis_helpers import (
    plot_cr_scatter_by_codec
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
    j2k_result = j2k_run_tiles(slide_path, tiles, jpeg_result)

    # -------------------- Analysis ---------------------------------- #
    plot_cr_scatter_by_codec(
        jpeg_result, jxl_result, j2k_result,
        savepath="/Users/didrikwiig-andersen/palette-research/projects/pathology-compression/results/wsi_compression/analysis",
        show=True
    )


if __name__ == '__main__':
    main()