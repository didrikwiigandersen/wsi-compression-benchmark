"""

"""
from utils.tile_sampler import sample_tiles_with_mask
from utils.tile_visualizer import visualize_tiles_on_overview


def main():

    slide_path = "/Users/didrikwiig-andersen/palette-research/projects/pathology-compression/data/lab_sample.ndpi"
    mask_path = "/Users/didrikwiig-andersen/palette-research/projects/pathology-compression/data/mask.png"

    tiles = sample_tiles_with_mask(slide_path, mask_path)
    visualize_tiles_on_overview(slide_path, mask_path, [t.as_dict() for t in tiles],
                                out_path="overview_with_tiles.png",
                                draw_every=1,  # draw every 5th to reduce clutter
                                show_mask=True)






if __name__ == '__main__':
    main()