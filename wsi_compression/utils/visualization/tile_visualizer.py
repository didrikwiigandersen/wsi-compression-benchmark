"""
Visualizes the chosen tiles over the mask.
"""

# ---------------- Packages --------------------
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw
import openslide
from wsi_compression.utils.sampling.sampling_helpers import load_mask_boolean
from wsi_compression.utils.classes.Tile import Tile

# ---------------- Main --------------------
def visualize_tiles(
    slide_path: str,
    mask_png_path: str,
    tiles: List[Tile],
    out_path: str,
    max_width: int = 2048,
    draw_every: int = 1,
    show_mask: bool = False,
    tile_outline: Tuple[int, int, int] = (0, 255, 0),  # green
    tile_width: int = 2,
    mask_tint_rgba: Tuple[int, int, int, int] = (255, 0, 0, 80)
) -> str:
    """

    :param slide_path:
    :param mask_png_path:
    :param tiles:
    :param out_path:
    :param max_width:
    :param draw_every:
    :param show_mask:
    :param tile_outline:
    :param tile_width:
    :param mask_tint_rgba:
    :return:
    """
    slide = openslide.OpenSlide(slide_path)

    try:
        slide_w, slide_h = slide.dimensions
        scale = max_width / float(slide_w)
        thumb_size = (max_width, int(round(slide_h * scale)))
        thumb = slide.get_thumbnail(thumb_size).convert("RGBA")

        if show_mask and mask_png_path:
            mask_bool = load_mask_boolean(mask_png_path)
            mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
            mask_resized = mask_img.resize(thumb_size)
            tint = Image.new("RGBA", thumb_size, mask_tint_rgba)
            thumb = Image.composite(tint, thumb, mask_resized).convert("RGBA")

        draw = ImageDraw.Draw(thumb)
        for i, t in enumerate(tiles):
            if (i % draw_every) != 0:
                continue
            x, y, w, h = t["x"] if isinstance(t, dict) else t.x, \
                         t["y"] if isinstance(t, dict) else t.y, \
                         t["w"] if isinstance(t, dict) else t.w, \
                         t["h"] if isinstance(t, dict) else t.h
            x0 = int(round(x * scale))
            y0 = int(round(y * scale))
            x1 = int(round((x + w) * scale))
            y1 = int(round((y + h) * scale))
            for k in range(tile_width):
                draw.rectangle([x0-k, y0-k, x1+k, y1+k], outline=tile_outline, width=1)
        thumb.save(out_path)
        return out_path
    finally:
        slide.close()
