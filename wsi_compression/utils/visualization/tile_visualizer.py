"""
Used to visualize the chosen tiles over the mask.
"""

# ---------------- Packages --------------------
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw
import openslide
from wsi_compression.utils.sampling.sampling_helpers import _load_mask_boolean

# ---------------- Main --------------------
def visualize_tiles_on_overview(
    slide_path: str,
    mask_png_path: str,
    tiles: List[dict] | List,
    out_path: str = "/Users/didrikwiig-andersen/palette-research/projects/pathology-compression/results/wsi_compression/images/overview_with_tiles.png",
    max_width: int = 2048,
    draw_every: int = 1,
    show_mask: bool = False,
    tile_outline: Tuple[int, int, int] = (0, 255, 0),  # green
    tile_width: int = 2,
    mask_tint_rgba: Tuple[int, int, int, int] = (255, 0, 0, 80)
) -> str:
    slide = openslide.OpenSlide(slide_path)
    try:
        slide_w, slide_h = slide.dimensions
        # Make an overview thumbnail
        scale = max_width / float(slide_w)
        thumb_size = (max_width, int(round(slide_h * scale)))
        thumb = slide.get_thumbnail(thumb_size).convert("RGBA")

        # Overlay mask as tint
        if show_mask and mask_png_path:
            mask_bool = _load_mask_boolean(mask_png_path)  # shape (H, W), True=tissue
            mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
            # Resize mask to thumbnail size for alignment
            mask_resized = mask_img.resize(thumb_size, resample=Image.NEAREST)
            # Create RGBA tint where mask is tissue
            tint = Image.new("RGBA", thumb_size, mask_tint_rgba)
            thumb = Image.composite(tint, thumb, mask_resized).convert("RGBA")

        # Draw tile rectangles
        draw = ImageDraw.Draw(thumb)
        for i, t in enumerate(tiles):
            if (i % draw_every) != 0:
                continue
            # support dict or dataclass
            x, y, w, h = t["x"] if isinstance(t, dict) else t.x, \
                         t["y"] if isinstance(t, dict) else t.y, \
                         t["w"] if isinstance(t, dict) else t.w, \
                         t["h"] if isinstance(t, dict) else t.h
            x0 = int(round(x * scale))
            y0 = int(round(y * scale))
            x1 = int(round((x + w) * scale))
            y1 = int(round((y + h) * scale))
            for k in range(tile_width): # thickness
                draw.rectangle([x0-k, y0-k, x1+k, y1+k], outline=tile_outline, width=1)

        thumb.save(out_path)
        print(f"[visualizer] wrote {out_path}")
        return out_path
    finally:
        slide.close()
