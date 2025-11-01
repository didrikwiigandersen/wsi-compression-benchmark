import math
import time
import numpy as np
from pathlib import Path

import openslide
import tempfile
import subprocess

from PIL import Image
from skimage.metrics import structural_similarity as ssim

# ------------ Settings --------------
SLIDE_PATH = "/Users/didrikwiig-andersen/palette-research/projects/pathology-compression/data/sample.ndpi"
TILE_SIZE = 1024 # Adjust: 256, 1024, and 4096
LEVEL = 0 # keep at 0
DISTANCE = 0 # 0 = lossless (compression goes down, SSIM up), 1 = near-lossless
EFFORT = 7 # how hard the encoder works to find the best compression (high => more time, but smaller file)
SAVE_IMAGES = True

def read_tile(slide_path, level=0, x=None, y=None, tile_size=1024):
    """Open the WSI and return a tile as a PIL.Image (RGB), its (x,y), and level dims."""
    slide = openslide.OpenSlide(slide_path) # opens the whole-slide image
    w, h = slide.level_dimensions[level] # gets the dimensions of the image

    # If x, y not given, automatically pick the center of the image
    if x is None or y is None:
        x = max(0, (w - tile_size) // 2)
        y = max(0, (h - tile_size) // 2)
    # return RGBA image containing the contents of the specified region
    region = slide.read_region((x, y), level, (tile_size, tile_size)).convert("RGB")
    slide.close() # close reference
    return region, (x, y), (w, h)

def run_cmd(cmd: list[str]) -> float:
    """Run a shell command; return elapsed seconds or raise with stderr."""
    t0 = time.perf_counter() # starts timer
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) # runs the command in a separate process
    dt = time.perf_counter() - t0 # contains the elapsed time

    # Error handling
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDERR:\n{p.stderr.decode(errors='ignore')}"
        )
    return dt

def main():
    print(f"Reading slide: {SLIDE_PATH}")

    # Reading tile
    img, (x, y), (W, H) = read_tile(SLIDE_PATH, LEVEL, None, None, TILE_SIZE)
    print(f"Tile @ level {LEVEL}: ({x},{y}) size {TILE_SIZE}x{TILE_SIZE} from slide {W}x{H}")

    # ------------- Encoding -----------------

    # Creating a temporary folder
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # Define file paths
        src_png = td / "tile.png" # original tile saved as PNG
        jxl = td / "tile.jxl" # the compressed JPEG-XL file
        dec_png = td / "tile_dec.png" # the decoded image (to compare quality)

        # Save the tile as a lossless PNG (no compression)
        img.save(src_png, format="PNG", compress_level=0)
        raw_bytes = src_png.stat().st_size # how large the file is in bytes

        # Building a shell command to compress the image
        enc_cmd = [
            "cjxl", str(src_png), str(jxl),
            f"--distance={DISTANCE}",
            f"--effort={EFFORT}",
            "--quiet"
        ]

        # Record the size and time to encode
        enc_time = run_cmd(enc_cmd)
        jxl_bytes = jxl.stat().st_size

        # ---------- Decoding -------------

        # Building a shell command to decode the image
        dec_cmd = [
            "djxl",
            str(jxl),
            str(dec_png),
            "--quiet"
        ]
        dec_time = run_cmd(dec_cmd)

        # ---------- Comparison -------------

        # Open the compressed PNG file djxl made
        # img = original uncompressed tile
        # dec = the same tile after compress + decompress
        dec = Image.open(dec_png).convert("RGB")

        # Convert the visualization into arrays for comparison
        a = np.asarray(img, dtype=np.float32) / 255.0
        b = np.asarray(dec, dtype=np.float32) / 255.0

        # Calls structural similarity index measure
        # 1: identical
        # 0.99+: visually indistinguishable
        # <0.9: noticeable degradation
        s = ssim(a, b, channel_axis=2, data_range=1.0)

        # Compression ratio
        # How many times smaller the compressed file is compared to the original
        ratio = raw_bytes / jxl_bytes if jxl_bytes > 0 else math.inf

        print("\n=== JPEG-XL Result ===")
        print(f"Distance (quality): {DISTANCE}")
        print(f"Effort:             {EFFORT}")
        print(f"PNG bytes:          {raw_bytes:,}")
        print(f"JXL bytes:          {jxl_bytes:,}")
        print(f"Compression ratio:  {ratio:.2f}×")
        print(f"SSIM:               {s:.6f}")
        print(f"Encode time:        {enc_time * 1000:.1f} ms")
        print(f"Decode time:        {dec_time * 1000:.1f} ms")

        # -------------- Saving files -------------
        if SAVE_IMAGES:
            img.save("/Users/didrikwiig-andersen/palette-research/projects/pathology-compression/results/tile_original.png")
            dec.save("/Users/didrikwiig-andersen/palette-research/projects/pathology-compression/results/tile_decoded.png")
            print("\nSaved: tile_original.png, tile_decoded.png")

if __name__ == '__main__':
    main()