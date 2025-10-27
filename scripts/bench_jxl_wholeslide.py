"""
Compress an entire WSI with JPEG-XL. Reports total bytes, encode/decode times, and SSIM.
"""

# -------------- Packages ----------------
import math, subprocess, tempfile, time, openslide
import numpy as np

from pathlib import Path
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# -------------- Settings ----------------
SLIDE_PATH = "/Users/didrikwiig-andersen/palette-research/projects/pathology-compression/data/sample.ndpi"
RESULTS_DIR  = "/Users/didrikwiig-andersen/palette-research/projects/pathology-compression/results/wholeslide"

LEVEL = 0 # keep at 0
DISTANCE = 0.5 # 0 = lossless (compression goes down, SSIM up), 1 = near-lossless
EFFORT = 7 # how hard the encoder works to find the best compression (high => more time, but smaller file)

TILE_SIZE = 1024 # Adjust: 256, 1024, and 4096
STRIDE = TILE_SIZE

SAVE_EXAMPLE = True
MAX_TILES = None
PROGRESS_EVERY = 10

# -------------- Helper Functions ----------------
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

def iter_tiles_full_coverage(W: int, H: int, tile: int, stride: int):
    """Divides the whole-slide image into tiles."""
    nx = math.ceil(W / stride)
    ny = math.ceil(H / stride)
    for j in range(ny):
        y = min(j * stride, max(0, H - tile))
        for i in range(nx):
            x = min(i * stride, max(0, W - tile))
            yield x, y

def to_float_rgb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

def main():
    print(f"Reading slide: {SLIDE_PATH}")

    # Open the slide and retrieve core metrics
    slide = openslide.OpenSlide(SLIDE_PATH)
    W, H = slide.level_dimensions[LEVEL] # width and height of slide

    print(f"Slide level {LEVEL} size: {W} x {H}")

    # Creating results directory
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # Initializing accumulators
    totals = {
        "tiles": 0,
        "src_png_bytes": 0,  # total bytes of reference PNGs
        "jxl_bytes": 0,  # total bytes of JXL-compressed tiles
        "enc_s": 0.0,  # total encode time (seconds)
        "dec_s": 0.0,  # total decode time (seconds)
        "ssim_sum": 0.0  # sum of per-tile SSIM values
    }

    example_saved = False

    # ------------- Encoding -----------------
    try:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src_png = td / "src.png"     # original source image as PNG
            jxl     = td / "tile.jxl"    # compressed JPEG-XL tile
            dec_png = td / "dec.png"     # the decoded image

            # Loop over all tile coordinates
            iters = 0
            for x, y in iter_tiles_full_coverage(W, H, TILE_SIZE, STRIDE):
                iters += 1

                # @REMOVE ME FOR PRODUCTION
                if MAX_TILES and iters > MAX_TILES:
                    break

                # Read one tile from the slide
                img = slide.read_region((x, y), LEVEL, (TILE_SIZE, TILE_SIZE)).convert("RGB")

                # Save a temporary reference in PNG format
                img.save(src_png, format="PNG", compress_level=0) # lossless (no compression)
                src_bytes = src_png.stat().st_size # size of the tile in bytes
                totals["src_png_bytes"] += src_bytes # update the byte count for png

                # Define helper metrics
                a = to_float_rgb(img) # converts image to a numpy array

                # Encode the tile to JPEG-XL
                # Builds a shell command for the JPEG-XL encoder
                enc_t = run_cmd([
                    "cjxl",
                    str(src_png), # input file
                    str(jxl),     # output file
                    f"--distance={DISTANCE}", # compression quality
                    f"--effort={EFFORT}",     # encoder effort
                    "--quiet"
                ])
                totals["enc_s"] += enc_t # update the time
                totals["jxl_bytes"] += jxl.stat().st_size # update the size of the compressed jxl

                # ---------- Decoding -------------
                dec_t = run_cmd([
                    "djxl",
                    str(jxl), # input
                    str(dec_png), # output
                    "--quiet"
                ])
                totals["dec_s"] += dec_t # update the time

                # -------- Quality comparison -------
                b = to_float_rgb(Image.open(dec_png)) # opening the decoded PNG
                s = ssim(a, b, channel_axis=2, data_range=1.0) # computes SSIM between a and b
                totals["ssim_sum"] += s # adding to the ssim

                # @REMOVE FOR PRODUCTION
                if SAVE_EXAMPLE and not example_saved:
                    (Path(RESULTS_DIR) / "example_tile_src.png").write_bytes(src_png.read_bytes())
                    (Path(RESULTS_DIR) / "example_tile_dec_jxl.png").write_bytes(Path(dec_png).read_bytes())
                    example_saved = True

                totals["tiles"] += 1
                if totals["tiles"] % PROGRESS_EVERY == 0:
                    print(f"Processed {totals['tiles']} tiles...")
    finally:
        slide.close()

    # ----------- Summary ----------
    tiles = totals["tiles"]
    mean_ssim    = totals["ssim_sum"] / tiles
    ratio_vs_png = (totals["src_png_bytes"] / totals["jxl_bytes"]) if totals["jxl_bytes"] else math.inf

    print("\n=== Whole-slide JPEG-XL results ===")
    print(f"Tiles processed          : {tiles}")
    print(f"Total PNG (ref) bytes    : {totals['src_png_bytes']:,}")
    print(f"Total JXL bytes          : {totals['jxl_bytes']:,}   (ratio vs PNG: {ratio_vs_png:.2f}×)")
    print(f"Mean SSIM                : {mean_ssim:.6f}") # inflated due to white area being most present
    print(f"Total encode time (JXL)  : {totals['enc_s']:.2f}s")
    print(f"Total decode time (JXL)  : {totals['dec_s']:.2f}s")

if __name__ == "__main__":
    main()
