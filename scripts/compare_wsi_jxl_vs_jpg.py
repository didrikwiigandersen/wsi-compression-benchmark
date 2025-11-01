# Created by CHATGPT and only used for quick testing a hypothesis. Not to be included in the project.

#!/usr/bin/env python3
"""
Compress an entire WSI (tiled at 1024x1024) with JPEG-XL and JPEG.
Report total bytes, encode/decode times, and SSIM (mean and tissue-weighted).

Prereqs (macOS):
  brew install openslide libjxl
Python:
  pip install openslide-python pillow numpy scikit-image
"""

import math, subprocess, time, tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import openslide

# ====== SETTINGS ======
SLIDE_PATH = "/Users/didrikwiig-andersen/palette-research/projects/pathology-compression/data/sample.ndpi"
LEVEL = 0
TILE_SIZE = 1024
STRIDE = 1024        # equal to tile size => non-overlapping full coverage
DISTANCE_JXL = 0.8   # JXL quality (lower = higher quality)
EFFORT_JXL = 7
JPEG_QUALITY = 85    # JPEG quality, 4:4:4 for fairness
SAVE_EXAMPLES = False
MAX_TILES = 200     # e.g., set to 200 for a quick run; None = full slide
WHITE_THRESH = 250   # threshold to detect "white" pixels
# ======================


def run(cmd):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr.decode()}")
    return dt

def to_float_rgb(im):
    return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0

def white_fraction(im, thresh=WHITE_THRESH):
    a = np.asarray(im.convert("RGB"))
    white = (a[...,0]>=thresh) & (a[...,1]>=thresh) & (a[...,2]>=thresh)
    return float(white.mean())

def iter_tiles(slide, level, tile, stride):
    W, H = slide.level_dimensions[level]
    xs = range(0, max(1, W - tile + 1), stride)
    ys = range(0, max(1, H - tile + 1), stride)
    for y in ys:
        for x in xs:
            # guard last partial edges: clamp to keep full tile
            xx = min(x, max(0, W - tile))
            yy = min(y, max(0, H - tile))
            yield xx, yy, (W, H)

def encode_jxl(png_in: Path, jxl_out: Path, distance: float, effort: int):
    return run(["cjxl", str(png_in), str(jxl_out),
                f"--distance={distance}", f"--effort={effort}", "--quiet"])

def decode_jxl(jxl_in: Path, png_out: Path):
    return run(["djxl", str(jxl_in), str(png_out), "--quiet"])

def encode_jpeg(img: Image.Image, jpg_out: Path, quality: int):
    t0 = time.perf_counter()
    img.save(jpg_out, "JPEG", quality=int(quality), subsampling=0, optimize=True)  # 4:4:4
    return time.perf_counter() - t0

def decode_jpeg(jpg_in: Path, png_out: Path):
    t0 = time.perf_counter()
    Image.open(jpg_in).convert("RGB").save(png_out, "PNG")
    return time.perf_counter() - t0

def main():
    slide = openslide.OpenSlide(SLIDE_PATH)
    W, H = slide.level_dimensions[LEVEL]
    print(f"Slide level {LEVEL} size: {W}x{H}")

    totals = {
        "tiles": 0,
        "src_png_bytes": 0,             # for ratio vs lossless PNG reference
        "jxl_bytes": 0,
        "jpg_bytes": 0,
        "enc_jxl_s": 0.0,
        "dec_jxl_s": 0.0,
        "enc_jpg_s": 0.0,
        "dec_jpg_s": 0.0,
        "ssim_jxl_sum": 0.0,
        "ssim_jpg_sum": 0.0,
        "w_ssim_jxl_sum": 0.0,          # tissue-weighted SSIM sums
        "w_ssim_jpg_sum": 0.0,
        "weight_sum": 0.0               # total tissue weight
    }

    example_saved = False
    n = 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for x, y, _ in iter_tiles(slide, LEVEL, TILE_SIZE, STRIDE):
            img = slide.read_region((x, y), LEVEL, (TILE_SIZE, TILE_SIZE)).convert("RGB")

            # Optional: cap total tiles for a quick run
            n += 1
            if MAX_TILES and n > MAX_TILES:
                break

            # Build a lossless PNG reference (no compression) to size against
            src_png = td / "src.png"
            img.save(src_png, format="PNG", compress_level=0)
            src_bytes = src_png.stat().st_size
            totals["src_png_bytes"] += src_bytes

            # Tissue weight: 1 - white_fraction (so white=0 weight, tissue=~1)
            wf = white_fraction(img)
            weight = max(0.0, 1.0 - wf)
            totals["weight_sum"] += weight

            a = to_float_rgb(img)

            # ---- JPEG-XL ----
            jxl = td / "t.jxl"; jxl_dec = td / "t_jxl.png"
            t_enc_jxl = encode_jxl(src_png, jxl, DISTANCE_JXL, EFFORT_JXL)
            t_dec_jxl = decode_jxl(jxl, jxl_dec)
            b_jxl = to_float_rgb(Image.open(jxl_dec))
            s_jxl = ssim(a, b_jxl, channel_axis=2, data_range=1.0)
            totals["jxl_bytes"] += jxl.stat().st_size
            totals["enc_jxl_s"] += t_enc_jxl
            totals["dec_jxl_s"] += t_dec_jxl
            totals["ssim_jxl_sum"] += s_jxl
            totals["w_ssim_jxl_sum"] += s_jxl * weight

            # ---- JPEG (4:4:4) ----
            jpg = td / "t.jpg"; jpg_dec = td / "t_jpg.png"
            t_enc_jpg = encode_jpeg(img, jpg, JPEG_QUALITY)
            t_dec_jpg = decode_jpeg(jpg, jpg_dec)
            b_jpg = to_float_rgb(Image.open(jpg_dec))
            s_jpg = ssim(a, b_jpg, channel_axis=2, data_range=1.0)
            totals["jpg_bytes"] += jpg.stat().st_size
            totals["enc_jpg_s"] += t_enc_jpg
            totals["dec_jpg_s"] += t_dec_jpg
            totals["ssim_jpg_sum"] += s_jpg
            totals["w_ssim_jpg_sum"] += s_jpg * weight

            if SAVE_EXAMPLES and not example_saved:
                img.save("example_tile.png")
                Image.open(jxl_dec).save("example_tile_jxl.png")
                Image.open(jpg_dec).save("example_tile_jpg.png")
                example_saved = True

            totals["tiles"] += 1
            if totals["tiles"] % 50 == 0:
                print(f"Processed {totals['tiles']} tiles...")

    slide.close()

    # Aggregates
    tiles = totals["tiles"]
    mean_ssim_jxl = totals["ssim_jxl_sum"] / tiles
    mean_ssim_jpg = totals["ssim_jpg_sum"] / tiles
    w_mean_ssim_jxl = totals["w_ssim_jxl_sum"] / max(1e-9, totals["weight_sum"])
    w_mean_ssim_jpg = totals["w_ssim_jpg_sum"] / max(1e-9, totals["weight_sum"])

    ratio_jxl = totals["src_png_bytes"] / totals["jxl_bytes"] if totals["jxl_bytes"] else math.inf
    ratio_jpg = totals["src_png_bytes"] / totals["jpg_bytes"] if totals["jpg_bytes"] else math.inf

    print("\n=== Whole-slide results (tile 1024, level 0) ===")
    print(f"Tiles processed         : {tiles}")
    print(f"Total PNG (ref) bytes   : {totals['src_png_bytes']:,}")
    print(f"Total JXL bytes         : {totals['jxl_bytes']:,}   (ratio vs PNG: {ratio_jxl:.2f}×)")
    print(f"Total JPEG bytes        : {totals['jpg_bytes']:,}   (ratio vs PNG: {ratio_jpg:.2f}×)")
    print(f"Mean SSIM (JXL / JPEG)  : {mean_ssim_jxl:.6f} / {mean_ssim_jpg:.6f}")
    print(f"Tissue-weighted SSIM    : {w_mean_ssim_jxl:.6f} / {w_mean_ssim_jpg:.6f}")
    print(f"Encode time (JXL/JPEG)  : {totals['enc_jxl_s']:.2f}s / {totals['enc_jpg_s']:.2f}s")
    print(f"Decode time (JXL/JPEG)  : {totals['dec_jxl_s']:.2f}s / {totals['dec_jpg_s']:.2f}s")




if __name__ == '__main__':
    main()