"""
A more efficient version of bench_jxl_wholeslide.py, using parallelization and removes comparison measures.
Focused on measuring time to compress end encode. Not used for comparison with other formats.
"""

# -------------- Packages ----------------
from pathlib import Path

# -------------- Settings ----------------
SLIDE_PATH = "/Users/didrikwiig-andersen/palette-research/projects/pathology-compression/data/sample.ndpi"
RESULTS_DIR  = "/Users/didrikwiig-andersen/palette-research/projects/pathology-compression/results/wholeslide/parallelized"

LEVEL = 0 # keep at 0
DISTANCE = 0.5 # 0 = lossless (compression goes down, SSIM up), 1 = near-lossless
EFFORT = 7 # how hard the encoder works to find the best compression (high => more time, but smaller file)

TILE_SIZE = 1024 # Adjust: 256, 1024, and 4096
STRIDE = TILE_SIZE

DO_SSIM      = False       # turn on only in benchmarking mode
WORKERS      = os.cpu_count() // 2 or 4  # start conservative; tune later

def main():
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    main()