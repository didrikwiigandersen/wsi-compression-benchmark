"""
Configuration file for wsi_compression.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    # Experiment controls
    NUM_TILES: int = 1000
    TILE_SIZE: int = 256
    LEVEL: int = 0
    RNG_SEED: int = 42
    MIN_TISSUE_FRAC: float = 1.0
    MAX_ATTEMPTS: int = 1_000_000
    MAX_IOU: float = 1.0

    # Quality matching
    JPEG_QUALITY: int = 80
    SSIM_TOL: float = 1e-3
    MAX_ITERS: int = 12

    # JXL tuning
    JXL_EFFORT: int = 7
    JXL_DIST_LO: float = 0.0
    JXL_DIST_HI: float = 3.0
    JXL_DIST_MAX: float = 6.0

    # J2k tuning
    J2K_RATE_LO: float = 1.0
    J2K_RATE_HI: float = 600.0
    J2K_RATE_MAX: float = 1200.0
