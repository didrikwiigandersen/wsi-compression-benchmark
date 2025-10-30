"""
Holds a result from encoding and decoding a tile.
"""

from dataclasses import dataclass
from typing import Dict
from wsi_compression.utils.classes.Tile import Tile

@dataclass
class Result:
    id: int
    tile_data: Tile
    codec: str
    raw_bytes: int # raw bytes of the tile
    cr: float     # compression ratio
    enc_ms: float # encode speed
    dec_ms: float # decode speed
    ssim: float   # quality

    def as_dict(self) -> Dict[str, float]:
        d = {
            "result_id": self.id,
            "tile_id": self.tile_data.id,
            "x": self.tile_data.x,
            "y": self.tile_data.y,
            "w": self.tile_data.w,
            "h": self.tile_data.h,
            "codec": self.codec,
            "raw_bytes": self.raw_bytes,
            "cr": self.cr,
            "enc_ms": self.enc_ms,
            "dec_ms": self.dec_ms,
            "ssim": self.ssim
        }
        return d
