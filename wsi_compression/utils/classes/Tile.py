"""
Tile class used to store information about the selected tile.
"""

from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    w: int
    h: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h
        }
