"""
Randomly samples 1000 tiles from a provided WSI in .ndpi format. Upon selecting a tile, it checks the corresponding
mask (.png) whether there is tissue there. If there is tissue, the tile's coordinates are selected and stored. The tile
is marked as selected. Otherwise, another tile is selected.

The sample is passed as an argument to the codec engines for compression. The process of tile sampling is separate
to ensure that the codec-engines work on the same data.
"""