# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule containing utilities for working with images.

Submodules
----------
draw
    Drawing utilities for images.

Functions
---------
color_euclidean_dist
    Compute the euclidean distance between two colors.
dominant_color
    Compute the dominant color in an image.
mean_color
    Compute the mean color in an image.
image_tiler
    Tile an image across another image.
create_tiled_image
    Tile an image across another image, or create a new image of the tile.

"""

from __future__ import annotations

from ._color import color_euclidean_dist, dominant_color, mean_color
from ._tiling import create_tiled_image, image_tiler
from . import draw

__all__ = [
    "draw",
    "color_euclidean_dist",
    "create_tiled_image",
    "dominant_color",
    "image_tiler",
    "mean_color",
]
