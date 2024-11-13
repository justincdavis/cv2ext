# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule containing utilities for working with images.

Submodules
----------
color
    Color utilities for images.
draw
    Drawing utilities for images.

Functions
---------
color_euclidean_dist
    Compute the euclidean distance between two colors.
divide
    Divide an image into multiple sub-images.
dominant_color
    Compute the dominant color in an image.
mean_color
    Compute the mean color in an image.
image_tiler
    Tile an image across another image.
create_tiled_image
    Tile an image across another image, or create a new image of the tile.
letterbox
    Resize an image using the letterbox method.
resize_linear
    Resize an image using linear scaling.
patch
    Divide an image into equal size patches.
rescale
    Rescale an image from [0:255] to another range.

"""

from __future__ import annotations

from . import color, draw
from ._augment import letterbox, resize_linear
from ._color import color_euclidean_dist, dominant_color, mean_color
from ._divide import divide, patch
from ._scale import rescale
from ._tiling import create_tiled_image, image_tiler

__all__ = [
    "color",
    "color_euclidean_dist",
    "create_tiled_image",
    "divide",
    "dominant_color",
    "draw",
    "image_tiler",
    "letterbox",
    "mean_color",
    "patch",
    "rescale",
    "resize_linear",
]
