# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule containing utilities for working with images.

Functions
---------
color_euclidean_dist
    Compute the euclidean distance between two colors.
dominant_color
    Compute the dominant color in an image.
mean_color
    Compute the mean color in an image.

"""

from __future__ import annotations

from ._color import color_euclidean_dist, dominant_color, mean_color

__all__ = ["color_euclidean_dist", "dominant_color", "mean_color"]
