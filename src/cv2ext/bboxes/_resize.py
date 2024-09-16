# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations


def resize(
    bbox: tuple[int, int, int, int],
    s1: tuple[int, int],
    s2: tuple[int, int],
) -> tuple[int, int, int, int]:
    """
    Resizes a bounding box based on one image size to another.

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        The bounding box to resize.
        Bounding box is in form xyxy.
    s1 : tuple[int, int]
        The size of the first image.
        In form (width, height).
    s2 : tuple[int, int]
        The size of the second image.
        In form (width, height).

    Returns
    -------
    tuple[int, int, int, int]
        The resized bounding box.
        Bounding box is in form xyxy.

    """
    x1, y1, x2, y2 = bbox
    w1, h1 = s1
    w2, h2 = s2
    width_ratio = w2 / w1
    height_ratio = h2 / h1
    nx1 = int(x1 * width_ratio)
    ny1 = int(y1 * height_ratio)
    nx2 = int(x2 * width_ratio)
    ny2 = int(y2 * height_ratio)
    return nx1, ny1, nx2, ny2
