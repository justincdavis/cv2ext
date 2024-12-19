# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from cv2ext._jit import jit


@jit
def _constrain_kernel(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width, height = image_size
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0
    if x1 > width:
        x1 = width
    if y1 > height:
        y1 = height
    if x2 > width:
        x2 = width
    if y2 > height:
        y2 = height
    return x1, y1, x2, y2


def constrain(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """
    Constrain a bounding box to the dimensions of an image.

    Coordinates are not assumed to be positive, and the bounding box is not assumed
    to be within the image at all. As such, it is possible to return (0, 0, 0, 0) or
    (width, height, width, height) if the bounding box is completely outside the image.

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        The bounding box to constrain.
        Format is: (x1, y1, x2, y2)
    image_size : tuple[int, int]
        The dimensions of the image.
        Format is: (width, height)

    Returns
    -------
    tuple[int, int, int, int]
        The constrained bounding box.
        Format is: (x1, y1, x2, y2)

    """
    return _constrain_kernel(bbox, image_size)
