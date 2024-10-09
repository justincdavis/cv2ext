# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations


def valid(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int] | None = None,
) -> bool:
    """
    Check if a bounding box is valid.

    The conditions for a valid bounding box are that:
    the top-left corner is strictly above the bottom-right corner,
    and the bounding box has strictly greater than zero area.
    Can also check if the bounding box is within the bounds of an image.

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        The bounding box to check.
        Bounding box is in form xyxy.
    shape : tuple[int, int], optional
        The shape of the image. If provided, will check if the bounding box
        is within the bounds of the image.

    Returns
    -------
    bool
        True if the bounding box is valid, False otherwise.

    """
    x1, y1, x2, y2 = bbox
    negative_check = not any(coord < 0 for coord in bbox)
    order_check = x1 < x2 and y1 < y2
    if shape:
        return negative_check and order_check and within(bbox, shape)
    return negative_check and order_check


def within(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    """
    Check if a bounding box is within the bounds of an image.

    The conditions for a bounding box to be within the bounds of an image
    are that the top-left corner is within the bounds of the image,
    and the bottom-right corner is within the bounds of the image.

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        The bounding box to check.
        Bounding box is in form xyxy.
    shape : tuple[int, int]
        The shape of the image.

    Returns
    -------
    bool
        True if the bounding box is within the bounds of the image, False otherwise.

    """
    x1, y1, x2, y2 = bbox
    height, width = shape
    return 0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height
