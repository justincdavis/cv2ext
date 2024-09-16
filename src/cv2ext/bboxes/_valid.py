# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations


def valid(bbox: tuple[int, int, int, int]) -> bool:
    """
    Check if a bounding box is valid.

    The conditions for a valid bounding box are that:
    the top-left corner is strictly above the bottom-right corner,
    and the bounding box has strictly greater than zero area.

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        The bounding box to check.
        Bounding box is in form xyxy.

    Returns
    -------
    bool
        True if the bounding box is valid, False otherwise.

    """
    x1, y1, x2, y2 = bbox
    if any(coord < 0 for coord in bbox):
        return False
    return x1 < x2 and y1 < y2
