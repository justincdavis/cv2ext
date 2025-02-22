# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from cv2ext._jit import register_jit

if TYPE_CHECKING:
    from collections.abc import Sequence


@register_jit()
def _bounding_kernel(
    bboxes: Sequence[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    min_x1, min_y1, max_x2, max_y2 = float("inf"), float("inf"), 0, 0
    for x1, y1, x2, y2 in bboxes:
        min_x1 = min(min_x1, x1)
        min_y1 = min(min_y1, y1)
        max_x2 = max(max_x2, x2)
        max_y2 = max(max_y2, y2)
    return int(min_x1), int(min_y1), max_x2, max_y2


def bounding(bboxes: Sequence[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """
    Get a bounding box which encloses all the given bounding boxes.

    Parameters
    ----------
    bboxes : Sequence[tuple[int, int, int, int]]
        A sequence of bounding boxes.
        Bounding boxes are in form xyxy.

    Returns
    -------
    tuple[int, int, int, int]
        The bounding box which encloses all the given bounding boxes.

    Raises
    ------
    ValueError
        If the length of bboxes is zero.

    """
    if len(bboxes) == 0:
        err_msg = "Cannot create bounding from 0 boxes."
        raise ValueError(err_msg)

    return _bounding_kernel(bboxes)
