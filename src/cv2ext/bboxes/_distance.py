# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import math


def euclidean(
    bbox1: tuple[int, int, int, int],
    bbox2: tuple[int, int, int, int],
) -> float:
    """
    Compute the euclidean distance between two bboxes.

    The computation is performed between the centers of the bounding boxes.

    Parameters
    ----------
    bbox1 : tuple[int, int, int, int]
        The first bounding box.
        Bounding box is in form xyxy.
    bbox2 : tuple[int, int, int, int]
        The second bounding box.
        Bounding box is in form xyxy.

    Returns
    -------
    float
        The euclidean distance between the two bounding boxes.

    """
    cx1 = (bbox1[0] + bbox1[2]) / 2
    cy1 = (bbox1[1] + bbox1[3]) / 2
    cx2 = (bbox2[0] + bbox2[2]) / 2
    cy2 = (bbox2[1] + bbox2[3]) / 2
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def manhattan(
    bbox1: tuple[int, int, int, int],
    bbox2: tuple[int, int, int, int],
) -> float:
    """
    Compute the manhattan distance between two bboxes.

    The computation is performed between the centers of the bounding boxes.

    Parameters
    ----------
    bbox1 : tuple[int, int, int, int]
        The first bounding box.
        Bounding box is in form xyxy.
    bbox2 : tuple[int, int, int, int]
        The second bounding box.
        Bounding box is in form xyxy.

    Returns
    -------
    float
        The manhattan distance between the two bounding boxes.

    """
    cx1 = (bbox1[0] + bbox1[2]) / 2
    cy1 = (bbox1[1] + bbox1[3]) / 2
    cx2 = (bbox2[0] + bbox2[2]) / 2
    cy2 = (bbox2[1] + bbox2[3]) / 2
    return abs(cx1 - cx2) + abs(cy1 - cy2)
