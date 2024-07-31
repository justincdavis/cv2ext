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
    return math.sqrt((bbox1[0] - bbox2[0]) ** 2 + (bbox1[1] - bbox2[1]) ** 2)


def manhattan(
    bbox1: tuple[int, int, int, int],
    bbox2: tuple[int, int, int, int],
) -> float:
    """
    Compute the manhattan distance between two bboxes.

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
    return abs(bbox1[0] - bbox2[0]) + abs(bbox1[1] - bbox2[1])
