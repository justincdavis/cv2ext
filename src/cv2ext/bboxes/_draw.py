# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np


def draw_bboxes(
    image: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes on an image.

    Parameters
    ----------
    image : np.ndarray
        The image to draw the bounding boxes on.
    bboxes : list[tuple[int, int, int, int]]
        The list of bounding boxes to draw.
        The bounding boxes should be in form:
        x1, y1, x2, y2 (top-left, bottom-right) format
    color : tuple[int, int, int], optional
        The color to draw the bounding boxes.
        In BGR format and the default is red.
    thickness : int, optional
        The thickness of the bounding box lines.
        Default is 2.

    Returns
    -------
    np.ndarray
        The image with the bounding boxes drawn.

    """
    drawing = image.copy()

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(drawing, (x1, y1), (x2, y2), color, thickness)

    return drawing
