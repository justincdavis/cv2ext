# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from cv2ext.image.color import Color
from cv2ext.image.draw import rectangle, text

if TYPE_CHECKING:
    from collections.abc import Sequence


def draw_bboxes(
    image: np.ndarray,
    bboxes: Sequence[tuple[int, int, int, int]],
    confidences: Sequence[float] | None = None,
    classes: Sequence[str | int] | None = None,
    class_map: dict[int, str] | None = None,
    color: Color | tuple[int, int, int] = Color.RED,
    thickness: int = 2,
    opacity: float | None = None,
    *,
    copy: bool | None = None,
) -> np.ndarray:
    """
    Draw bounding boxes on an image.

    Parameters
    ----------
    image : np.ndarray
        The image to draw the bounding boxes on.
    bboxes : Sequence[tuple[int, int, int, int]]
        The Sequence of bounding boxes to draw.
        The bounding boxes should be in form:
        x1, y1, x2, y2 (top-left, bottom-right) format
    confidences : Sequence[float], optional
        The confidence values for each bounding box.
        If provided, the confidence values will be drawn on the image.
    classes : Sequence[str | int], optional
        The class labels for each bounding box.
        If provided, the class labels will be drawn on the image
        with the bounding boxes. If the label is a string, it will
        be used as is, otherwise it will be used as an index into
        the class_map.
    class_map : dict[int, str], optional
        The class map to use for converting class indices to labels.
    color : Color, tuple[int, int, int], optional
        The color to draw the bounding boxes.
        In BGR format and the default is Color.RED.
    thickness : int, optional
        The thickness of the bounding box lines.
        Default is 2.
    opacity : float, optional
        The opacity to draw the bounding boxes with.
        By default None or 100%. Can have a high performance impact.
    copy : bool, optional
        Whether or not to copy the image before drawing.
        Default is False.

    Returns
    -------
    np.ndarray
        The image with the bounding boxes drawn.

    """
    drawing = image
    if copy:
        drawing = image.copy()

    # generate the tags
    tags: list[str] = []
    for idx in range(len(bboxes)):
        tag = ""
        if confidences is not None:
            confidence = confidences[idx]
            tag = f"{confidence:.2f}"
        if classes is not None:
            classid = classes[idx]
            if isinstance(classid, int) and class_map is not None:
                classid = class_map[classid]
            tag = f"{classid}: {tag}"
        tags.append(tag)

    shapes: np.ndarray | None = None
    if opacity:
        shapes = np.zeros_like(drawing, np.uint8)

    # only draw the boxes
    for bbox in bboxes:
        if opacity and shapes is not None:
            rectangle(shapes, bbox, color=color, thickness=thickness)
        else:
            rectangle(drawing, bbox, color=color, thickness=thickness)

    # blend if opacity and shapes
    if opacity and shapes is not None:
        mask: np.ndarray = shapes.astype(bool)
        drawing[mask] = cv2.addWeighted(drawing, 1 - opacity, shapes, opacity, 0)[mask]

    # draw the tags
    for tag, bbox in zip(tags, bboxes):
        if tag:
            text(
                drawing,
                tag,
                bbox[:2],
                color=color,
                thickness=1,
                bottom_left_origin=True,
            )

    return drawing
