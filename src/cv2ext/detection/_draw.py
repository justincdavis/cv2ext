# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from cv2ext.image.color import Color
from cv2ext.image.draw import rectangle, text

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


def draw_detections(
    image: np.ndarray,
    dets: Sequence[tuple[tuple[int, int, int, int], float, int]],
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
    dets : Sequence[tuple[tuple[int, int, int, int], float, int]]
        The detections to draw.
        The detections should be in form:
        (bbox, confidence, classid)
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

    for bbox, conf, classid in dets:
        drawing = rectangle(
            drawing,
            bbox,
            color,
            thickness,
            opacity,
        )

        class_label = class_map[classid] if class_map else str(classid)
        label = f"{class_label}: {conf:.2f}"
        drawing = text(
            drawing,
            label,
            bbox[0:2],
            color,
        )

    return drawing
