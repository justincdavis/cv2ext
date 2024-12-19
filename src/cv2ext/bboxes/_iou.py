# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from itertools import starmap

from cv2ext._jit import register_jit


@register_jit
def _iou_kernel(
    bbox1: tuple[int, int, int, int],
    bbox2: tuple[int, int, int, int],
) -> float:
    x1a, y1a, x2a, y2a = bbox1
    x1b, y1b, x2b, y2b = bbox2

    x1 = max(x1a, x1b)
    y1 = max(y1a, y1b)
    x2 = min(x2a, x2b)
    y2 = min(y2a, y2b)

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    if inter == 0.0:
        return 0.0

    area1 = (x2a - x1a) * (y2a - y1a)
    area2 = (x2b - x1b) * (y2b - y1b)

    union = area1 + area2 - inter

    return inter / union if union != 0.0 else 0.0


@register_jit
def _iou_kernel_list(
    bboxes1: list[tuple[int, int, int, int]],
    bboxes2: list[tuple[int, int, int, int]],
) -> list[float]:
    ious: list[float] = []
    for idx in range(len(bboxes1)):
        ious.append(_iou_kernel(bboxes1[idx], bboxes2[idx]))
    return ious


def iou(
    bbox1: tuple[int, int, int, int],
    bbox2: tuple[int, int, int, int],
) -> float:
    """
    Calculate the intersection over union of two bounding boxes.

    Parameters
    ----------
    bbox1 : tuple[int, int, int, int]
        The first bounding box in the format (x1, y1, x2, y2).
    bbox2 : tuple[int, int, int, int]
        The second bounding box in the format (x1, y1, x2, y2).

    Returns
    -------
    float
        The intersection over union of the two bounding boxes.

    """
    return _iou_kernel(bbox1, bbox2)


def ious(
    bboxes1: list[tuple[int, int, int, int]],
    bboxes2: list[tuple[int, int, int, int]],
) -> list[float]:
    """
    Calculate the intersection over union of two lists of bounding boxes.

    Parameters
    ----------
    bboxes1 : list[tuple[int, int, int, int]]
        The first list of bounding boxes in the format (x1, y1, x2, y2).
    bboxes2 : list[tuple[int, int, int, int]]
        The second list of bounding boxes in the format (x1, y1, x2, y2).

    Returns
    -------
    list[float]
        The intersection over union of the two lists of bounding boxes.

    """
    return _iou_kernel_list(bboxes1, bboxes2)
