# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import operator

from cv2ext._jit import register_jit

from ._iou import _iou_kernel


@register_jit
def _nms_kernel(
    bboxes: list[tuple[tuple[int, int, int, int], float, int]],
    iou_threshold: float = 0.5,
    *,
    agnostic: bool | None = None,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    bboxes = sorted(bboxes, key=operator.itemgetter(1), reverse=True)
    keep = [True] * len(bboxes)

    for i in range(len(bboxes)):
        if not keep[i]:
            continue

        box1 = bboxes[i]
        for j in range(i + 1, len(bboxes)):
            if not keep[j]:
                continue

            box2 = bboxes[j]
            if not agnostic and box1[2] != box2[2]:
                continue

            if _iou_kernel(box1[0], box2[0]) > iou_threshold:
                keep[j] = False

    return [box for i, box in enumerate(bboxes) if keep[i]]


def nms(
    bboxes: list[tuple[tuple[int, int, int, int], float, int]],
    iou_threshold: float = 0.5,
    *,
    agnostic: bool | None = None,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    """
    Perform non-maximum suppression on a list of bounding boxes.

    Parameters
    ----------
    bboxes : list[tuple[tuple[int, int, int, int], float, int]]
        A list of bounding boxes, each represented as a tuple of the form
        ((x1, y1, x2, y2), confidence, class
    iou_threshold : float
        The intersection over union threshold for non-maximum suppression.
    agnostic : bool, optional
        If set to True, then bounding boxes of different classes can
        be compared. By default None, will only compare same classes.

    Returns
    -------
    list[tuple[tuple[int, int, int, int], float, int]]
        A list of bounding boxes, each represented as a tuple of the form
        ((x1, y1, x2, y2), confidence, class

    """
    return _nms_kernel(bboxes, iou_threshold, agnostic=agnostic)
