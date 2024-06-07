# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging
import operator
from typing import Callable

from cv2ext import _FLAGSOBJ

from ._iou import _iou_kernel

_log = logging.getLogger(__name__)

try:
    from numba import jit  # type: ignore[import-untyped]
except ImportError:
    jit = None
    if _FLAGSOBJ.USEJIT:
        _log.warning(
            "Numba not installed, but JIT has been enabled. Not using JIT for NMS.",
        )


def _nmsjit(
    nmsfunc: Callable[
        [list[tuple[tuple[int, int, int, int], int, float]], float],
        list[tuple[tuple[int, int, int, int], int, float]],
    ],
) -> Callable[
    [list[tuple[tuple[int, int, int, int], int, float]], float],
    list[tuple[tuple[int, int, int, int], int, float]],
]:
    if _FLAGSOBJ.USEJIT and jit is not None:
        _log.info("JIT Compiling: NMS")
        nmsfunc = jit(nmsfunc, nopython=True)
    return nmsfunc


@_nmsjit
def _nms_kernel(
    bboxes: list[tuple[tuple[int, int, int, int], int, float]],
    iou_threshold: float = 0.5,
) -> list[tuple[tuple[int, int, int, int], int, float]]:
    bboxes = sorted(bboxes, key=operator.itemgetter(2), reverse=False)
    final_bboxes = []
    for idx1 in range(len(bboxes)):
        box1 = bboxes[idx1]
        discard = False
        for idx2 in range(idx1 + 1, len(bboxes)):
            box2 = bboxes[idx2]
            if _iou_kernel(box1[0], box2[0]) > iou_threshold and box1[2] < box2[2]:
                discard = True
                break
        if not discard:
            final_bboxes.append(box1)
    return final_bboxes


def nms(
    bboxes: list[tuple[tuple[int, int, int, int], int, float]],
    iou_threshold: float = 0.5,
) -> list[tuple[tuple[int, int, int, int], int, float]]:
    """
    Perform non-maximum suppression on a list of bounding boxes.

    Parameters
    ----------
    bboxes : list[tuple[tuple[int, int, int, int], int, float]]
        A list of bounding boxes, each represented as a tuple of the form
        ((x1, y1, x2, y2), class, confidence
    iou_threshold : float
        The intersection over union threshold for non-maximum suppression.

    Returns
    -------
    list[tuple[tuple[int, int, int, int], int, float]]
        A list of bounding boxes, each represented as a tuple of the form
        ((x1, y1, x2, y2), class, confidence

    """
    return _nms_kernel(bboxes, iou_threshold)
