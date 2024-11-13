# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging
import operator
from typing import TYPE_CHECKING, Callable

from typing_extensions import ParamSpec, TypeVar

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

if TYPE_CHECKING:
    from typing_extensions import TypeAlias


Detections: TypeAlias = "list[tuple[tuple[int, int, int, int], float, int]]"
P = ParamSpec("P")
R = TypeVar("R")


def _nmsjit(
    nmsfunc: Callable[P, R],
) -> Callable[P, R]:
    if _FLAGSOBJ.USEJIT and jit is not None:
        _log.info("JIT Compiling: NMS")
        nmsfunc = jit(nmsfunc, nopython=True)
    return nmsfunc


# ignore the arg type for the agnostic variable
@_nmsjit  # type: ignore[arg-type]
def _nms_kernel(
    bboxes: Detections,
    iou_threshold: float = 0.5,
    *,
    agnostic: bool | None = None,
) -> Detections:
    # bboxes = sorted(bboxes, key=operator.itemgetter(1), reverse=False)
    # final_bboxes = []
    # for idx1 in range(len(bboxes)):
    #     box1 = bboxes[idx1]
    #     discard = False
    #     for idx2 in range(idx1 + 1, len(bboxes)):
    #         box2 = bboxes[idx2]
    #         if _iou_kernel(box1[0], box2[0]) > iou_threshold and box1[1] < box2[1]:
    #             discard = True
    #             break
    #     if not discard:
    #         final_bboxes.append(box1)
    # return final_bboxes
    # Pre-sort by confidence in descending order (highest confidence first)

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
