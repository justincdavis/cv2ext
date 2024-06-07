# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging
from itertools import starmap
from typing import Callable

from cv2ext import _FLAGSOBJ

_log = logging.getLogger(__name__)

try:
    from numba import jit  # type: ignore[import-untyped]
except ImportError:
    jit = None
    if _FLAGSOBJ.USEJIT:
        _log.warning(
            "Numba not installed, but JIT has been enabled. Not using JIT for IOU.",
        )


def _iou_kernel_jit(
    iouk_func: Callable[[tuple[int, int, int, int], tuple[int, int, int, int]], float],
) -> Callable[[tuple[int, int, int, int], tuple[int, int, int, int]], float]:
    if _FLAGSOBJ.USEJIT and jit is not None:
        _log.info("JIT Compiling: iou")
        iouk_func = jit(iouk_func, nopython=True)
    return iouk_func


def _iou_list_kernel_jit(
    iouk_func: Callable[
        [list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]],
        list[float],
    ],
) -> Callable[
    [list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]],
    list[float],
]:
    if _FLAGSOBJ.USEJIT and jit is not None:
        _log.info("JIT Compiling: iou_list")
        iouk_func = jit(iouk_func, nopython=True)
    return iouk_func


@_iou_kernel_jit
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


@_iou_list_kernel_jit
def _iou_kernel_list(
    bboxes1: list[tuple[int, int, int, int]],
    bboxes2: list[tuple[int, int, int, int]],
) -> list[float]:
    return list(starmap(_iou_kernel, zip(bboxes1, bboxes2)))


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
