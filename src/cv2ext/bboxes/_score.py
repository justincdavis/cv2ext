# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from cv2ext import _FLAGSOBJ

_log = logging.getLogger(__name__)

try:
    from numba import jit  # type: ignore[import-untyped]
except ImportError:
    jit = None

if TYPE_CHECKING:
    from collections.abc import Callable


def _score_bbox_kernel_jit(
    score_func: Callable[[tuple[int, int, int, int], tuple[int, int, int, int]], float],
) -> Callable[[tuple[int, int, int, int], tuple[int, int, int, int]], float]:
    if _FLAGSOBJ.USEJIT and jit is not None:
        _log.info("JIT Compiling: score_bbox")
        score_func = jit(score_func, nopython=True)
    return score_func


def _score_bboxes_kernel_jit(
    score_func: Callable[
        [tuple[int, int, int, int], list[tuple[int, int, int, int]]],
        list[float],
    ],
) -> Callable[
    [tuple[int, int, int, int], list[tuple[int, int, int, int]]],
    list[float],
]:
    if _FLAGSOBJ.USEJIT and jit is not None:
        _log.info("JIT Compiling: score_bboxs")
        score_func = jit(score_func, nopython=True)
    return score_func


@_score_bbox_kernel_jit
def _score_bbox_kernel(
    target_bbox: tuple[int, int, int, int],
    pred_bbox: tuple[int, int, int, int],
) -> float:
    tx1, ty1, tx2, ty2 = target_bbox
    th = ty2 - ty1
    tw = tx2 - tx1
    ts = th * tw
    tcx = tx1 + tw / 2
    tcy = ty1 + th / 2
    nx1, ny1, nx2, ny2 = pred_bbox
    nh = ny2 - ny1
    nw = nx2 - nx1
    ns = nh * nw
    ncx = nx1 + nw / 2
    ncy = ny1 + nh / 2
    dist = math.sqrt((tcx - ncx) ** 2 + (tcy - ncy) ** 2) / max(th + nh, tw + nw)
    area_diff = abs(ts - ns) / max(ts, ns)
    return 1.0 - min(1.0, dist + area_diff)


@_score_bboxes_kernel_jit
def _score_bboxes_kernel(
    target_bbox: tuple[int, int, int, int],
    pred_bboxs: list[tuple[int, int, int, int]],
) -> list[float]:
    return [_score_bbox_kernel(target_bbox, pred_bbox) for pred_bbox in pred_bboxs]


def score_bbox(
    target_bbox: tuple[int, int, int, int],
    pred_bbox: tuple[int, int, int, int],
) -> float:
    """
    Compute a simple score metric for two bounding boxes.

    The score is computed as an aggregate between
    the euclidean distance and the area difference.

    Parameters
    ----------
    target_bbox : tuple[int, int, int, int]
        The target bounding box.
        Bounding box is format (xmin, ymin, xmax, ymax)
    pred_bbox : tuple[int, int, int, int]
        The predicted bounding box.
        Bounding box is format (xmin, ymin, xmax, ymax)

    Returns
    -------
    float
        The score between the two bounding boxes.

    """
    return _score_bbox_kernel(target_bbox, pred_bbox)


def score_bboxes(
    target_bbox: tuple[int, int, int, int],
    pred_bboxs: list[tuple[int, int, int, int]],
) -> list[float]:
    """
    Compute a simple score metric for a target bounding box and a list of predicted bounding boxes.

    The score is computed as an aggregate between
    the euclidean distance and the area difference.

    Parameters
    ----------
    target_bbox : tuple[int, int, int, int]
        The target bounding box.
        Bounding box is format (xmin, ymin, xmax, ymax)
    pred_bboxs : list[tuple[int, int, int, int]]
        The list of predicted bounding boxes.
        Bounding box is format (xmin, ymin, xmax, ymax)

    Returns
    -------
    list[float]
        The scores between the target bounding box and the list of predicted bounding boxes.

    """
    return _score_bboxes_kernel(target_bbox, pred_bboxs)
