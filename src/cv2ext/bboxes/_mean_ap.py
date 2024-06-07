# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging
import operator
from typing import Callable

import numpy as np

from cv2ext import _FLAGSOBJ

from ._iou import _iou_kernel

_log = logging.getLogger(__name__)

try:
    from numba import jit  # type: ignore[import-untyped]
except ImportError:
    jit = None
    if _FLAGSOBJ.USEJIT:
        _log.warning(
            "Numba not installed, but JIT has been enabled. Not using JIT for meanAP.",
        )


def _meanapjit(
    meanapfunc: Callable[
        [
            list[list[tuple[tuple[int, int, int, int], int, float]]],
            list[list[tuple[tuple[int, int, int, int], int]]],
            int,
            float,
        ],
        float,
    ],
) -> Callable[
    [
        list[list[tuple[tuple[int, int, int, int], int, float]]],
        list[list[tuple[tuple[int, int, int, int], int]]],
        int,
        float,
    ],
    float,
]:
    if _FLAGSOBJ.USEJIT and jit is not None:
        _log.info("JIT Compiling: meanAP")
        meanapfunc = jit(meanapfunc, nopython=True)
    return meanapfunc


@_meanapjit
def _meanap_kernel(
    bboxes: list[list[tuple[tuple[int, int, int, int], int, float]]],
    gt_bboxes: list[list[tuple[tuple[int, int, int, int], int]]],
    num_classes: int,
    iou_threshold: float,
) -> float:
    precision: list[list[float]] = [[] for _ in range(num_classes)]
    recall: list[list[float]] = [[] for _ in range(num_classes)]

    for image_bboxes, image_gt_bboxes in zip(bboxes, gt_bboxes):
        s_image_bboxes = sorted(image_bboxes, key=operator.itemgetter(2), reverse=True)

        true_postives = np.zeros(num_classes)
        false_postives = np.zeros(num_classes)

        for bbox, class_id, _ in s_image_bboxes:
            gt_match = False
            for gt_bbox, gt_class_id in image_gt_bboxes:
                if (
                    class_id == gt_class_id
                    and _iou_kernel(bbox, gt_bbox) >= iou_threshold
                ):
                    true_postives[class_id] += 1
                    gt_match = True
                    break
            if not gt_match:
                false_postives[class_id] += 1

        for c in range(num_classes):
            npos = len(
                [
                    gt_bbox
                    for gt_bbox, gt_class_id in image_gt_bboxes
                    if gt_class_id == c
                ],
            )
            if npos == 0:
                continue
            if true_postives[c] + false_postives[c] > 0:
                precision[c].append(
                    true_postives[c] / (true_postives[c] + false_postives[c]),
                )
                recall[c].append(true_postives[c] / npos)

    ap = {
        c: np.sum(
            [
                (recall[c][i] - recall[c][i - 1]) * precision[c][i]
                for i in range(1, len(precision[c]))
            ],
        )
        for c in range(num_classes)
    }

    return float(np.mean(list(ap.values())))


def mean_ap(
    bboxes: list[list[tuple[tuple[int, int, int, int], int, float]]],
    gt_bboxes: list[list[tuple[tuple[int, int, int, int], int]]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> float:
    """
    Calculate the mean average precision for a set of bounding boxes.

    bboxes and gt_bboxes are lists of lists representing the bounding boxes for each image.
    Each bounding box is represented as a tuple of the form ((x1, y1, x2, y2), class, confidence).

    Parameters
    ----------
    bboxes : list[list[tuple[tuple[int, int, int, int], int, float]]]
        A list of lists of bounding boxes, each represented as a tuple of the form
        ((x1, y1, x2, y2), class, confidence
    gt_bboxes : list[list[tuple[tuple[int, int, int, int], int]]]
        A list of lists of ground truth bounding boxes, each represented as a tuple of the form
        ((x1, y1, x2, y2), class)
    num_classes : int
        The number of classes in the dataset.
    iou_threshold : float, optional
        The threshold for considering a detection a true positive, by default 0.5

    Returns
    -------
    float
        The mean average precision of the bounding boxes.

    Raises
    ------
    ValueError
        If the length of bboxes and gt_bboxes are not equal.
    ValueError
        If the length is zero.

    """
    if len(bboxes) != len(gt_bboxes):
        err_msg = f"Length of bboxes ({len(bboxes)}) and gt_bboxes ({len(gt_bboxes)}) must be equal."
        raise ValueError(err_msg)
    if len(bboxes) == 0:
        err_msg = "Length of bboxes and gt_bboxes must be greater than zero."
        raise ValueError(err_msg)
    return _meanap_kernel(bboxes, gt_bboxes, num_classes, iou_threshold)
