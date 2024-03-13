# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

import logging
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
            "Numba not installed, but JIT has been enabled. Not using JIT for meanAP.",
        )


def _meanapjit(
    meanapfunc: Callable[
        [
            list[tuple[tuple[int, int, int, int], int, float]],
            list[tuple[tuple[int, int, int, int], int]],
            int,
            float,
        ],
        float,
    ],
) -> Callable[
    [
        list[tuple[tuple[int, int, int, int], int, float]],
        list[tuple[tuple[int, int, int, int], int]],
        int,
        float,
    ],
    float,
]:
    if _FLAGSOBJ.USEJIT and jit is not None:
        meanapfunc = jit(meanapfunc, nopython=True)
    return meanapfunc


@_meanapjit
def _meanap_kernel(
    bboxes: list[tuple[tuple[int, int, int, int], int, float]],
    gt_bboxes: list[tuple[tuple[int, int, int, int], int]],
    num_classes: int,
    iou_threshold: float,
) -> float:
    def calculate_ap(tp: int, fp: int, npos: int) -> float:
        rec = [0.0]
        prec = [0.0]
        for i in range(tp + fp):
            if i > 0:
                rec.append(tp / npos)
                prec.append(tp / (tp + fp))
            if tp + fp > 0:
                tp -= 1
            else:
                break
        return sum([p * (r - rn) for p, r, rn in zip(prec[:-1], rec[1:], rec[:-1])])

    bboxes = sorted(bboxes, key=lambda x: x[2], reverse=True)
    tp: list[int] = [0] * num_classes
    fp: list[int] = [0] * num_classes
    ap: list[float] = [0.0] * num_classes
    for bbox, class_id, _ in bboxes:
        best_match_iou = 0.0
        best_match_gt_idx = None
        for gt_idx, (gt_bbox, gt_class_id) in enumerate(gt_bboxes):
            if class_id == gt_class_id:
                iou = _iou_kernel(bbox, gt_bbox)
                if iou > best_match_iou:
                    best_match_iou = iou
                    best_match_gt_idx = gt_idx

        if best_match_iou >= iou_threshold and best_match_gt_idx is not None:
            tp[class_id] += 1
            gt_bboxes.pop(best_match_gt_idx)
        else:
            fp[class_id] += 1

    for class_id in range(num_classes):
        npos = len(
            [gt_bbox for gt_bbox, gt_class_id in gt_bboxes if gt_class_id == class_id],
        )
        ap[class_id] = calculate_ap(tp[class_id], fp[class_id], npos)

    return sum(ap) / num_classes


def mean_ap(
    bboxes: list[tuple[tuple[int, int, int, int], int, float]],
    gt_bboxes: list[tuple[tuple[int, int, int, int], int]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> float:
    """
    Calculate the mean average precision for a set of bounding boxes.

    Parameters
    ----------
    bboxes : list[tuple[tuple[int, int, int, int], int, float]]
        A list of bounding boxes, each represented as a tuple of the form
        ((x1, y1, x2, y2), class, confidence
    gt_bboxes : list[tuple[tuple[int, int, int, int], int]]
        A list of ground truth bounding boxes, each represented as a tuple of the form
        ((x1, y1, x2, y2), class)
    num_classes : int
        The number of classes in the dataset.
    iou_threshold : float, optional
        The threshold for considering a detection a true positive, by default 0.5

    Returns
    -------
    float
        The mean average precision of the bounding boxes.

    """
    return _meanap_kernel(bboxes, gt_bboxes, num_classes, iou_threshold)
