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
        nmsfunc = jit(nmsfunc, nopython=True)
    return nmsfunc


@_nmsjit
def _nms_kernel(
    bboxes: list[tuple[tuple[int, int, int, int], int, float]],
    iou_threshold: float = 0.5,
) -> list[tuple[tuple[int, int, int, int], int, float]]:
    bboxes = sorted(bboxes, key=lambda x: x[2], reverse=True)
    final_bboxes = []
    for box1 in bboxes:
        discard = False
        for box2 in bboxes:
            if box1 == box2:
                continue
            if _iou_kernel(box1[0], box2[0]) > iou_threshold:
                discard = True
                break
        if not discard:
            final_bboxes.append(box1)
    return final_bboxes


def nms(
    bboxes: list[tuple[tuple[int, int, int, int], int, float]],
    iou_threshold: float = 0.5,
) -> list[tuple[tuple[int, int, int, int], int, float]]:
    return _nms_kernel(bboxes, iou_threshold)
