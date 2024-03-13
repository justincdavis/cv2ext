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

import cv2
import numpy as np

from cv2ext import _FLAGSOBJ

_log = logging.getLogger(__name__)

try:
    from numba import jit  # type: ignore[import-untyped]
except ImportError:
    jit = None
    if _FLAGSOBJ.USEJIT:
        _log.warning(
            "Numba not installed, but JIT has been enabled. Not using JIT for template matching.",
        )


def _multiplejit(
    matchfunc: Callable[
        [np.ndarray, tuple[int, int] | tuple[int, int, int], int, float],
        list[tuple[int, int, int, int]],
    ],
) -> Callable[
    [np.ndarray, tuple[int, int] | tuple[int, int, int], int, float],
    list[tuple[int, int, int, int]],
]:
    if _FLAGSOBJ.USEJIT and jit is not None:
        matchfunc = jit(matchfunc, nopython=True)
    return matchfunc


@_multiplejit
def _core_match_multiple(
    result: np.ndarray,
    template_shape: tuple[int, int] | tuple[int, int, int],
    method: int,
    threshold: float,
) -> list[tuple[int, int, int, int]]:
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        locations = np.where(result <= threshold)  # type: ignore[operator]
    else:
        locations = np.where(result >= threshold)  # type: ignore[operator]
    matches = []
    for pt in zip(*locations[::-1]):
        bottom_right = (pt[0] + template_shape[1], pt[1] + template_shape[0])
        matches.append((*pt, *bottom_right))
    return matches


def match_single(
    image: np.ndarray,
    template: np.ndarray,
    method: int = cv2.TM_CCOEFF_NORMED,
) -> tuple[int, int, int, int]:
    """
    Find all matches of the template in the image.

    Parameters
    ----------
    image : np.ndarray
        The image to search for the template in.
    template : np.ndarray
        The template to search for in the image.
    method : int
        The method to use for template matching. One of cv2.TM_*. Default is cv2.TM_CCOEFF_NORMED.

    Returns
    -------
    tuple[int, int, int, int]
        A tuple containing the x1, y1, x2, and y2 coordinates of the match.

    """
    result = cv2.matchTemplate(image, template, method)
    _, _, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = min_loc if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_loc
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
    x1, y1 = top_left
    x2, y2 = bottom_right
    return x1, y1, x2, y2


def match_multiple(
    image: np.ndarray,
    template: np.ndarray,
    method: int = cv2.TM_CCOEFF_NORMED,
    threshold: float = 0.8,
) -> list[tuple[int, int, int, int]]:
    """
    Find all matches of the template in the image.

    Parameters
    ----------
    image : np.ndarray
        The image to search for the template in.
    template : np.ndarray
        The template to search for in the image.
    method : int
        The method to use for template matching. One of cv2.TM_*. Default is cv2.TM_CCOEFF_NORMED.
    threshold : float
        The threshold to use for matches. Default is 0.8.

    Returns
    -------
    list[tuple[int, int, int, int]]
        A list of tuples containing the x1, y1, x2, and y2 coordinates of the matches.

    """
    result = cv2.matchTemplate(image, template, method)
    return _core_match_multiple(result, template.shape, method, threshold)  # type: ignore[arg-type]
