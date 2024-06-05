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

from typing import Sequence


def xyxy_to_xywh(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """
    Convert a bounding box from (xmin, ymin, xmax, ymax) to (x, y, w, h).

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        The bounding box to convert.
        Bounding box is format (xmin, ymin, xmax, ymax),
        where (xmin, ymin) is the top-left corner and (xmax, ymax) is the bottom-right corner.

    Returns
    -------
    tuple[int, int, int, int]
        The converted bounding box.
        Bounding box is format (x, y, w, h),
        where (x, y) is the top-left corner and (w, h) is the width and height.

    """
    x1, y1, x2, y2 = bbox
    return x1, y1, x2 - x1, y2 - y1


def xywh_to_xyxy(
    bbox: tuple[int, int, int, int] | Sequence[int],
) -> tuple[int, int, int, int]:
    """
    Convert a bounding box from (x, y, w, h) to (xmin, ymin, xmax, ymax).

    Parameters
    ----------
    bbox : tuple[int, int, int, int] | Sequence[int]
        The bounding box to convert.
        Bounding box is format (x, y, w, h),
        where (x, y) is the top-left corner and (w, h) is the width and height.

    Returns
    -------
    tuple[int, int, int, int]
        The converted bounding box.
        Bounding box is format (xmin, ymin, xmax, ymax),
        where (xmin, ymin) is the top-left corner and (xmax, ymax) is the bottom-right corner.

    """
    x, y, w, h = bbox
    return x, y, x + w, y + h
