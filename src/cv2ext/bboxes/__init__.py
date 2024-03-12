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
"""
Subpackage containing tools for working with simple bounding boxes.

Functions
---------
iou
    Calculate the intersection over union of two bounding boxes.
mean_ap
    Calculate the mean average precision of a set of bounding boxes.
nms
    Perform non-maximum suppression on a set of bounding boxes.

"""
from __future__ import annotations

from ._iou import iou, ious
from ._mean_ap import mean_ap
from ._nms import nms

__all__ = ["iou", "ious", "mean_ap", "nms"]
