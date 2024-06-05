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
"""Example showcasing how to use the mean_ap function."""

from __future__ import annotations

import cv2ext

if __name__ == "__main__":
    bboxes = [
        ((0, 0, 10, 10), 0, 0.75),
        ((1, 1, 9, 9), 0, 0.75),
        ((2, 2, 8, 8), 0, 0.75),
    ]
    gt = [
        ((0, 0, 10, 10), 0),
        ((1, 1, 9, 9), 0),
        ((2, 2, 8, 8), 0),
    ]
    mean_ap = cv2ext.bboxes.mean_ap([bboxes], [gt], 1)
    print(mean_ap)
