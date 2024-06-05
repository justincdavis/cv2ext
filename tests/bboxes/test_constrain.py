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

from cv2ext.bboxes import constrain


def test_constrain_zeros():
    assert constrain((0, 0, 0, 0), (640, 480)) == (0, 0, 0, 0)


def test_constrain_all_negative():
    bbox = (-10, -10, -5, -5)
    assert constrain(bbox, (640, 480)) == (0, 0, 0, 0)


def test_constrain_all_exceed():
    bbox = (700, 600, 800, 650)
    assert constrain(bbox, (640, 480)) == (640, 480, 640, 480)


def test_constrain_all_within():
    bbox = (10, 10, 20, 20)
    assert constrain(bbox, (640, 480)) == (10, 10, 20, 20)
