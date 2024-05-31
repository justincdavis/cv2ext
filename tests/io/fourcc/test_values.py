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

from cv2ext import Fourcc


def test_all_ints():
    values = [e.value for e in Fourcc]
    for v in values:
        assert isinstance(v, int)
        assert v >= 0

def test_all_unique():
    values = [e.value for e in Fourcc]
    num_vals = len(values)
    set_vals = set()
    for v in values:
        set_vals.add(v)
    assert len(set_vals) == num_vals
