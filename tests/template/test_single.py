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

import importlib
from pathlib import Path

import cv2
import cv2ext



def test_match_single():
    importlib.reload(cv2ext)

    template = cv2.imread(str(Path("data") / "template.png"))
    image = cv2.imread(str(Path("data") / "pictograms.png"))

    output = cv2ext.template.match_single(image, template)

    assert output is not None
    assert len(output) == 4
    assert output == (308, 308, 458, 454)
