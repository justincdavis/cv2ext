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


def test_match_multiple_jit():
    importlib.reload(cv2ext)
    cv2ext.enable_jit()

    template = cv2.imread(str(Path("data") / "template.png"))
    image = cv2.imread(str(Path("data") / "pictograms.png"))

    output = cv2ext.template.match_multiple(image, template, threshold=0.99)

    assert output is not None
    assert len(output) == 1
    output = output[0]
    assert len(output) == 4

    importlib.reload(cv2ext)


def test_match_multiple_threshold_jit():
    importlib.reload(cv2ext)
    cv2ext.enable_jit()

    template = cv2.imread(str(Path("data") / "template.png"))
    image = cv2.imread(str(Path("data") / "pictograms.png"))


    num_matches = []
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
        output = cv2ext.template.match_multiple(image, template, threshold=thresh)
        assert output is not None
        assert isinstance(output, list)
        assert len(output) > 0
        num_matches.append(len(output))

    # check numbers of matches is equal or decreasing as we increase the threshold
    for i in range(len(num_matches) - 1):
        assert num_matches[i] >= num_matches[i + 1]

    importlib.reload(cv2ext)


def test_match_multiple_max_thresh_jit():
    importlib.reload(cv2ext)
    cv2ext.enable_jit()

    template = cv2.imread(str(Path("data") / "template.png"))
    image = cv2.imread(str(Path("data") / "pictograms.png"))

    output = cv2ext.template.match_multiple(image, template, threshold=0.999)

    assert output is not None
    assert len(output) == 1
    assert output[0] == (308, 308, 458, 454)

    importlib.reload(cv2ext)


def test_match_multiple_above_max_thresh_jit():
    importlib.reload(cv2ext)
    cv2ext.enable_jit()
    
    template = cv2.imread(str(Path("data") / "template.png"))
    image = cv2.imread(str(Path("data") / "pictograms.png"))

    output = cv2ext.template.match_multiple(image, template, threshold=1.1)

    assert output is not None
    assert len(output) == 0

    importlib.reload(cv2ext)
