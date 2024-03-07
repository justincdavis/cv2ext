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

from pathlib import Path

import cv2
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays

from cv2ext.metrics import ncc


def test_same_image():
    img1 = cv2.imread(str(Path("data") / "testpicto1.png"))
    img2 = cv2.imread(str(Path("data") / "testpicto1.png"))

    assert ncc(img1, img2) == 1.0


def test_different_image_noresize():
    img1 = cv2.imread(str(Path("data") / "testpicto1.png"))
    img2 = cv2.imread(str(Path("data") / "testpicto2.png"))

    try:
        ncc(img1, img2, resize=False)
    except ValueError:
        assert True
    else:
        assert False


def test_different_image():
    img1 = cv2.imread(str(Path("data") / "testpicto1.png"))
    img2 = cv2.imread(str(Path("data") / "testpicto2.png"))

    assert ncc(img1, img2) < 1.0
    assert ncc(img1, img2) > 0.0


@given(arrays(shape=(5,5,3), dtype=np.uint8), arrays(shape=(5,5,3), dtype=np.uint8))
def test_random_images1(i1, i2) -> None:
    retval = ncc(i1, i2, resize=True)
    assert retval <= 1.0
    assert retval >= 0.0


@given(arrays(shape=(10,10,3), dtype=np.uint8), arrays(shape=(10,10,3), dtype=np.uint8))
def test_random_images2(i1, i2) -> None:
    retval = ncc(i1, i2, resize=True)
    assert retval <= 1.0
    assert retval >= 0.0


def test_same_retvals():
    img1 = cv2.imread(str(Path("data") / "testpicto1.png"))
    img2 = cv2.imread(str(Path("data") / "testpicto1.png"))

    assert ncc(img1, img2) == ncc(img2, img1)
