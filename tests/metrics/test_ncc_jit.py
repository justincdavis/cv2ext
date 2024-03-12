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
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays


def test_same_image_jit():
    importlib.reload(cv2ext)
    cv2ext.enable_jit()
    
    img1 = cv2.imread(str(Path("data") / "testpicto1.png"))
    img2 = cv2.imread(str(Path("data") / "testpicto1.png"))

    assert cv2ext.metrics.ncc(img1, img2) == 1.0

    importlib.reload(cv2ext)

def test_different_image_noresize_jit():
    importlib.reload(cv2ext)
    cv2ext.enable_jit()

    img1 = cv2.imread(str(Path("data") / "testpicto1.png"))
    img2 = cv2.imread(str(Path("data") / "testpicto2.png"))

    try:
        cv2ext.metrics.ncc(img1, img2, resize=False)
    except ValueError:
        assert True
    else:
        assert False

    importlib.reload(cv2ext)

def test_different_image_jit():
    importlib.reload(cv2ext)
    cv2ext.enable_jit()

    img1 = cv2.imread(str(Path("data") / "testpicto1.png"))
    img2 = cv2.imread(str(Path("data") / "testpicto2.png"))

    assert cv2ext.metrics.ncc(img1, img2) <= 1.0
    assert cv2ext.metrics.ncc(img1, img2) >= -1.0

    importlib.reload(cv2ext)


@given(arrays(shape=(5,5,3), dtype=np.uint8), arrays(shape=(5,5,3), dtype=np.uint8))
def test_random_images1_jit(i1, i2) -> None:
    importlib.reload(cv2ext)
    cv2ext.enable_jit()

    retval = cv2ext.metrics.ncc(i1, i2, resize=True)
    assert retval <= 1.0
    assert retval >= -1.0

    importlib.reload(cv2ext)


@given(arrays(shape=(10,10,3), dtype=np.uint8), arrays(shape=(10,10,3), dtype=np.uint8))
def test_random_images2_jit(i1, i2) -> None:
    importlib.reload(cv2ext)
    cv2ext.enable_jit()

    retval = cv2ext.metrics.ncc(i1, i2, resize=True)
    assert retval <= 1.0
    assert retval >= -1.0

    importlib.reload(cv2ext)


def test_same_retvals_jit():
    importlib.reload(cv2ext)
    cv2ext.enable_jit()

    img1 = cv2.imread(str(Path("data") / "testpicto1.png"))
    img2 = cv2.imread(str(Path("data") / "testpicto1.png"))

    assert cv2ext.metrics.ncc(img1, img2) == cv2ext.metrics.ncc(img2, img1)

    importlib.reload(cv2ext)
