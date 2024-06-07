# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

import cv2
import cv2ext
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays

from ..helpers import wrapper


@wrapper
def test_same_image():
    img1 = cv2.imread(str(Path("data") / "testpicto1.png"))
    img2 = cv2.imread(str(Path("data") / "testpicto1.png"))

    assert cv2ext.metrics.ncc(img1, img2) == 1.0


@wrapper
def test_different_image_noresize():
    img1 = cv2.imread(str(Path("data") / "testpicto1.png"))
    img2 = cv2.imread(str(Path("data") / "testpicto2.png"))

    try:
        cv2ext.metrics.ncc(img1, img2, resize=False)
    except ValueError:
        assert True
    else:
        assert False


@wrapper
def test_different_image():
    img1 = cv2.imread(str(Path("data") / "testpicto1.png"))
    img2 = cv2.imread(str(Path("data") / "testpicto2.png"))

    assert cv2ext.metrics.ncc(img1, img2) <= 1.0
    assert cv2ext.metrics.ncc(img1, img2) >= -1.0


@wrapper
@given(arrays(shape=(5, 5, 3), dtype=np.uint8), arrays(shape=(5, 5, 3), dtype=np.uint8))
def test_random_images1(i1, i2) -> None:
    retval = cv2ext.metrics.ncc(i1, i2, resize=True)
    assert retval <= 1.0
    assert retval >= -1.0


@wrapper
@given(
    arrays(shape=(10, 10, 3), dtype=np.uint8), arrays(shape=(10, 10, 3), dtype=np.uint8)
)
def test_random_images2(i1, i2) -> None:
    retval = cv2ext.metrics.ncc(i1, i2, resize=True)
    assert retval <= 1.0
    assert retval >= -1.0


@wrapper
def test_same_retvals():
    img1 = cv2.imread(str(Path("data") / "testpicto1.png"))
    img2 = cv2.imread(str(Path("data") / "testpicto1.png"))

    assert cv2ext.metrics.ncc(img1, img2) == cv2ext.metrics.ncc(img2, img1)
