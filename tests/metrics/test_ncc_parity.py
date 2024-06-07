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

from ..helpers import wrapper, wrapper_jit


@wrapper
def get_results_same() -> float:
    img1 = cv2.imread(str(Path("data") / "testpicto1.png"))
    img2 = cv2.imread(str(Path("data") / "testpicto1.png"))

    return cv2ext.metrics.ncc(img1, img2)


@wrapper_jit
def get_jit_results_same() -> float:
    img1 = cv2.imread(str(Path("data") / "testpicto1.png"))
    img2 = cv2.imread(str(Path("data") / "testpicto1.png"))

    return cv2ext.metrics.ncc(img1, img2)


@wrapper
def get_results_different() -> float:
    img1 = cv2.imread(str(Path("data") / "testpicto1.png"))
    img2 = cv2.imread(str(Path("data") / "testpicto2.png"))

    return cv2ext.metrics.ncc(img1, img2)


@wrapper_jit
def get_jit_results_different() -> float:
    img1 = cv2.imread(str(Path("data") / "testpicto1.png"))
    img2 = cv2.imread(str(Path("data") / "testpicto2.png"))

    return cv2ext.metrics.ncc(img1, img2)


def test_same_results():
    results = get_results_same()
    jit_results = get_jit_results_same()
    assert results == jit_results


def test_different_results():
    results = get_results_different()
    jit_results = get_jit_results_different()
    assert results == jit_results


@wrapper
def run_images(i1: np.ndarray, i2: np.ndarray) -> float:
    return cv2ext.metrics.ncc(i1, i2)


@wrapper_jit
def run_jit_images(i1: np.ndarray, i2: np.ndarray) -> float:
    return cv2ext.metrics.ncc(i1, i2)


@given(arrays(shape=(5, 5, 3), dtype=np.uint8), arrays(shape=(5, 5, 3), dtype=np.uint8))
def test_random(i1, i2) -> None:
    results = run_images(i1, i2)
    jit_results = run_jit_images(i1, i2)
    assert results == jit_results
