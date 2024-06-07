# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from cv2ext.bboxes import constrain

from ..helpers import wrapper, wrapper_jit


@wrapper
def test_constrain_zeros():
    assert constrain((0, 0, 0, 0), (640, 480)) == (0, 0, 0, 0)


@wrapper
def test_constrain_all_negative():
    bbox = (-10, -10, -5, -5)
    assert constrain(bbox, (640, 480)) == (0, 0, 0, 0)


@wrapper
def test_constrain_all_exceed():
    bbox = (700, 600, 800, 650)
    assert constrain(bbox, (640, 480)) == (640, 480, 640, 480)


@wrapper
def test_constrain_all_within():
    bbox = (10, 10, 20, 20)
    assert constrain(bbox, (640, 480)) == (10, 10, 20, 20)


@wrapper_jit
def test_constrain_zeros_jit():
    assert constrain((0, 0, 0, 0), (640, 480)) == (0, 0, 0, 0)


@wrapper_jit
def test_constrain_all_negative_jit():
    bbox = (-10, -10, -5, -5)
    assert constrain(bbox, (640, 480)) == (0, 0, 0, 0)


@wrapper_jit
def test_constrain_all_exceed_jit():
    bbox = (700, 600, 800, 650)
    assert constrain(bbox, (640, 480)) == (640, 480, 640, 480)


@wrapper_jit
def test_constrain_all_within_jit():
    bbox = (10, 10, 20, 20)
    assert constrain(bbox, (640, 480)) == (10, 10, 20, 20)
