# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from cv2ext.bboxes import valid


def test_valid_zeros():
    assert not valid((0, 0, 0, 0))


def test_valid_negative():
    assert not valid((-10, -10, -20, -20))
    assert not valid((-20, -20, -10, -10))
    assert not valid((-10, -10, 10, 10))
    assert not valid((10, 10, -10, -10))


def test_valid_basic():
    assert valid((10, 10, 20, 20))
    assert valid((0, 0, 10, 10))
    assert valid((1, 2, 3, 4))


def test_not_valid_basic():
    assert not valid((10, 10, 5, 5))
    assert not valid((10, 10, 0, 0))
