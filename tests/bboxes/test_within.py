# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from cv2ext.bboxes import within


def test_within_zeros():
    assert within((0, 0, 1, 1), (10, 10))


def test_within_equal():
    assert within((0, 0, 9, 9), (10, 10))


def test_within_basic():
    assert within((1, 1, 8, 8), (10, 10))


def test_within_negative():
    assert not within((-1, -1, 1, 1), (10, 10))


def test_within_fully_above():
    assert not within((11, 11, 20, 20), (10, 10))


def test_within_fully_below():
    assert not within((-5, -5, -1, -1), (10, 10))
