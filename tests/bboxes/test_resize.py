# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from cv2ext.bboxes import resize


def test_resize_zeros():
    assert resize((0, 0, 0, 0), (640, 480), (320, 240)) == (0, 0, 0, 0)


def test_resize_size_limit_down():
    assert resize((0, 0, 640, 480), (640, 480), (320, 240)) == (0, 0, 320, 240)


def test_resize_size_limit_up():
    assert resize((0, 0, 320, 240), (320, 240), (640, 480)) == (0, 0, 640, 480)


def test_resize_double():
    assert resize((10, 10, 50, 50), (640, 480), (1280, 960)) == (20, 20, 100, 100)


def test_resize_half():
    assert resize((10, 10, 50, 50), (640, 480), (320, 240)) == (5, 5, 25, 25)
