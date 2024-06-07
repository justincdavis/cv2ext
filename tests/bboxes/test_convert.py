# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from cv2ext.bboxes import xywh_to_xyxy, xyxy_to_xywh


def test_xyxy_to_xywh_zeros():
    xywh = xyxy_to_xywh((0, 0, 0, 0))
    assert xywh == (0, 0, 0, 0)


def test_xyxy_to_xywh_equal():
    xywh = xyxy_to_xywh((10, 10, 10, 10))
    assert xywh == (10, 10, 0, 0)


def test_xyxy_to_xywh_basic():
    xywh = xyxy_to_xywh((10, 10, 20, 20))
    assert xywh == (10, 10, 10, 10)


def test_xywh_to_xyxy_zeros():
    xyxy = xywh_to_xyxy((0, 0, 0, 0))
    assert xyxy == (0, 0, 0, 0)


def test_xywh_to_xyxy_equal():
    xyxy = xywh_to_xyxy((10, 10, 0, 0))
    assert xyxy == (10, 10, 10, 10)


def test_xywh_to_xyxy_basic():
    xyxy = xywh_to_xyxy((10, 10, 10, 10))
    assert xyxy == (10, 10, 20, 20)


def test_xyxy_to_xywh_negative():
    xywh = xyxy_to_xywh((-10, -10, 10, 10))
    assert xywh == (-10, -10, 20, 20)


def test_xywh_to_xyxy_negative():
    xyxy = xywh_to_xyxy((-10, -10, 20, 20))
    assert xyxy == (-10, -10, 10, 10)


def test_xyxy_to_xywh_float():
    xywh = xyxy_to_xywh((10.5, 10.5, 20.5, 20.5))
    assert xywh == (10.5, 10.5, 10.0, 10.0)


def test_xywh_to_xyxy_float():
    xyxy = xywh_to_xyxy((10.5, 10.5, 10.0, 10.0))
    assert xyxy == (10.5, 10.5, 20.5, 20.5)
