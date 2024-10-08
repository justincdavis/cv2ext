# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from cv2ext.bboxes import xywh_to_xyxy, xyxy_to_xywh, xyxy_to_nxywh, nxywh_to_xyxy, xywh_to_nxywh, nxywh_to_xywh


# Each conversion has the following tests:
# 1. Test with zeros
# 2. Test with equal values
# 3. Test with basic values (10, 10, 20, 20) or equivalent
# The xyxy <=> xywh also have:
# 4. Test with negative values
# 5. Test with float values

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


def test_xyxy_to_nxywh_zeros():
    nxywh = xyxy_to_nxywh((0, 0, 0, 0), 640, 480)
    assert nxywh == (0, 0, 0, 0)


def test_nxywh_to_xyxy_zeros():
    xyxy = nxywh_to_xyxy((0, 0, 0, 0), 640, 480)
    assert xyxy == (0, 0, 0, 0)


def test_xywh_to_nxywh_zeros():
    nxywh = xywh_to_nxywh((0, 0, 0, 0), 640, 480)
    assert nxywh == (0, 0, 0, 0)


def test_nxywh_to_xywh_zeros():
    xywh = nxywh_to_xywh((0, 0, 0, 0), 640, 480)
    assert xywh == (0, 0, 0, 0)


def test_xyxy_to_nxywh_equal():
    nxywh = xyxy_to_nxywh((10, 10, 10, 10), 640, 640)
    assert nxywh == (0.015625, 0.015625, 0.0, 0.0)


def test_nxywh_to_xyxy_equal():
    xyxy = nxywh_to_xyxy((0.015625, 0.015625, 0.0, 0.0), 640, 640)
    assert xyxy == (10, 10, 10, 10)


def test_xywh_to_nxywh_equal():
    nxywh = xywh_to_nxywh((10, 10, 0, 0), 640, 640)
    assert nxywh == (0.015625, 0.015625, 0.0, 0.0)


def test_nxywh_to_xywh_equal():
    xywh = nxywh_to_xywh((0.015625, 0.015625, 0.0, 0.0), 640, 640)
    assert xywh == (10, 10, 0, 0)


def test_xyxy_to_nxywh_basic():
    nxywh = xyxy_to_nxywh((10, 10, 20, 20), 640, 640)
    assert nxywh == (0.015625, 0.015625,  0.015625, 0.015625)


def test_nxywh_to_xyxy_basic():
    xyxy = nxywh_to_xyxy((0.015625, 0.015625,  0.015625,  0.015625), 640, 640)
    assert xyxy == (10, 10, 20, 20)


def test_xywh_to_nxywh_basic():
    nxywh = xywh_to_nxywh((10, 10, 10, 10), 640, 640)
    assert nxywh == (0.015625, 0.015625, 0.015625, 0.015625)


def test_nxywh_to_xywh_basic():
    xywh = nxywh_to_xywh((0.015625, 0.015625, 0.015625, 0.015625), 640, 640)
    assert xywh == (10, 10, 10, 10)
