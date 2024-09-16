# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from cv2ext.bboxes import bounding


def test_bounding_single():
    bbox = (0, 0, 10, 10)
    assert bounding([bbox]) == bbox


def test_bounding_simple():
    bboxes = [
        (0, 0, 10, 10),
        (10, 10, 20, 20),
        (20, 20, 30, 30),
    ]
    bbox = bounding(bboxes)
    assert bbox == (0, 0, 30, 30)
