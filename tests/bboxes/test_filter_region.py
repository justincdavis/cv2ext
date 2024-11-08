# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from cv2ext.bboxes import filter_bboxes_by_region


def test_basic_0():
    bboxes = [
        (80, 90, 100, 110),
        (90, 100, 110, 120),
    ]

    filt = filter_bboxes_by_region(
        bboxes, (70, 70, 200, 200)
    )

    assert len(filt) == 2


def test_basic_1():
    bboxes = [
        (0, 0, 100, 100)
    ]

    filt = filter_bboxes_by_region(
        bboxes, (0, 0, 50, 100), 0.5
    )
    assert len(filt) == 1
    
    filt = filter_bboxes_by_region(
        bboxes, (0, 0, 50, 100), 0.6
    )
    assert len(filt) == 0
