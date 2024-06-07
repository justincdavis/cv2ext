# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from cv2ext.tracking import TrackerType

from .generic import check_basic_tracking, check_full_tracking


def test_mil_basic():
    check_basic_tracking(TrackerType.MIL)


def test_mil_full():
    check_full_tracking(TrackerType.MIL, use_gray=False)


def test_mil_full_gray():
    check_full_tracking(TrackerType.MIL, use_gray=True)
