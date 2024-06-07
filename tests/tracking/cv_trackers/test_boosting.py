# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from cv2ext.tracking import TrackerType

from ..generic import check_basic_tracking, check_full_tracking


def test_boosting_basic():
    check_basic_tracking(TrackerType.BOOSTING)


def test_boosting_full():
    check_full_tracking(TrackerType.BOOSTING, use_gray=False)


def test_boosting_full_gray():
    # this should throw a ValueError, assert it does
    try:
        check_full_tracking(TrackerType.BOOSTING, use_gray=True)
        assert False
    except ValueError:
        assert True
