# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from cv2ext.tracking import TrackerType, MultiTrackerType

from ..generic import check_basic_tracking, check_full_tracking, check_basic_multi_tracking, check_full_multi_tracking


def test_klt_basic():
    check_basic_tracking(TrackerType.KLT)


def test_klt_full():
    check_full_tracking(TrackerType.KLT, use_gray=False)


def test_klt_full_gray():
    check_full_tracking(TrackerType.KLT, use_gray=True)


def test_multi_klt_basic():
    check_basic_multi_tracking(MultiTrackerType.KLT)


def test_multi_klt_full():
    check_full_multi_tracking(MultiTrackerType.KLT, use_gray=False)


def test_multi_klt_full_gray():
    check_full_multi_tracking(MultiTrackerType.KLT, use_gray=True)
