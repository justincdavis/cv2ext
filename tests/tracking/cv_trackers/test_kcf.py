# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from cv2ext.tracking import TrackerType

from ..generic import check_basic_tracking, check_full_tracking


def test_kcf_basic():
    check_basic_tracking(TrackerType.KCF)


def test_kcf_full():
    check_full_tracking(TrackerType.KCF, use_gray=False)


def test_kcf_full_gray():
    try:
        check_full_tracking(TrackerType.KCF, use_gray=True)
        assert False
    except ValueError:
        assert True
