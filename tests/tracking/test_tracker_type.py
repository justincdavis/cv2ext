# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from cv2ext.tracking import cv_trackers, TrackerType


def test_all_types_exist():
    for tracker_type in TrackerType:
        if "_" in tracker_type.name:
            first, second = tracker_type.name.split("_")
            first = first.lower().capitalize()
            second = second.lower().capitalize()
            assert hasattr(cv_trackers, first + second + "Tracker")
        else:
            try:
                assert hasattr(cv_trackers, tracker_type.name.lower().capitalize() + "Tracker")
            except AssertionError:
                assert hasattr(cv_trackers, tracker_type.name + "Tracker")
