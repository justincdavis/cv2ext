# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from cv2ext.tracking import cv_trackers, TrackerType, trackers


def test_all_types_exist():
    for tracker_type in TrackerType:
        if "_" in tracker_type.name:
            first, second = tracker_type.name.split("_")
            first = first.lower().capitalize()
            second = second.lower().capitalize()
            assert hasattr(cv_trackers, first + second + "Tracker")
        else:
            cv_check1 = hasattr(cv_trackers, tracker_type.name.lower().capitalize() + "Tracker")
            cv_check2 = hasattr(cv_trackers, tracker_type.name + "Tracker")
            try:
                assert cv_check1 or cv_check2
            except AssertionError:
                # if tracker is not a CV tracker than must be in trackers module
                other_check1 = hasattr(trackers, tracker_type.name.lower().capitalize() + "Tracker")
                other_check2 = hasattr(trackers, tracker_type.name + "Tracker")
                assert other_check1 or other_check2
