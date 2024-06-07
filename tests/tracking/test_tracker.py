# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

import cv2
from cv2ext.tracking import Tracker, TrackerType


def test_create_tracker():
    for tracker_type in TrackerType:
        tracker = Tracker(tracker_type)
        assert tracker is not None


def test_results_close():
    results = []
    image = cv2.imread(str(Path("data") / "pictograms.png"))
    init_bbox = (308, 308, 458, 454)
    for tracker_type in TrackerType:
        # TLD tracker has poor results
        if tracker_type == TrackerType.TLD:
            continue
        tracker = Tracker(tracker_type)
        tracker.init(image, init_bbox)

        results.append(tracker.update(image)[1])

    for i in range(1, len(results)):
        for c1, c2 in zip(results[i - 1], results[i]):
            assert abs(c1 - c2) <= 3, f"Expected {c1} but got {c2}"
