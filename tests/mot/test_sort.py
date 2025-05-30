# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Simple test script for SORT implementation.

This script provides a basic test to verify that the SORT tracker
is working correctly with sample detections.
"""

from __future__ import annotations

from cv2ext.mot._sort import SORT


TEST_DETS = [
    [
        ((100, 50, 200, 150), 0.9, 1),
        ((300, 100, 400, 200), 0.8, 1),
    ],
    [
        ((105, 55, 205, 155), 0.85, 1),
        ((305, 105, 405, 205), 0.82, 1),
    ],
    [
        ((110, 60, 210, 160), 0.87, 1),
    ],
    [
        ((115, 65, 215, 165), 0.9, 1),
    ],
    [
        ((120, 70, 220, 170), 0.88, 1),
        ((250, 80, 350, 180), 0.75, 1),
    ],
]


def test_sort_run() -> None:
    """Test SORT runs."""
    tracker = SORT(iou_threshold=0.3, max_age=5, min_hits=3)
    
    for frame_num, detections in enumerate(TEST_DETS):
        dets = tracker.update(detections)
        if frame_num == 3:
            assert len(dets) > 0
