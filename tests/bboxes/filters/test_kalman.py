# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from cv2ext.bboxes.filters import KalmanFilter


def test_kalman_filter_no_change():
    bbox1 = (50, 50, 100, 100)
    kfilter = KalmanFilter(bbox1)
    for _ in range(100):
        bbox2 = kfilter(bbox1)
        assert bbox2 == bbox1
