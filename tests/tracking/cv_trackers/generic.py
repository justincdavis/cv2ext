# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

from pathlib import Path

import cv2
from cv2ext.tracking import Tracker, TrackerType


def check_basic_tracking(tracker_type: TrackerType):
    tracker = Tracker(tracker_type)
    image = cv2.imread(str(Path("data") / "pictograms.png"))
    init_bbox = (308, 308, 458, 454)
    tracker.init(image, init_bbox)

    if tracker_type == TrackerType.TLD:
        for _ in range(10):
            success, bbox = tracker.update(image)

            assert success

            for c in bbox:
                assert isinstance(c, int), f"Expected int but got {type(c2)}"
            assert 0 <= bbox[0] <= image.shape[1] and 0 <= bbox[1] <= image.shape[0]
            assert 0 <= bbox[2] <= image.shape[1] and 0 <= bbox[3] <= image.shape[0]
    else:
        for _ in range(10):
            success, bbox = tracker.update(image)

            assert success
            
            for c1, c2 in zip(init_bbox, bbox):
                assert isinstance(c2, int), f"Expected int but got {type(c2)}"
                assert abs(c1 - c2) <= 3, f"Expected {c1} but got {c2}"
                init_bbox = bbox
            assert 0 <= bbox[0] <= image.shape[1] and 0 <= bbox[1] <= image.shape[0]
            assert 0 <= bbox[2] <= image.shape[1] and 0 <= bbox[3] <= image.shape[0]
