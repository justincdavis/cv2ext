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
