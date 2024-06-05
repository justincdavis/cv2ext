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
from cv2ext.tracking import MultiTracker, TrackerType


def _create_tracker(use_threads: bool):
    for tracker_type in TrackerType:
        tracker = MultiTracker(tracker_type, use_threads=use_threads)
        assert tracker is not None


def test_create_tracker():
    _create_tracker(False)


def test_create_tracker_threads():
    _create_tracker(True)


def _init_no_error(use_threads: bool):
    image = cv2.imread(str(Path("data") / "pictograms.png"))
    init_bbox = (308, 308, 458, 454)

    for tracker_type in TrackerType:
        tracker = MultiTracker(tracker_type, use_threads=use_threads)
        tracker.init(image, [init_bbox])


def test_init_no_error():
    _init_no_error(False)    


def test_init_no_error_threads():
    _init_no_error(True)


def _data_cycle(use_threads: bool):
    image = cv2.imread(str(Path("data") / "pictograms.png"))
    init_bbox = (308, 308, 458, 454)

    for tracker_type in TrackerType:
        tracker = MultiTracker(tracker_type, use_threads=use_threads)
        tracker.init(image, [init_bbox])
        results = tracker.update(image)

        for success, bbox in results:
            assert success
            assert isinstance(bbox, tuple)


def test_data_cycle():
    _data_cycle(False)


def test_data_cycle_threads():
    _data_cycle(True)


def _many_data_cycle(use_threads: bool):
    image = cv2.imread(str(Path("data") / "pictograms.png"))
    init_bbox = (308, 308, 458, 454)
    init_bboxes = [init_bbox] * 10

    for tracker_type in TrackerType:
        tracker = MultiTracker(tracker_type, use_threads=use_threads)
        tracker.init(image, init_bboxes)
        results = tracker.update(image)
        
        assert len(results) == 10
        for success, bbox in results:
            assert success
            assert isinstance(bbox, tuple)


def test_many_data_cycle():
    _many_data_cycle(False)


def test_many_data_cycle_threads():
    _many_data_cycle(True)
