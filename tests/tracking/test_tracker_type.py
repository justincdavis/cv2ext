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
