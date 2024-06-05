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

from cv2ext.tracking.cv_trackers import BoostingTracker, CSRTTracker, KCFTracker, MedianFlowTracker, MILTracker, MOSSETracker, TLDTracker


def test_create_boosting_tracker():
    tracker = BoostingTracker()
    assert tracker is not None


def test_create_csrt_tracker():
    tracker = CSRTTracker()
    assert tracker is not None


def test_create_kcf_tracker():
    tracker = KCFTracker()
    assert tracker is not None


def test_create_median_flow_tracker():
    tracker = MedianFlowTracker()
    assert tracker is not None


def test_create_mil_tracker():
    tracker = MILTracker()
    assert tracker is not None


def test_create_mosse_tracker():
    tracker = MOSSETracker()
    assert tracker is not None


def test_create_tld_tracker():
    tracker = TLDTracker()
    assert tracker is not None
