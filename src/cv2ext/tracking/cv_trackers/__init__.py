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
"""
Contains the wrapped OpenCV trackers.

Classes
-------
BoostingTracker
    A class for tracking objects in videos using the Boosting tracker.
CSRTTracker
    A class for tracking objects in videos using the CSRT tracker.
KCFTracker
    A class for tracking objects in videos using the KCF tracker.
MedianFlowTracker
    A class for tracking objects in videos using the MedianFlow tracker.
MILTracker
    A class for tracking objects in videos using the MIL tracker.
MOSSETracker
    A class for tracking objects in videos using the MOSSE tracker.
TLDTracker
    A class for tracking objects in videos using the TLD tracker.

"""

from __future__ import annotations

from ._boosting import BoostingTracker
from ._csrt import CSRTTracker
from ._kcf import KCFTracker
from ._medianflow import MedianFlowTracker
from ._mil import MILTracker
from ._mosse import MOSSETracker
from ._tld import TLDTracker

__all__ = [
    "BoostingTracker",
    "CSRTTracker",
    "KCFTracker",
    "MILTracker",
    "MOSSETracker",
    "MedianFlowTracker",
    "TLDTracker",
]
