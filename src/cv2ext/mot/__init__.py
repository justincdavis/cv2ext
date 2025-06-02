# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Implementations of tracking algorithms.

Classes
-------
:class:`SORT`
    SORT (Simple Online and Realtime Tracking) tracker.
:class:`Tracker`
    Abstract base class for tracking algorithms.
:class:`Track`
    Track representation for multi-object tracking algorithms.

"""

from __future__ import annotations

from ._sort import SORT
from ._track import Track
from ._tracker import Tracker

__all__ = [
    "SORT",
    "Track",
    "Tracker",
]
