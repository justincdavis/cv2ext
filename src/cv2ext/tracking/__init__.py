# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule containing tools for tracking objects in videos.

Submodules
----------
cv_trackers
    Contains the wrapped OpenCV trackers.

Classes
-------
AbstractTracker
    An abstract class for tracking objects in videos.
AbstractMultiTracker
    An abstract class for tracking multiple objects in videos.
CVTrackerInterface
    A class for making OpenCV trackers compatible with the AbstractTracker interface.
MultiTracker
    A class for tracking multiple objects in videos.
TrackerType
    An enumeration of the available tracker types.
Tracker
    A generic class, which allows many tracking algorithm backends.

"""

from __future__ import annotations

from ._interface import AbstractMultiTracker, AbstractTracker, CVTrackerInterface
from ._multi_tracker import MultiTracker
from ._tracker import Tracker
from ._tracker_type import TrackerType

__all__ = [
    "AbstractMultiTracker",
    "AbstractTracker",
    "CVTrackerInterface",
    "MultiTracker",
    "Tracker",
    "TrackerType",
]
