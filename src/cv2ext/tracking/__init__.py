# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule containing tools for tracking objects in videos.

Submodules
----------
:mod:`cv_trackers`
    Contains the wrapped OpenCV trackers.
:mod:`trackers`
    Contains trackers implemented in cv2ext.

Classes
-------
:class:`AbstractTracker`
    An abstract class for tracking objects in videos.
:class:`AbstractMultiTracker`
    An abstract class for tracking multiple objects in videos.
:class:`CVTrackerInterface`
    A class for making OpenCV trackers compatible with the AbstractTracker interface.
:class:`MultiTracker`
    A class for tracking multiple objects in videos.
:class:`TrackerType`
    An enumeration of the available tracker types.
:class:`MultiTrackerType`
    An enumeration of available multi-tracker types without MultiTracker wrapper.
:class:`Tracker`
    A generic class, which allows many tracking algorithm backends.

"""

from __future__ import annotations

from . import cv_trackers, trackers
from ._interface import AbstractMultiTracker, AbstractTracker, CVTrackerInterface
from ._multi_tracker import MultiTracker
from ._tracker import Tracker
from ._tracker_type import MultiTrackerType, TrackerType

__all__ = [
    "AbstractMultiTracker",
    "AbstractTracker",
    "CVTrackerInterface",
    "MultiTracker",
    "MultiTrackerType",
    "Tracker",
    "TrackerType",
    "cv_trackers",
    "trackers",
]
