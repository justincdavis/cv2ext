# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Implementations of tracking algorithms.

This module provides implementations of multi-object tracking algorithms
including SORT (Simple Online and Realtime Tracking) and its variants.

Classes
-------
:class:`SORT`
    SORT (Simple Online and Realtime Tracking) tracker.
:class:`Tracker`
    Abstract base class for tracking algorithms.
:class:`Track`
    Track representation for multi-object tracking algorithms.

Functions
---------
:func:`associate_tracks_to_detections`
    Associate tracks to detections using the Hungarian algorithm.
:func:`compute_iou_matrix`
    Compute the IoU matrix for track and detection pairs.
:func:`linear_assignment`
    Solve the linear assignment problem using the Hungarian algorithm.

"""

from __future__ import annotations

from ._core import associate_tracks_to_detections, compute_iou_matrix, linear_assignment
from ._sort import SORT
from ._track import Track
from ._tracker import Tracker

__all__ = [
    "SORT",
    "Track",
    "Tracker",
    "associate_tracks_to_detections",
    "compute_iou_matrix",
    "linear_assignment",
]
