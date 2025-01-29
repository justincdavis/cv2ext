# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Contains trackers implemented outside of OpenCV.

Classes
-------
:class:`KLTTracker`
    A class for tracking objects in videos using the KLT algorithm.
:class:`KLTMultiTracker`
    A class for tracking multi objects in videos using the KLT algorithm.

"""

from __future__ import annotations

from ._klt import KLTMultiTracker, KLTTracker

__all__ = [
    "KLTMultiTracker",
    "KLTTracker",
]
