# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Contains trackers implemented outside of OpenCV.

Classes
-------
KLTTracker
    A class for tracking objects in videos using the KLT algorithm.

"""

from __future__ import annotations

from ._klt import KLTTracker

__all__ = [
    "KLTTracker",
]
