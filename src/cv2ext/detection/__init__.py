# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule for performing simple types of detection.

Classes
-------
BlobDetector
    A simple blob detector class.

Functions
---------
detect_blobs
    Detect blobs in an image.

"""

from __future__ import annotations

from ._blob import BlobDetector, detect_blobs

__all__ = ["BlobDetector", "detect_blobs"]
