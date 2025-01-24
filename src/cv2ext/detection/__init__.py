# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule for performing simple types of detection.

Classes
-------
:class:`AbstractFramePacker`
    Abstract class for frame packers.
:class:`AnnealingFramePacker`
    A frame packer that uses simulated annealing.
:class:`BlobDetector`
    A simple blob detector class.
:class:`RandomFramePacker`
    A frame packer that randomly samples.

Functions
---------
:func:`detect_blobs`
    Detect blobs in an image.

"""

from __future__ import annotations

from ._blob import BlobDetector, detect_blobs
from ._packer import (
    AbstractFramePacker,
    AbstractGridFramePacker,
    AnnealingFramePacker,
    RandomFramePacker,
)

__all__ = [
    "AbstractFramePacker",
    "AbstractGridFramePacker",
    "AnnealingFramePacker",
    "BlobDetector",
    "RandomFramePacker",
    "detect_blobs",
]
