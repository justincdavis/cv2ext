# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Utilities for working with videos.

Functions
---------
create_timeline
    Create a timeline image of a video.
video_from_images
    Create a video from a directory of images.

"""

from __future__ import annotations

from ._images import video_from_images
from ._timeline import create_timeline

__all__ = ["create_timeline", "video_from_images"]
