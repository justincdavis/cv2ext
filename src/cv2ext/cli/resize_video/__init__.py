# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Resize videos using opencv.

Functions
---------
:func:`resize_video_cli`
    Resize a video file.
"""

from __future__ import annotations

from ._resize_video import resize_video_cli

__all__ = ["resize_video_cli"]
