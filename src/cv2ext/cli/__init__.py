# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Command line interface for cv2ext.

Submodules
----------
annotate
    Annotate videos with bounding boxes.
resize_video
    Resize videos using opencv.
"""

from __future__ import annotations

from . import annotate, convert_annotations, resize_video, timeline

__all__ = ["annotate", "convert_annotations", "resize_video", "timeline"]
