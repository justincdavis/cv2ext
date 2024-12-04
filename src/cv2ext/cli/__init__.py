# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Command line interface for cv2ext.

Submodules
----------
:mod:`annotate`
    Annotate videos with bounding boxes.
:mod:`convert_annotations`
    Convert annotations from one format to another.
:mod:`convert_video_color`
    Convert the color of a video between BGR/RGB.
:mod:`resize_image`
    Resize an image from the command line.
:mod:`resize_video`
    Resize and entire video from the command line.
:mod:`timeline`
    Generate a timeline of a given video.
:mod:`video_from_images`
    Generate a video from a directory of images.

"""

from __future__ import annotations

from . import (
    annotate,
    convert_annotations,
    convert_video_color,
    resize_image,
    resize_video,
    timeline,
    video_from_images,
)

__all__ = [
    "annotate",
    "convert_annotations",
    "convert_video_color",
    "resize_image",
    "resize_video",
    "timeline",
    "video_from_images",
]
