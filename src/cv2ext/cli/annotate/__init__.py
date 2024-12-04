# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Annotate videos with bounding boxes.

Functions
---------
:func:`annotate_cli`
    Annotate a video with bounding boxes
"""

from __future__ import annotations

from ._annotate import annotate_cli

__all__ = ["annotate_cli"]
