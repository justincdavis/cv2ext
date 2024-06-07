# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Subpackage containing abstractions for making IO easier through cv2.

Classes
-------
Display
    A display object for showing images.
Fourcc
    A fourcc codec enum. Used for video writing.
IterableVideo
    An iterable video object.
VideoWriter
    A video writer object.

"""

from __future__ import annotations

from ._display import Display
from ._fourcc import Fourcc
from ._iterablevideo import IterableVideo
from ._writer import VideoWriter

__all__ = ["Display", "Fourcc", "IterableVideo", "VideoWriter"]
