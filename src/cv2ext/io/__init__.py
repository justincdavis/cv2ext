# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Subpackage containing abstractions for making IO easier through cv2.

Classes
-------
:class:`Display`
    A display object for showing images.
:class:`Fourcc`
    A fourcc codec enum. Used for video writing.
:class:`IterableVideo`
    An iterable video object.
:class:`VideoWriter`
    A video writer object.

Functions
---------
:func:`find_all_cameras`
    Find all available webcams using cv2ext.IterableVideo.

"""

from __future__ import annotations

from ._display import Display
from ._fourcc import Fourcc
from ._iterablevideo import IterableVideo
from ._webcam import find_all_cameras
from ._writer import VideoWriter

__all__ = ["Display", "Fourcc", "IterableVideo", "VideoWriter", "find_all_cameras"]
