# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Tools for working with cameras.

Functions
---------
:func:`generate_camera_intrinsics`
    Generate camera intrinsics from a list of images containing a chessboard pattern.

"""
from __future__ import annotations

from ._intrinsics import generate_camera_intrinsics

__all__ = ["generate_camera_intrinsics"]
