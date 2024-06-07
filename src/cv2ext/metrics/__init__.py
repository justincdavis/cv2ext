# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule containing tools for working with image metrics.

Functions
---------
ncc
    Compute the normalized cross-correlation between two images.
"""

from __future__ import annotations

from ._ncc import ncc

__all__ = ["ncc"]
