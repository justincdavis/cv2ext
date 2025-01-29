# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
A submodule containing the implementation of the BigLittle paper.

Paper: https://ieeexplore.ieee.org/document/7331375

Classes
-------
:class:`BigLittle`
    The BigLittle methodology as a detection interface.

"""

from __future__ import annotations

from ._core import BigLittle

__all__ = [
    "BigLittle",
]
