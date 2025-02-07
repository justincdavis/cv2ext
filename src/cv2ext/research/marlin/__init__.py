# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule for the Marlin methodology.

Paper: https://www.cs.ucr.edu/~jiasi/pub/marlin_sensys19.pdf

Classes
-------
:class:`ChangeDetector`
    The change detector as described by the paper.
:class:`Marlin`
    The overall Marlin methdology.

"""

from __future__ import annotations

from ._change import ChangeDetector
from ._core import Marlin

__all__ = [
    "ChangeDetector",
    "Marlin",
]
