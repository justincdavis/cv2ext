# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _FLAGS:
    """
    Class for storing flags for cv2ext.

    Attributes
    ----------
    JIT : bool
        Whether or not to use jit.

    """

    JIT: bool = False


FLAGS = _FLAGS()
