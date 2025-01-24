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
    FOUND_NUMBA : bool
        Whether or not a Numba installation was found.
    WARNED_NUMBA_NOT_FOUND : bool
        Whether or not the user has been warned that Numba was
        not found when calling enable_jit.

    """

    JIT: bool = False
    FOUND_NUMBA: bool = False
    WARNED_NUMBA_NOT_FOUND: bool = False


FLAGS = _FLAGS()
