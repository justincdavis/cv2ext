# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Research implementations for various papers/methods.

These submodules do not fit directly within any other submodule
as an algorithm. Instead these represent higher-order methods
which typically have some offline/online workflow.

These are implemented to work together with the cv2ext API
and provide ease-of-use.

Submodules
----------
:mod:`biglittle`
    An implementation of the BigLittle paper.
:mod:`marlin`
    An implementation of the MARLIN paper.
:mod:`shift`
    An implementation of the SHIFT paper.

"""

from __future__ import annotations

import contextlib

from . import biglittle

__all__ = [
    "biglittle",
    "marlin",
]

# additional modules have specific imports
# ========================================

# marlin module
with contextlib.suppress(ImportError):
    from . import marlin

    __all__ += ["marlin"]

# shift module
with contextlib.suppress(ImportError):
    from . import shift

    __all__ += ["shift"]
