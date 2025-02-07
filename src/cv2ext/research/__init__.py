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
:mod:`flexpatch`
    An implementation of the FlexPatch paper.
:mod:`marlin`
    An implementation of the MARLIN paper.

"""

from __future__ import annotations

import contextlib

from . import biglittle

__all__ = [
    "biglittle",
]

# additional modules have specific imports
# ========================================

# marlin module
with contextlib.suppress(ImportError):
    from . import marlin

    __all__ += ["marlin"]

# flexpatch module
with contextlib.suppress(ImportError):
    from . import flexpatch

    __all__ += ["flexpatch"]
