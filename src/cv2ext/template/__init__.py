# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Subpackage containing tools for working with templates in images.

Functions
---------
match_single
    Find the best match of a template in an image.
match_multiple
    Find all matches of a template in an image above a certain threshold.

"""

from __future__ import annotations

from ._core import match_multiple, match_single

__all__ = ["match_multiple", "match_single"]
