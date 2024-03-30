# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
Submodule for consolidating common JIT applied functions.

This module contains functions that are commonly used in the JIT compilation
process. The functions are imported into other modules for use in the JIT
compilation process.
If Numba is not installed, then all functions in this module will be None.

Functions
---------
color_mean
    Calculate the mean of a color image.
color_std
    Calculate the standard deviation of a color image.
grayscale_mean
    Calculate the mean of a grayscale image.
grayscale_std
    Calculate the standard deviation of a grayscale image.

"""
from __future__ import annotations

from ._metrics import color_mean, color_std, grayscale_mean, grayscale_std

__all__ = ["color_mean", "color_std", "grayscale_mean", "grayscale_std"]
