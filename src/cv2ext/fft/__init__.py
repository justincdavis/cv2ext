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
Subpackage containing compabilitiy wrappers for FFT functions.

All functions will match the numpy.fft equalivalent functions, but will
use OpenCV or other backends to perform the computation.

Functions
---------
fft2
    Perform a 2D FFT on an image. Same interface as np.fft.fft2.

"""
from __future__ import annotations

from ._fft import fft2

__all__ = ["fft2"]
