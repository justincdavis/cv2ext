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
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import numpy as np

try:
    from numba import jit  # type: ignore[import-untyped]

    def _meandec(f: Callable[[np.ndarray], float]) -> Callable[[np.ndarray], float]:
        return jit(f, nopython=True)  # type: ignore[no-any-return]

    def _stddec(
        f: Callable[[np.ndarray, float], float],
    ) -> Callable[[np.ndarray, float], float]:
        return jit(f, nopython=True)  # type: ignore[no-any-return]

    @_meandec
    def grayscale_mean(img1: np.ndarray) -> float:
        """
        Calculate the mean of a grayscale image.

        Parameters
        ----------
        img1 : np.ndarray
            The grayscale image.

        Returns
        -------
        float
            The mean of the image.

        """
        m = 0.0
        width, height = img1.shape[0], img1.shape[1]
        for i in range(width):
            for j in range(height):
                m += img1[i, j]
        return m / (width * height)

    @_meandec
    def color_mean(img1: np.ndarray) -> float:
        """
        Calculate the mean of a color image.

        Parameters
        ----------
        img1 : np.ndarray
            The color image.

        Returns
        -------
        float
            The mean of the image.

        """
        m = 0.0
        width, height = img1.shape[0], img1.shape[1]
        for i in range(width):
            for j in range(height):
                m += img1[i, j, 0] + img1[i, j, 1] + img1[i, j, 2]
        return m / (width * height * 3)

    @_stddec
    def grayscale_std(img1: np.ndarray, mean: float) -> float:
        """
        Calculate the standard deviation of a grayscale image.

        Parameters
        ----------
        img1 : np.ndarray
            The grayscale image.
        mean : float
            The mean of the image.

        Returns
        -------
        float
            The standard deviation of the image.

        """
        m = 0.0
        width, height = img1.shape[0], img1.shape[1]
        for i in range(width):
            for j in range(height):
                t = img1[i, j] - mean
                m += t * t
        return float((m / (width * height)) ** 0.5)

    @_stddec
    def color_std(img1: np.ndarray, mean: float) -> float:
        """
        Calculate the standard deviation of a color image.

        Parameters
        ----------
        img1 : np.ndarray
            The color image.
        mean : float
            The mean of the image.

        Returns
        -------
        float
            The standard deviation of the image.

        """
        m = 0.0
        width, height = img1.shape[0], img1.shape[1]
        for i in range(width):
            for j in range(height):
                t1 = img1[i, j, 0] - mean
                t2 = img1[i, j, 1] - mean
                t3 = img1[i, j, 2] - mean
                m += t1 * t1 + t2 * t2 + t3 * t3
        return float((m / (width * height * 3)) ** 0.5)

except ImportError:
    grayscale_mean = None  # type: ignore[assignment]
    color_mean = None  # type: ignore[assignment]
    grayscale_std = None  # type: ignore[assignment]
    color_std = None  # type: ignore[assignment]
