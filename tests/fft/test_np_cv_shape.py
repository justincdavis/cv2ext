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

import time

from hypothesis import given
from hypothesis.extra.numpy import arrays
import numpy as np

from cv2ext.tracking.kernels import fft2


@given(arrays(shape=(10, 10, 1), dtype=np.uint8))
def test_shape(image: np.ndarray):
    fft_numpy = np.fft.fft2(image)
    fft_opencv = fft2(image)

    assert fft_numpy.shape == fft_opencv.shape
    assert fft_numpy.ndim == fft_opencv.ndim
    assert fft_numpy.size == fft_opencv.size
    # double on itemsize, since fft2 returns complex64, while np.fft.fft2 returns complex128
    assert fft_numpy.itemsize == fft_opencv.itemsize * 2


@given(arrays(shape=(20, 20, 1), dtype=np.uint8))
def test_perf(image: np.ndarray):
    times = []

    for _ in range(100):
        t0 = time.perf_counter()
        fft_numpy = np.fft.fft2(image)
        t1 = time.perf_counter()
        fft_opencv = fft2(image)
        t2 = time.perf_counter()

        assert fft_numpy.shape == fft_opencv.shape
        times.append((t1 - t0, t2 - t1))

    numpy_time = np.mean([t[0] for t in times])
    opencv_time = np.mean([t[1] for t in times])

    assert  numpy_time > opencv_time
