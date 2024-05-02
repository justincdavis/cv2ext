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

import cv2
import numpy as np


def fft2(image: np.ndarray) -> np.ndarray:
    """
    Compute an FFT of an image.

    Parameters
    ----------
    image : np.ndarray
        The input image.
        Should be of shape [height, width, 1].

    Returns
    -------
    np.ndarray
        The FFT of the image.

    Raises
    ------
    ValueError
        If the image has more than one channel.

    """
    if image.shape[2] != 1:
        err_msg = f"Image must have one channel. Got {image.shape[2]} channels."
        raise ValueError(err_msg)

    if image.dtype not in [np.float32, np.float64]:
        image = image.astype(np.float32)

    dft: np.ndarray = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft = dft[:, :, 0] + 1j * dft[:, :, 1]
    return dft.view(np.complex64)[:, :, np.newaxis]
