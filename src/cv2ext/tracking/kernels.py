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
A collection of kernels for building CV trackers.

Functions
---------
window
    Applies a windowing to the image.
crop
    Crops an image given a bounding box.
max_response
    Finds the maximum response in the response map.
dense_gaussian_kernel
    Computes the dense Gaussian kernel as used in the CSK tracker.
csk_target
    Generates the target for the CSK tracker.
csk_train
    Trains the CSK tracker.
csk_detection
    Calculates new responses for the CSK tracker.

"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from cv2ext import _FLAGSOBJ

_log = logging.getLogger(__name__)

try:
    from numba import jit  # type: ignore[import-untyped]
except ImportError:
    jit = None
    if _FLAGSOBJ.USEJIT:
        _log.warning(
            "Numba not installed, but JIT has been enabled. Not using JIT for IOU.",
        )


def _window_kernel_jit(
    window_func: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    if _FLAGSOBJ.USEJIT and jit is not None:
        _log.info("JIT Compiling: Window")
        window_func = jit(window_func, nopython=True, parallel=_FLAGSOBJ.PARALLEL)
    return window_func


@_window_kernel_jit
def _window_kernel(image: np.ndarray) -> np.ndarray:
    height = image.shape[0]
    width = image.shape[1]

    j = np.arange(0, width)
    i = np.arange(0, height)
    jj, ii = np.meshgrid(j, i)
    left: np.ndarray = np.sin(np.pi * jj / width)
    right: np.ndarray = np.sin(np.pi * ii / height)
    window: np.ndarray = left * right
    normalized: np.ndarray = image / 255
    offset: np.ndarray = normalized - 0.5
    result: np.ndarray = window * offset
    return result


def window(image: np.ndarray) -> np.ndarray:
    """
    Window a given image.

    The input image should be range 0-255.
    The output image is -0.5 to 0.5.
    The windowing function is a sin window.

    Parameters
    ----------
    image : np.ndarray
        The input image.

    Returns
    -------
    np.ndarray
        The windowed image.

    """
    return _window_kernel(image)


def _crop_kernel_jit(
    crop_func: Callable[[np.ndarray, tuple[int, int, int, int]], np.ndarray],
) -> Callable[[np.ndarray, tuple[int, int, int, int]], np.ndarray]:
    if _FLAGSOBJ.USEJIT and jit is not None:
        _log.info("JIT Compiling: Crop")
        crop_func = jit(crop_func, nopython=True, parallel=_FLAGSOBJ.PARALLEL)
    return crop_func


@_crop_kernel_jit
def _crop_kernel(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    height, width = y2 - y1, x2 - x1
    pad_y = [0, 0]
    pad_x = [0, 0]

    if (y1 - height / 2) < 0:
        y_up = 0
        pad_y[0] = int(-(y1 - height / 2))
    else:
        y_up = int(y1 - height / 2)

    if (y1 + 3 * height / 2) > image.shape[0]:
        y_down = image.shape[0]
        pad_y[1] = int((y1 + 3 * height / 2) - image.shape[0])
    else:
        y_down = int(y1 + 3 * height / 2)

    if (x1 - width / 2) < 0:
        x_left = 0
        pad_x[0] = int(-(x1 - width / 2))
    else:
        x_left = int(x1 - width / 2)

    if (x1 + 3 * width / 2) > image.shape[1]:
        x_right = image.shape[1]
        pad_x[1] = int((x1 + 3 * width / 2) - image.shape[1])
    else:
        x_right = int(x1 + 3 * width / 2)

    # print(pad_y, pad_x)
    # print(y_up, y_down, x_left, x_right)
    cropped_img = image[y_up:y_down, x_left:x_right]
    padded_img = np.pad(cropped_img, (pad_y, pad_x), mode="edge")
    padded_img = np.pad(
        padded_img,
        (
            (
                max(0, (image.shape[0] - padded_img.shape[0]) // 2),
                max(
                    0,
                    (image.shape[0] - padded_img.shape[0])
                    - (image.shape[0] - padded_img.shape[0]) // 2,
                ),
            ),
            (
                max(0, (image.shape[1] - padded_img.shape[1]) // 2),
                max(
                    0,
                    (image.shape[1] - padded_img.shape[1])
                    - (image.shape[1] - padded_img.shape[1]) // 2,
                ),
            ),
        ),
        mode="edge",
    )
    return _window_kernel(padded_img)


def crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """
    Crops an image given a bounding box.

    If the image is too large, the function pads the image with the edge values.
    The image is cropped such that double the height and width
    of the bounding box is included.

    Parameters
    ----------
    image : np.ndarray
        The input image.
    bbox : tuple[int, int, int, int]
        The bounding box of the target.
        The bbox is represented as (x1, y1, x2, y2).

    Returns
    -------
    np.ndarray
        The cropped image.

    """
    return _crop_kernel(image, bbox)


def _csk_target_kernel_jit(
    csk_target_func: Callable[[int, int], np.ndarray],
) -> Callable[[int, int], np.ndarray]:
    if _FLAGSOBJ.USEJIT and jit is not None:
        _log.info("JIT Compiling: CSK Target")
        csk_target_func = jit(
            csk_target_func,
            nopython=True,
            parallel=_FLAGSOBJ.PARALLEL,
        )
    return csk_target_func


@_csk_target_kernel_jit
def _csk_target_kernel(height: int, width: int) -> np.ndarray:
    double_w, double_h = width * 2, height * 2
    s = np.sqrt(double_w * double_h) / 16

    x = np.arange(0, double_w)
    y = np.arange(0, double_h)
    xx, yy = np.meshgrid(x, y)
    result: np.ndarray = np.exp(
        -1.0 * ((xx - width) ** 2 + (yy - height) ** 2) / (s**2),
    )
    return result


def csk_target(height: int, width: int) -> np.ndarray:
    """
    Generate the target for the CSK tracker.

    Parameters
    ----------
    height : int
        The height of the target.
    width : int
        The width of the target.

    Returns
    -------
    np.ndarray
        The target for the CSK tracker.

    """
    return _csk_target_kernel(height, width)


def _max_response_kernel_jit(
    max_response_func: Callable[[np.ndarray], tuple[int, int]],
) -> Callable[[np.ndarray], tuple[int, int]]:
    if _FLAGSOBJ.USEJIT and jit is not None:
        _log.info("JIT Compiling: Max Response")
        max_response_func = jit(
            max_response_func,
            nopython=True,
            parallel=_FLAGSOBJ.PARALLEL,
        )
    return max_response_func


@_max_response_kernel_jit
def _max_response_kernel(response: np.ndarray) -> tuple[int, int]:
    result: tuple[int, int] = np.unravel_index(np.argmax(response, axis=None), response.shape)  # type: ignore[assignment]
    return result


def max_response(response: np.ndarray) -> tuple[int, int]:
    """
    Find the maximum response in the response map.

    Parameters
    ----------
    response : np.ndarray
        The response map.

    Returns
    -------
    tuple[int, int]
        The coordinates of the maximum response.

    """
    return _max_response_kernel(response)


def _dgk_sub_kernel_jit(
    dgk_sub_func: Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray],
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]:
    if _FLAGSOBJ.USEJIT and jit is not None:
        _log.info("JIT Compiling: Dense Gaussian Kernel Sub")
        dgk_sub_func = jit(dgk_sub_func, nopython=True, parallel=_FLAGSOBJ.PARALLEL)
    return dgk_sub_func


@_dgk_sub_kernel_jit
def _dgk_sub_kernel(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Sub-routine for computing the dense Gaussian kernel.

    Parameters
    ----------
    x : np.ndarray
        The x data.
    y : np.ndarray
        The y data.
    z : np.ndarray
        The z data.
    sigma : float
        The bandwidth of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        The dense Gaussian kernel.

    """
    dot_x = np.dot(
        np.conj(x.flatten()),
        x.flatten(),
    )
    dot_y = np.dot(
        np.conj(y.flatten()),
        y.flatten(),
    )
    intermediate = dot_x + dot_y - 2 * z
    result: np.ndarray = np.exp(-1.0 / sigma**2 * np.abs(intermediate) / np.size(x))
    return result


def _dgk_kernel_jit(
    dgk_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
    if _FLAGSOBJ.USEJIT and jit is not None:
        _log.info("JIT Compiling: Dense Gaussian Kernel")
        dgk_func = jit(dgk_func, nopython=True, parallel=_FLAGSOBJ.PARALLEL)
    return dgk_func


@_dgk_kernel_jit
def _dgk_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    fft_x = np.fft.fft2(x)
    fft_y = np.fft.fft2(y)
    conj_fft_y = np.conj(fft_y)
    combo = fft_x * conj_fft_y
    fft_response = np.fft.fftshift(np.fft.ifft2(combo))
    return _dgk_sub_kernel(x, x, fft_response, sigma)


def dense_gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute the dense Gaussian kernel as used in the CSK tracker.

    Parameters
    ----------
    x : np.ndarray
        The x coordinates.
    y : np.ndarray
        The y coordinates.
    sigma : float
        The bandwidth of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        The dense Gaussian kernel.

    """
    return _dgk_kernel(x, y, sigma)


def _csk_train_kernel_jit(
    csk_train_func: Callable[[np.ndarray, np.ndarray, float, float], np.ndarray],
) -> Callable[[np.ndarray, np.ndarray, float, float], np.ndarray]:
    if _FLAGSOBJ.USEJIT and jit is not None:
        _log.info("JIT Compiling: CSK Training")
        csk_train_func = jit(csk_train_func, nopython=True, parallel=_FLAGSOBJ.PARALLEL)
    return csk_train_func


@_csk_train_kernel_jit
def _csk_train_kernel(
    image: np.ndarray,
    target: np.ndarray,
    sigma: float,
    lmbda: float,
) -> np.ndarray:
    kernel = _dgk_kernel(image, image, sigma)
    return np.fft.fft2(target) / (np.fft.fft2(kernel) + lmbda)


def csk_train(
    image: np.ndarray,
    target: np.ndarray,
    sigma: float,
    lmbda: float,
) -> np.ndarray:
    """
    Create the trained data for the CSK tracker.

    Parameters
    ----------
    image : np.ndarray
        The input image.
    target : np.ndarray
        The target for the CSK tracker.
    sigma : float
        The bandwidth of the Gaussian kernel.
    lmbda : float
        The regularization parameter.

    Returns
    -------
    np.ndarray
        The trained CSK tracker.

    """
    return _csk_train_kernel(image, target, sigma, lmbda)


def _csk_detection_kernel_jit(
    csk_detection_func: Callable[
        [np.ndarray, np.ndarray, np.ndarray, float],
        np.ndarray,
    ],
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]:
    if _FLAGSOBJ.USEJIT and jit is not None:
        _log.info("JIT Compiling: CSK Detection")
        csk_detection_func = jit(
            csk_detection_func,
            nopython=True,
            parallel=_FLAGSOBJ.PARALLEL,
        )
    return csk_detection_func


@_csk_detection_kernel_jit
def _csk_detection_kernel(
    last_alphaf: np.ndarray,
    prev_window: np.ndarray,
    window: np.ndarray,
    sigma: float,
) -> np.ndarray:
    kernel = _dgk_kernel(prev_window, window, sigma)
    fft_kernel = np.fft.fft2(kernel)
    combo = np.fft.ifft2(last_alphaf * fft_kernel)
    return np.real(combo)


def csk_detection(
    last_alphaf: np.ndarray,
    prev_window: np.ndarray,
    window: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Calculate new responses for the CSK tracker.

    Parameters
    ----------
    window : np.ndarray
        The window of the current frame.
    prev_window : np.ndarray
        The window of the previous frame.
    last_alphaf : np.ndarray
        The previously trained alpha_f.
    sigma : float
        The bandwidth of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        The new responses.

    """
    return _csk_detection_kernel(last_alphaf, prev_window, window, sigma)
