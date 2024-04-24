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

import numpy as np


def window(image: np.ndarray) -> np.ndarray:
    """
    Applies a windowing to the image.

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
    height = image.shape[0]
    width = image.shape[1]

    j = np.arange(0,width)
    i = np.arange(0,height)
    jj, ii = np.meshgrid(j,i)
    window = np.sin(np.pi*jj/width)*np.sin(np.pi*ii/height)
    return window*((image/255)-0.5)


def crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """
    This function crops an image given a bounding box.

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
    x1, y1, x2, y2 = bbox
    height, width = y2 - y1, x2 - x1
    pad_y = [0, 0]
    pad_x = [0, 0]

    if (y1-height/2) < 0:
        y_up = 0
        pad_y[0] = int(-(y1-height/2))
    else:
        y_up = int(y1-height/2)

    if (y1+3*height/2) > image.shape[0]:
        y_down = image.shape[0]
        pad_y[1] = int((y1+3*height/2) - image.shape[0])
    else:
        y_down = int(y1+3*height/2)

    if (x1-width/2) < 0:
        x_left = 0
        pad_x[0] = int(-(x1-width/2))
    else:
        x_left = int(x1-width/2)

    if (x1+3*width/2) > image.shape[1]:
        x_right = image.shape[1]
        pad_x[1] = int((x1+3*width/2) - image.shape[1])
    else:
        x_right = int(x1+3*width/2)
    
    # print(pad_y, pad_x)
    # print(y_up, y_down, x_left, x_right)
    cropped_img = image[y_up:y_down, x_left:x_right]
    padded_img = np.pad(cropped_img, (pad_y, pad_x), mode="edge")
    return window(padded_img)


def csk_target(height: int, width: int) -> np.ndarray:
    """
    This function generates the target for the CSK tracker.

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
    double_w, double_h = width * 2, height * 2
    s = np.sqrt(double_w * double_h) / 16

    x = np.arange(0, double_w)
    y = np.arange(0, double_h)
    xx, yy = np.meshgrid(x, y)
    return np.exp(-1.0 * ((xx - width)**2 + (yy - height)**2) / (s**2))


def max_response(response: np.ndarray) -> tuple[int, int]:
    """
    This function finds the maximum response in the response map.

    Parameters
    ----------
    response : np.ndarray
        The response map.

    Returns
    -------
    tuple[int, int]
        The coordinates of the maximum response.
    """
    return np.unravel_index(np.argmax(response, axis=None), response.shape)


def _dgk_sub(x: np.ndarray, y: np.ndarray, z: np.ndarray, sigma: float) -> np.ndarray:
    """
    Sub-routine for computing the dense Gaussian kernel.
    
    Parameters
    ----------
    x : np.ndarray
        The x coordinates.
    y : np.ndarray
        The y coordinates.
    z : np.ndarray
        The z coordinates.
    sigma : float
        The bandwidth of the Gaussian kernel.
    
    Returns
    -------
    np.ndarray
        The dense Gaussian kernel.
    """
    dot_x = np.dot(
        np.conj(x.flatten()), x.flatten()
    )
    dot_y = np.dot(
        np.conj(y.flatten()), y.flatten()
    )
    intermediate = dot_x + dot_y - 2 * z
    return np.exp(-1.0 / sigma ** 2 * np.abs(intermediate) / np.size(x))


def dense_gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    """
    Computes the dense Gaussian kernel as used in the CSK tracker.
    
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
    fft_x = np.fft.fft2(x)
    fft_y = np.fft.fft2(y)
    conj_fft_y = np.conj(fft_y)
    combo = fft_x * conj_fft_y
    fft_response = np.fft.fftshift(np.fft.ifft2(combo))
    return _dgk_sub(x, x, fft_response, sigma)


def csk_train(image: np.ndarray, target: np.ndarray, sigma: float, lmbda: float) -> np.ndarray:
    """
    This function trains the CSK tracker.

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
    kernel = dense_gaussian_kernel(image, image, sigma)
    return np.fft.fft2(target) / (np.fft.fft2(kernel) + lmbda)

def csk_detection(last_alphaf: np.ndarray, prev_window: np.ndarray, window: np.ndarray, sigma: float) -> np.ndarray:
    """
    Calculates new responses for the CSK tracker.

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
    kernel = dense_gaussian_kernel(prev_window, window, sigma)
    fft_kernel = np.fft.fft2(kernel)
    combo = np.fft.ifft2(last_alphaf * fft_kernel)
    return np.real(combo)
