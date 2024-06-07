# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
import logging
from typing import Callable

import cv2
import numpy as np

from cv2ext import _FLAGSOBJ

_log = logging.getLogger(__name__)

try:
    from numba import jit  # type: ignore[import-untyped]
except ImportError:
    jit = None
    if _FLAGSOBJ.USEJIT:
        _log.warning(
            "Numba not installed, but JIT has been enabled. Not using JIT for NCC.",
        )


def _nccjit(
    nccfunc: Callable[[np.ndarray, np.ndarray], float],
) -> Callable[[np.ndarray, np.ndarray], float]:
    if _FLAGSOBJ.USEJIT and jit is not None:
        _log.info("JIT Compiling: NCC")
        nccfunc = jit(nccfunc, nopython=True, parallel=_FLAGSOBJ.PARALLEL)
    return nccfunc


@_nccjit
def _core_ncc(
    image1: np.ndarray,
    image2: np.ndarray,
) -> float:
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    img1std = np.std(image1)
    img2std = np.std(image2)

    if img1std == 0.0 or img2std == 0.0:
        return 0.0

    image1_numerator = image1 - np.mean(image1) / img1std
    image2_numerator = image2 - np.mean(image2) / img2std

    val = float(
        np.sum(image1_numerator * image2_numerator)
        / (np.sqrt(np.sum(image1_numerator**2)) * np.sqrt(np.sum(image2_numerator**2))),
    )

    # clamp to [-1, 1] incase of floating point errors
    return max(min(1.0, val), -1.0)


def ncc(
    image1: np.ndarray,
    image2: np.ndarray,
    size: tuple[int, int] | None = (112, 112),
    *,
    resize: bool | None = True,
) -> float:
    """
    Compute the normalized cross-correlation between two images.

    Parameters
    ----------
    image1 : np.ndarray
        The first image. Can be color or grayscale.
        Converted to grayscale if color.
    image2 : np.ndarray
        The second image. Can be color or grayscale.
        Converted to grayscale if color.
    size : tuple[int, int], optional
        The size to resize the images to, by default (112, 112).
        If None, the images are not resized.
    resize : bool, optional
        If True, the images are resized to the given size before
        computing the correlation, by default False.

    Returns
    -------
    float
        The normalized cross-correlation between the two images.

    Raises
    ------
    ValueError
        If the images are not the same size and not resizing.

    """
    if resize is None:
        resize = False

    if image1.shape != image2.shape and not resize:
        err_msg = f"Images must be the same size if not resizing. Got {image1.shape} and {image2.shape}."
        raise ValueError(err_msg)

    colorchannels = 3
    with contextlib.suppress(IndexError):
        if image1.shape[2] == colorchannels:  # type: ignore[misc]
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    with contextlib.suppress(IndexError):
        if image2.shape[2] == colorchannels:  # type: ignore[misc]
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    if size is not None and resize:
        image1 = cv2.resize(image1, size, interpolation=cv2.INTER_LINEAR)
        image2 = cv2.resize(image2, size, interpolation=cv2.INTER_LINEAR)

    return _core_ncc(image1, image2)
