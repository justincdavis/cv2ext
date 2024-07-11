# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from cv2ext import _FLAGSOBJ

try:
    from numba import jit  # type: ignore[import-untyped]
except ImportError:
    jit = None

if TYPE_CHECKING:
    from collections.abc import Callable

_log = logging.getLogger(__name__)


def _kalman_predict_kernel_jit(
    predict_func: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ],
) -> Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    if _FLAGSOBJ.USEJIT and jit is not None:
        _log.info("JIT Compiling: bboxes.filters.kalman_predict")
        predict_func = jit(predict_func, nopython=True)
    return predict_func


@_kalman_predict_kernel_jit
def kalman_predict_kernel(
    x: np.ndarray,
    f: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Kernel function for predicting from a Kalman filter.

    Can be JIT compiled with Numba if JIT-compilation is enabled.
    This allows the kernel to be embedded in other JIT-compiled

    Parameters
    ----------
    x : np.ndarray
        The state vector.
    f : np.ndarray
        The state transition matrix.
    p : np.ndarray
        The a priori error covariance matrix.
    q : np.ndarray
        The covariance of process noise matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The predicted state vector and a priori error covariance matrix.

    """
    new_x = f @ x
    new_p = f @ p @ f.T + q
    return new_x, new_p


def _kalman_update_kernel_jit(
    update_func: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ],
) -> Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    if _FLAGSOBJ.USEJIT and jit is not None:
        _log.info("JIT Compiling: bboxes.filters.kalman_update")
        update_func = jit(update_func, nopython=True)
    return update_func


@_kalman_update_kernel_jit
def kalman_update_kernel(
    z: np.ndarray,
    x: np.ndarray,
    h: np.ndarray,
    p: np.ndarray,
    r: np.ndarray,
    i: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Kernel function for updating a Kalman filter.

    Can be JIT compiled with Numba if JIT-compilation is enabled.
    This allows the kernel to be embedded in other JIT-compiled
    functions and methods for faster execution.

    Parameters
    ----------
    z : np.ndarray
        The measurement vector.
    x : np.ndarray
        The state vector.
    h : np.ndarray
        The observation matrix.
    p : np.ndarray
        The a posteriori error covariance matrix.
    r : np.ndarray
        The covariance of observation noise matrix.
    i : np.ndarray
        The identity matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The updated state vector and a posteriori error covariance matrix.

    Notes
    -----
    The kernel function does not perform checking on the dimensions of
    the input matrices.

    """
    y = z - h @ x
    s = r + h @ p @ h.T
    k = p @ h.T @ np.linalg.inv(s)
    ik = i - k @ h
    new_x = x + k @ y
    new_p = (ik @ p) @ (ik.T + k @ r @ k.T)
    return new_x, new_p
