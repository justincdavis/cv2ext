# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import numpy as np

from cv2ext._jit import register_jit


@register_jit(nogil=True, fastmath=True, inline="always")
def kalman_predict(
    x: np.ndarray,
    p: np.ndarray,
    u: np.ndarray,
    b: np.ndarray,
    f: np.ndarray,
    q: np.ndarray,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict the next state using Kalman filter prediction step.

    Parameters
    ----------
    x : np.ndarray
        State vector
    p : np.ndarray
        State covariance matrix
    u : np.ndarray
        Control vector
    b : np.ndarray
        Control transition matrix
    f : np.ndarray
        State tranistion matrix
    q : np.ndarray
        Process noise matrix
    alpha : float, optional
        Process noise scaling factor, by default 1.0

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Predicted state and covariance

    """
    # x = Fx + Bu
    x_pred = np.dot(f, x) + np.dot(b, u)

    # P = FPF` + Q
    p_pred = alpha * np.dot(np.dot(f, p), f.T) + q

    return x_pred, p_pred


@register_jit(nogil=True, fastmath=True, inline="always")
def kalman_update(
    z: np.ndarray,
    r : np.ndarray,
    h: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
    identity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Update the state using Kalman filter update step.

    Parameters
    ----------
    z : np.ndarray
        Measurement vector
    r : np.ndarray
        Measurement noise matrix
    h : np.ndarray
        Observation matrix
    x : np.ndarray
        State vector
    p : np.ndarray
        State covariance matrix
    identity : np.ndarray
        Identity matrix

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Updated state and covariance

    """
    # y = z - Hx
    y = z - np.dot(h, x)

    # set subexpression
    pht = np.dot(p, h.T)

    # S = HPH` + R
    s = np.dot(h, pht) + r
    si = np.linalg.inv(s)

    # K = PH`inv(S)
    k = np.dot(pht, si)

    # x = x + Ky
    x_update = x + np.dot(k, y)

    # P = (I-KH)P(I-KH)' + KRK'
    i_kh = identity - np.dot(k, h)
    p_update = np.dot(np.dot(i_kh, p), i_kh.T) + np.dot(np.dot(k, r), k.T)

    return x_update, p_update
