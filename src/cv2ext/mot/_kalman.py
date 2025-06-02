# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import numpy as np

from cv2ext._jit import register_jit


@register_jit(nogil=True, fastmath=True, inline="always")
def kalman_predict(
    u: np.ndarray | None,
    x: np.ndarray,
    p: np.ndarray,
    b: np.ndarray,
    f: np.ndarray,
    q: np.ndarray,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict the next state using Kalman filter prediction step.

    Parameters
    ----------
    u : np.ndarray | None
        Control vector
    x : np.ndarray
        State vector
    p : np.ndarray
        State covariance matrix
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
    x_pred = np.dot(f, x)
    if u is not None:
        x_pred += np.dot(b, u)

    # P = FPF` + Q
    p_pred = alpha * np.dot(np.dot(f, p), f.T) + q

    return x_pred, p_pred


@register_jit(nogil=True, fastmath=True, inline="always")
def kalman_update(
    z: np.ndarray,
    r: np.ndarray,
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


@register_jit(nogil=True, fastmath=True, inline="always")
def kalman_init(
    dim_x: int,
    dim_z: int,
    dim_u: int = 0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Initialize state matrices and vectors for Kalman filter.

    Parameters
    ----------
    dim_x : int
        Dimension of state vector
    dim_z : int
        Dimension of measurement vector
    dim_u : int, optional
        Dimension of control vector, by default 0

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        state
        uncertainty covariance
        process uncertainty
        control transition matrix
        state transition matrix
        measurement function
        measurement uncertainty
        identity matrix

    """
    # state
    x = np.zeros((dim_x, 1))

    # uncertainty covariance
    p = np.eye(dim_x)

    # process uncertainty
    q = np.eye(dim_x)

    # control transition matrix
    b = np.zeros((dim_x, dim_u))

    # state transition matrix
    f = np.eye(dim_x)

    # measurement function
    h = np.zeros((dim_z, dim_x))

    # measurement uncertainty
    r = np.eye(dim_z)

    # identity matrix
    identity = np.eye(dim_x)

    return x, p, q, b, f, h, r, identity


class KalmanFilter:
    """Simple Kalman filter class wrapper."""

    __slots__ = ("_b", "_f", "_h", "_identity", "_p", "_q", "_r", "_x")

    def __init__(self, dim_x: int, dim_z: int, dim_u: int = 0) -> None:
        """
        Initialize Kalman filter.

        Parameters
        ----------
        dim_x : int
            Dimension of state vector
        dim_z : int
            Dimension of measurement vector
        dim_u : int, optional
            Dimension of control vector, by default 0

        """
        (
            self._x,
            self._p,
            self._q,
            self._b,
            self._f,
            self._h,
            self._r,
            self._identity,
        ) = kalman_init(dim_x, dim_z, dim_u)

    def x(
        self,
        value: np.ndarray | None = None,
        *,
        no_copy: bool | None = None,
    ) -> np.ndarray:
        """
        Get or set the state vector.

        Parameters
        ----------
        value : np.ndarray | None, optional
            If provided, sets the state vector to this value, by default None.
        no_copy : bool | None, optional
            If True, return the array without copying. If False or None,
            return a copy of the array, by default None. Only used when getting.

        Returns
        -------
        np.ndarray
            The current state vector, copied or not based on no_copy parameter.

        """
        if value is not None:
            self._x = value
        return self._x if no_copy else self._x.copy()

    def p(
        self,
        value: np.ndarray | None = None,
        *,
        no_copy: bool | None = None,
    ) -> np.ndarray:
        """
        Get or set the uncertainty covariance matrix.

        Parameters
        ----------
        value : np.ndarray | None, optional
            If provided, sets the uncertainty covariance matrix to this value, by default None.
        no_copy : bool | None, optional
            If True, return the array without copying. If False or None,
            return a copy of the array, by default None. Only used when getting.

        Returns
        -------
        np.ndarray
            The current uncertainty covariance matrix, copied or not based on no_copy parameter.

        """
        if value is not None:
            self._p = value
        return self._p if no_copy else self._p.copy()

    def q(
        self,
        value: np.ndarray | None = None,
        *,
        no_copy: bool | None = None,
    ) -> np.ndarray:
        """
        Get or set the process uncertainty matrix.

        Parameters
        ----------
        value : np.ndarray | None, optional
            If provided, sets the process uncertainty matrix to this value, by default None.
        no_copy : bool | None, optional
            If True, return the array without copying. If False or None,
            return a copy of the array, by default None. Only used when getting.

        Returns
        -------
        np.ndarray
            The current process uncertainty matrix, copied or not based on no_copy parameter.

        """
        if value is not None:
            self._q = value
        return self._q if no_copy else self._q.copy()

    def b(
        self,
        value: np.ndarray | None = None,
        *,
        no_copy: bool | None = None,
    ) -> np.ndarray:
        """
        Get or set the control transition matrix.

        Parameters
        ----------
        value : np.ndarray | None, optional
            If provided, sets the control transition matrix to this value, by default None.
        no_copy : bool | None, optional
            If True, return the array without copying. If False or None,
            return a copy of the array, by default None. Only used when getting.

        Returns
        -------
        np.ndarray
            The current control transition matrix, copied or not based on no_copy parameter.

        """
        if value is not None:
            self._b = value
        return self._b if no_copy else self._b.copy()

    def f(
        self,
        value: np.ndarray | None = None,
        *,
        no_copy: bool | None = None,
    ) -> np.ndarray:
        """
        Get or set the state transition matrix.

        Parameters
        ----------
        value : np.ndarray | None, optional
            If provided, sets the state transition matrix to this value, by default None.
        no_copy : bool | None, optional
            If True, return the array without copying. If False or None,
            return a copy of the array, by default None. Only used when getting.

        Returns
        -------
        np.ndarray
            The current state transition matrix, copied or not based on no_copy parameter.

        """
        if value is not None:
            self._f = value
        return self._f if no_copy else self._f.copy()

    def h(
        self,
        value: np.ndarray | None = None,
        *,
        no_copy: bool | None = None,
    ) -> np.ndarray:
        """
        Get or set the measurement function matrix.

        Parameters
        ----------
        value : np.ndarray | None, optional
            If provided, sets the measurement function matrix to this value, by default None.
        no_copy : bool | None, optional
            If True, return the array without copying. If False or None,
            return a copy of the array, by default None. Only used when getting.

        Returns
        -------
        np.ndarray
            The current measurement function matrix, copied or not based on no_copy parameter.

        """
        if value is not None:
            self._h = value
        return self._h if no_copy else self._h.copy()

    def r(
        self,
        value: np.ndarray | None = None,
        *,
        no_copy: bool | None = None,
    ) -> np.ndarray:
        """
        Get or set the measurement uncertainty matrix.

        Parameters
        ----------
        value : np.ndarray | None, optional
            If provided, sets the measurement uncertainty matrix to this value, by default None.
        no_copy : bool | None, optional
            If True, return the array without copying. If False or None,
            return a copy of the array, by default None. Only used when getting.

        Returns
        -------
        np.ndarray
            The current measurement uncertainty matrix, copied or not based on no_copy parameter.

        """
        if value is not None:
            self._r = value
        return self._r if no_copy else self._r.copy()

    def predict(
        self,
        u: np.ndarray | None = None,
        alpha: float = 1.0,
        *,
        no_copy: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict the next state using Kalman filter prediction step.

        Parameters
        ----------
        u : np.ndarray | None, optional
            Control vector, by default None
        alpha : float, optional
            Process noise scaling factor, by default 1.0
        no_copy : bool | None, optional
            If True, return the array without copying. If False or None,
            return a copy of the array, by default None. Only used when getting.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Predicted state and covariance

        """
        x_pred, p_pred = kalman_predict(
            u,
            self._x,
            self._p,
            self._b,
            self._f,
            self._q,
            alpha,
        )
        self._x = x_pred
        self._p = p_pred

        if no_copy:
            return self._x, self._p
        return x_pred.copy(), p_pred.copy()

    def update(
        self,
        z: np.ndarray,
        *,
        no_copy: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Update the state using Kalman filter update step.

        Parameters
        ----------
        z : np.ndarray
            Measurement vector
        no_copy : bool | None, optional
            If True, return the array without copying. If False or None,
            return a copy of the array, by default None. Only used when getting.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Updated state and covariance

        """
        x_update, p_update = kalman_update(
            z,
            self._r,
            self._h,
            self._x,
            self._p,
            self._identity,
        )
        self._x = x_update
        self._p = p_update

        if no_copy:
            return self._x, self._p
        return x_update.copy(), p_update.copy()
