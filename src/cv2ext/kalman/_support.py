# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import numpy as np


def create_system(
    measurement_size: int,
    state_size: int | None = None,
    initial_state_cov: float = 100.0,
    process_noise_cov: float = 0.1,
    measurement_noise_cov: float = 1.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Create matrices for a Kalman filter system.

    Parameters
    ----------
    measurement_size : int
        The size of the measurement vector.
    state_size : int, optional
        The size of the state vector, by default None.
        If None, the state size will be set to the measurement size.
    initial_state_cov : float, optional
        The initial state covariance, by default 100.0.
    process_noise_cov : float, optional
        The process noise covariance, by default 0.1.
    measurement_noise_cov : float, optional
        The measurement noise covariance, by default 1.0.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The system matrices for the Kalman filter.
        The matrices are in the following order: state vector, state transition matrix,
        observation matrix, a priori error covariance matrix, process noise covariance matrix,
        observation noise covariance matrix, and identity matrix.

    """
    if state_size is None:
        state_size = measurement_size

    x = np.zeros((state_size, 1))
    f = np.eye(state_size)
    h = np.zeros((measurement_size, state_size))
    h[:measurement_size, :measurement_size] = np.eye(measurement_size)
    p = np.eye(state_size) * initial_state_cov
    q = np.eye(state_size) * process_noise_cov
    r = np.eye(measurement_size) * measurement_noise_cov
    i = np.eye(state_size)

    return x, f, h, p, q, r, i
