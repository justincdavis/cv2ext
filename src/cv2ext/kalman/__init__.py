# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule implementing basic Kalman filter functionality.

Classes
-------
KalmanFilter
    A basic Kalman filter class.


Functions
---------
create_system
    Create matrices for a Kalman filter system.
kalman_predict_kernel
    A kernel function for predicting the state of a Kalman filter.
kalman_update_kernel
    A kernel function for updating the state of a Kalman filter.

"""

from __future__ import annotations

from ._kalman import KalmanFilter
from ._kernels import kalman_predict_kernel, kalman_update_kernel
from ._support import create_system

__all__ = [
    "KalmanFilter",
    "create_system",
    "kalman_predict_kernel",
    "kalman_update_kernel",
]
