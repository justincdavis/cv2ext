# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._kernels import kalman_predict_kernel, kalman_update_kernel
from ._support import create_system

if TYPE_CHECKING:
    from typing_extensions import Self


class KalmanFilter:
    """A simple Kalman filter."""

    def __init__(
        self: Self,
        measurement_size: int,
        state_size: int | None = None,
        isc: float = 100.0,
        pnc: float = 0.1,
        mnc: float = 1.0,
    ) -> None:
        """
        Create a new Kalman filter object.

        Parameters
        ----------
        measurement_size : int
            The size of the measurement vector.
        state_size : int, optional
            The size of the state vector, by default None.
        isc : float, optional
            The initial state covariance, by default 100.0.
        pnc : float, optional
            The process noise covariance, by default 0.1.
        mnc : float, optional
            The measurement noise covariance, by default 1.0.

        """
        if state_size is None:
            state_size = measurement_size
        self._measurementsize = measurement_size
        self._statesize = state_size

        self._x, self._f, self._h, self._p, self._q, self._r, self._i = create_system(
            self._measurementsize,
            self._statesize,
            isc,
            pnc,
            mnc,
        )

    def predict(self: Self, dt: float | None = None) -> np.ndarray:
        """
        Predict the next datapoint.

        Parameters
        ----------
        dt : float, optional
            The time delta for the prediction, by default None.

        Returns
        -------
        np.ndarray
            The predicted data.

        """
        if dt is not None:
            self._f = np.eye(self._statesize) * dt
        new_x, new_p = kalman_predict_kernel(self._x, self._f, self._p, self._q)
        self._x = new_x
        self._p = new_p
        return self._x

    def update(
        self: Self,
        measurement: np.ndarray,
    ) -> None:
        """
        Update the filter with a new measurement.

        Parameters
        ----------
        measurement : np.ndarray
            The new measurement.

        """
        new_x, new_p = kalman_update_kernel(
            measurement,
            self._x,
            self._h,
            self._p,
            self._r,
            self._i,
        )
        self._x = new_x
        self._p = new_p

    def __call__(
        self: Self,
        measurement: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Predict the next data point and update the filter with a new measurement.

        Parameters
        ----------
        measurement : np.ndarray, optional
            The new measurement, by default None.

        Returns
        -------
        np.ndarray
            The predicted data.

        """
        if measurement is None:
            return self.predict()
        new_data = self.predict()
        self.update(measurement)
        return new_data
