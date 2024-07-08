# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self


class KalmanFilter:
    """A simple Kalman filter for tracking bounding boxes."""

    def __init__(
        self: Self,
        bbox: tuple[int, int, int, int],
        isc: float = 100.0,
        pnc: float = 0.1,
        mnc: float = 1.0,
    ) -> None:
        """
        Create a new Kalman filter object.

        Parameters
        ----------
        bbox : tuple[int, int, int, int]
            The initial bounding box to track.
            Bounding box is in the format '(x1, y1, x2, y2)'.
        isc : float, optional
            The initial state covariance, by default 100.0.
        pnc : float, optional
            The process noise covariance, by default 0.1.
        mnc : float, optional
            The measurement noise covariance, by default 1.0.

        """
        bbox = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
        self._statesize = 8
        self._measurementsize = 4

        # setup the state vector
        self._x = np.zeros((self._statesize, 1))
        self._x[: self._measurementsize] = np.array(bbox).reshape(
            (self._measurementsize, 1),
        )

        # setup the state transition matrix
        self._f = np.eye(self._statesize)
        self._f[: self._measurementsize, self._measurementsize :] = np.eye(
            self._measurementsize,
        )

        # setup the measurement matrix
        self._h = np.zeros((self._measurementsize, self._statesize))
        self._h[: self._measurementsize, : self._measurementsize] = np.eye(
            self._measurementsize,
        )

        # setup the covariance matrices
        self._p = np.eye(self._statesize) * isc
        self._q = np.eye(self._statesize) * pnc
        sub_pnc = pnc / 10.0
        self._q[: self._measurementsize, : self._measurementsize] *= sub_pnc
        self._r = np.eye(self._measurementsize) * mnc

    def predict(self: Self) -> tuple[int, int, int, int]:
        """
        Predict the next bounding box.

        Returns
        -------
        tuple[int, int, int, int]
            The predicted bounding box.
            Bounding box is in the format '(x1, y1, x2, y2)'.

        """
        self._x = self._f @ self._x
        self._p = self._f @ self._p @ self._f.T + self._q
        bbox: np.ndarray = self._x[: self._measurementsize].flatten().astype(int)
        return (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])

    def update(
        self: Self,
        bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """
        Update the filter with a new measurement.

        Parameters
        ----------
        bbox : tuple[int, int, int, int]
            The new bounding box to track.
            Bounding box is in the format '(x1, y1, x2, y2)'.

        Returns
        -------
        tuple[int, int, int, int]
            The updated bounding box.
            Bounding box is in the format '(x1, y1, x2, y2)'.

        """
        bbox = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
        z = np.array(bbox).reshape((self._measurementsize, 1))

        s = self._h @ self._p @ self._h.T + self._r
        k = self._p @ self._h.T @ np.linalg.inv(s)
        self._x += k @ (z - self._h @ self._x)
        self._p = (np.eye(self._statesize) - k @ self._h) @ self._p

        new_bbox: np.ndarray = self._x[: self._measurementsize].flatten().astype(int)
        return (
            new_bbox[0],
            new_bbox[1],
            new_bbox[0] + new_bbox[2],
            new_bbox[1] + new_bbox[3],
        )

    def __call__(
        self: Self,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> tuple[int, int, int, int]:
        """
        Predict the next bounding box or update the filter with a new measurement.

        Parameters
        ----------
        bbox : tuple[int, int, int, int], optional
            The new bounding box to track, by default None.
            If None, the filter will predict the next bounding box.
            Bounding box is in the format '(x1, y1, x2, y2)'.

        Returns
        -------
        tuple[int, int, int, int]
            The predicted or updated bounding box.
            Bounding box is in the format '(x1, y1, x2, y2)'.

        """
        if bbox is None:
            return self.predict()
        return self.update(bbox)
