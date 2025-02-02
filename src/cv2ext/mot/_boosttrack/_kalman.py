# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

from cv2ext.tracking._interface import AbstractTracker

if TYPE_CHECKING:
    from typing_extensions import Self


CHI2INV95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


def _get_init_state_cov(x_dim: int) -> np.ndarray:
    P: np.ndarray = np.eye(x_dim)
    P[4:, 4:] *= 1000.0
    P *= 10.0
    return P


def _get_R() -> np.ndarray:
    return np.diag([1, 1, 10, 0.01])


def _get_Q(x_dim: int) -> np.ndarray:
    Q: np.ndarray = np.eye(x_dim)
    Q[4:, 4:] *= 0.01
    return Q


class KalmanFilter:
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, h, a, vx, vy, vh, va

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, h, a) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(
        self: Self,
        bbox: np.ndarray,
        ndim: int = 8,
        dt: int = 1,
    ):
        """
        Create a KalmanFilter.

        Parameters
        ----------
        bbox : np.ndarray
            The bounding box in form: [cx, cy, h, r]
            Where cx, cy is the center, h is height, r is aspect ratio
        ndim : int
            The number of dimensions to compute the Kalman over.
            By default 8.
        dt : int
            The time delta.
            By default 1.

        """
        if bbox.ndim == 2:
            bbox = deepcopy(bbox.reshape((-1,)))

        self._dt: int = dt
        self._ndim: int = ndim
        self._motion_mat: np.ndarray = np.eye(ndim, ndim)
        for i in range(4 - (ndim % 2)):
            self._motion_mat[i, i + 4] = dt
        self._update_mat: np.ndarray = np.eye(4, ndim)
        self._x: np.ndarray = np.zeros((ndim,))
        self._x[:4] = bbox[:]
        self._covariance: np.ndarray = _get_init_state_cov(self._ndim)

    def predict(
        self: Self,
        mean: np.ndarray | None = None,
        covariance: np.ndarray | None = None,
    ):
        """
        Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        tuple[ndarray, ndarray]
            The mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        # if mean and covariance are None, use previous data
        if mean is None and covariance is None:
            w_mean = self._x
            w_covariance = self._covariance
        else:
            w_mean = mean
            w_covariance = covariance

        # compute the update
        motion_cov = _get_Q(self._ndim)
        n_mean: np.ndarray = np.dot(self._motion_mat, w_mean)  # type: ignore[assignment]
        n_covariance = (
            np.linalg.multi_dot((self._motion_mat, w_covariance, self._motion_mat.T))
            + motion_cov
        )

        # if the mean was None, we should update the filter
        if mean is None and covariance is None:
            self._x = n_mean
            self._covariance = n_covariance

        return n_mean, n_covariance

    def project(self: Self):
        """
        Project state distribution to measurement space.

        Returns
        -------
        tuple[ndarray, ndarray]
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        innovation_cov = _get_R()

        mean = np.dot(self._update_mat, self._x)
        covariance = np.linalg.multi_dot(
            (self._update_mat, self._covariance, self._update_mat.T),
        )

        return mean, covariance + innovation_cov

    def update(self, bbox: np.ndarray):
        """
        Run Kalman filter correction step.

        Parameters
        ----------
        bbox : np.ndarray
            The bbox

        Returns
        -------
        tuple[ndarray, ndarray]
            Returns the measurement-corrected state distribution.

        """
        if bbox.ndim == 2:
            bbox = deepcopy(bbox.reshape((-1,)))

        projected_mean, projected_cov = self.project()

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov,
            lower=True,
            check_finite=False,
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(self._covariance, self._update_mat.T).T,
            check_finite=False,
        ).T

        innovation = bbox - projected_mean

        self._x = self._x + np.dot(innovation, kalman_gain.T)
        self._covariance = self._covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T),
        )

        return self._x, self._covariance


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,h,r] where x,y is the centre of the box and h is the height and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0

    r = w / float(h + 1e-6)

    return np.array([x, y, h, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,h,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    h = x[2]
    r = x[3]
    w = 0 if r <= 0 else r * h

    if score is None:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0],
        ).reshape((1, 4))
    return np.array(
        [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score],
    ).reshape((1, 5))


class KalmanBoxTracker(AbstractTracker):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox, emb: Optional[np.ndarray] = None):
        """
        Initialises a tracker using initial bounding box.
        """
        self.bbox_to_z_func = convert_bbox_to_z
        self.x_to_bbox_func = convert_x_to_bbox

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.kf = KalmanFilter(self.bbox_to_z_func(bbox))
        self.emb = emb
        self.hit_streak = 0
        self.age = 0

    def get_confidence(self, coef: float = 0.9) -> float:
        n = 7

        if self.age < n:
            return coef ** (n - self.age)
        return coef ** (self.time_since_update - 1)

    def update(self, bbox: np.ndarray, score: float = 0):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(self.bbox_to_z_func(bbox), score)

    def camera_update(self, transform: np.ndarray):
        x1, y1, x2, y2 = self.get_state()[0]
        x1_, y1_, _ = transform @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = transform @ np.array([x2, y2, 1]).T
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w / 2, y1_ + h / 2
        self.kf.x[:4] = [cx, cy, h, w / h]

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.get_state()

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.x_to_bbox_func(self.kf.x)

    def update_emb(self, emb, alpha=0.9):
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb
