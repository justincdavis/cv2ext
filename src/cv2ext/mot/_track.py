# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import numpy as np

from cv2ext._jit import register_jit

from ._kalman import KalmanFilter


@register_jit(nogil=True, inline="always")
def bbox_to_z(bbox: tuple[int, int, int, int]) -> np.ndarray:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2
    cy = bbox[1] + h / 2
    s = w * h
    r = w / float(h)
    return np.array([cx, cy, s, r]).reshape(4, 1)


@register_jit(nogil=True, inline="always")
def x_to_bbox(x: np.ndarray) -> tuple[int, int, int, int]:
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    hw = w / 2.0
    hh = h / 2.0
    return (int(x[0] - hw), int(x[1] - hh), int(x[0] + hw), int(x[1] + hh))


class Track:
    """A single object track for SORT-like tracking algorithms."""

    __slots__ = (
        "_age",
        "_bbox",
        "_class_id",
        "_confidence",
        "_detection",
        "_hit_streak",
        "_hits",
        "_kf",
        "_time_since_update",
        "_track_id",
    )

    def __init__(
        self,
        detection: tuple[tuple[int, int, int, int], float, int],
        track_id: int,
    ) -> None:
        """
        Initialize a new track.

        Parameters
        ----------
        detection : tuple[tuple[int, int, int, int], float, int]
            Initial detection in format ((x1, y1, x2, y2), confidence, class_id)
        track_id : int
            Unique track identifier

        """
        # unpack the detection
        bbox, confidence, class_id = detection

        # setup the kalman filter
        self._kf = KalmanFilter(dim_x=7, dim_z=4)
        self._kf.f(
            np.array(
                [
                    [1, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                ],
            ),
        )
        self._kf.h(
            np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                ],
            ),
        )
        r = self._kf.r(no_copy=True)
        r[2:, 2:] *= 10.0
        p = self._kf.p(no_copy=True)
        p[4:, 4:] *= 1000.0
        p *= 10.0
        q = self._kf.q(no_copy=True)
        q[-1, -1] *= 0.01
        q[4:, 4:] *= 0.01

        # update kf from the bbox
        x = self._kf.x(no_copy=True)
        x[:4] = bbox_to_z(bbox)

        # update det state
        self._detection = (bbox, confidence, class_id)
        self._bbox = bbox
        self._class_id = class_id
        self._confidence = confidence

        # remaining track state
        self._age = 0
        self._hits = 0
        self._time_since_update = 0
        self._hit_streak = 0
        self._track_id = track_id

    @property
    def detection(self) -> tuple[tuple[int, int, int, int], float, int]:
        """
        Get the current detection state.

        Returns
        -------
        tuple[tuple[int, int, int, int], float, int]
            The current detection in format ((x1, y1, x2, y2), confidence, class_id)

        """
        return self._detection

    @property
    def state(self) -> tuple[int, int, int, int]:
        """
        Get the tracking state of the track.

        Returns
        -------
        tuple[int, int, int, int]
            The age, hits, time since update, and hit streak

        """
        return self._age, self._hits, self._time_since_update, self._hit_streak

    def predict(self) -> tuple[tuple[int, int, int, int], float, int]:
        """
        Predict the next state of the track.

        Advances the state vector.

        Returns
        -------
        tuple[tuple[int, int, int, int], float, int]
            The predicted detection in format ((x1, y1, x2, y2), confidence, class_id)

        """
        x_pred, _ = self._kf.predict(no_copy=True)

        # update track state
        if self._time_since_update > 0:
            self._hit_streak = 0
        self._age += 1
        self._time_since_update += 1

        # return bbox
        self._detection = (x_to_bbox(x_pred), self._confidence, self._class_id)
        return self._detection

    def update(self, detection: tuple[tuple[int, int, int, int], float, int]) -> None:
        """
        Update the track with a new bounding box.

        Parameters
        ----------
        detection : tuple[tuple[int, int, int, int], float, int]
            The new detection in format ((x1, y1, x2, y2), confidence, class_id)

        """
        # unpack the detection
        bbox, confidence, class_id = detection

        # update the kf
        x_pred, _ = self._kf.update(bbox_to_z(bbox), no_copy=True)
        self._bbox = x_to_bbox(x_pred)

        # update det state
        self._class_id = class_id
        self._confidence = (self._confidence + confidence) / 2.0

        # update track state
        self._time_since_update = 0
        self._hits += 1
        self._hit_streak += 1

        # update detection state
        self._detection = (self._bbox, self._confidence, self._class_id)
