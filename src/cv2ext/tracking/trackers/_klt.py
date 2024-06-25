# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from cv2ext.bboxes import constrain
from cv2ext.tracking._interface import AbstractTracker

if TYPE_CHECKING:
    from typing_extensions import Self


class KLTTracker(AbstractTracker):
    """Class for tracking objects with the KLT algorithm."""

    def __init__(
        self: Self,
        num_features: int = 500,
        window_size: tuple[int, int] = (15, 15),
        max_level: int = 2,
        criteria: tuple[int, int, float] = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            10,
            0.03,
        ),
    ) -> None:
        """
        Create a new KLTTracker object.

        Parameters
        ----------
        num_features : int
            The number of features to track.
            By default, this is set to 500.
        window_size : tuple[int, int]
            The size of the window used for tracking.
            By default, this is set to (15, 15).
        max_level : int
            The maximum pyramid level for tracking.
            By default, this is set to 2.
        criteria : tuple[int, int, float]
            The criteria used for tracking.
            By default, this is set to (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03).

        """
        self._window_size = window_size
        self._max_level = max_level
        self._criteria = criteria
        self._lk_params = {
            "winSize": self._window_size,
            "maxLevel": self._max_level,
            "criteria": self._criteria,
        }
        self._orb: cv2.ORB = cv2.ORB_create(nfeatures=num_features)  # type: ignore[attr-defined]

        # state storage
        self._prev_frame: np.ndarray = np.zeros((1, 1))
        self._prev_roi: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._prev_keypoints: np.ndarray = np.zeros((1, 1))

    def _detect_keypoints(self: Self, image: np.ndarray) -> np.ndarray:
        keypoints = self._orb.detect(image, None)
        return np.asarray([kp.pt for kp in keypoints], dtype=np.float32)

    def init(self: Self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        """
        Initialize the tracker.

        Parameters
        ----------
        image : np.ndarray
            The image to track the object in.
        bbox : tuple[int, int, int, int]
            The bounding box of the object to track.
            In format: (x1, y1, x2, y2)

        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]
        bbox = constrain(bbox, (width, height))
        self._prev_frame = image
        self._prev_keypoints = self._detect_keypoints(image)

    def update(self: Self, image: np.ndarray) -> tuple[bool, tuple[int, int, int, int]]:
        """
        Update the tracker.

        Parameters
        ----------
        image : np.ndarray
            The image to track the object in.

        Returns
        -------
        bool
            Whether the update was successful.
        tuple[int, int, int, int]
            The bounding box of the object.
            In format: (x1, y1, x2, y2)

        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        data = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
            self._prev_frame,
            image,
            self._prev_keypoints,
            None,
            **self._lk_params,
        )
        current_keypoints = data[0]
        status: np.ndarray = data[1]
        mask = status.ravel() == 1
        current_keypoints = current_keypoints[mask]
        x, y, w, h = cv2.boundingRect(current_keypoints)
        bbox = (x, y, x + w, y + h)
        self._prev_frame = image
        self._prev_keypoints = current_keypoints
        return True, bbox
