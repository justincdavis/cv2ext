# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2

from cv2ext.tracking._interface import CVTrackerInterface

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self

_log = logging.getLogger(__name__)


class BoostingTracker(CVTrackerInterface):
    """A class for tracking objects in videos using the Boosting tracker."""

    def __init__(self: Self) -> None:
        """Create a new BoostingTracker object."""
        _log.debug("Creating a legacy tracker (Boosting).")
        try:
            tracker = cv2.legacy_TrackerBoosting.create()  # type: ignore[attr-defined]
        except AttributeError:
            tracker = cv2.TrackerBoosting_create()  # type: ignore[attr-defined]
        super().__init__(tracker)

    def init(self: Self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        """
        Initialize the tracker with an image and bounding box.

        Parameters
        ----------
        image : np.ndarray
            The image to use for tracking.
        bbox : tuple[int, int, int, int]
            The bounding box of the object to track.
            Bounding box is format (x, y, x, y),
            where (x, y) is the top-left/bottom-right corner of the box.

        """
        if len(image.shape) == 2 or image.shape[2] == 1:
            err_msg = "Boosting tracker requires a 3-channel image."
            raise ValueError(err_msg)
        super()._init(image, bbox)

    def update(self: Self, image: np.ndarray) -> tuple[bool, tuple[int, int, int, int]]:
        """
        Update the tracker with a new image.

        Parameters
        ----------
        image : np.ndarray
            The new image to use for tracking.

        Returns
        -------
        bool
            Whether the update was successful.
        tuple[int, int, int, int]
            The bounding box of the tracked object.
            Bounding box is format (x, y, x, y),
            where (x, y) is the top-left/bottom-right corner of the box.

        """
        if len(image.shape) == 2 or image.shape[2] == 1:
            err_msg = "Boosting tracker requires a 3-channel image."
            raise ValueError(err_msg)
        return super()._update(image)
