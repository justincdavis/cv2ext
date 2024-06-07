# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from ._interface import AbstractTracker
from ._tracker_type import TrackerType

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self


class Tracker(AbstractTracker):
    """Handles tracking an object in a video."""

    def __init__(self: Self, tracker: TrackerType = TrackerType.KCF) -> None:
        """
        Create a new Tracker object.

        Parameters
        ----------
        tracker : TrackerType, optional
            The type of tracker to use for tracking objects, by default TrackerType.KCF.

        """
        self._tracker = tracker.value()

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
        self._tracker.init(image, bbox)

    def update(self: Self, image: np.ndarray) -> tuple[bool, tuple[int, int, int, int]]:
        """
        Update the tracker with a new image.

        Parameters
        ----------
        image : np.ndarray
            The new image to use for tracking.

        Returns
        -------
        tuple[bool, tuple[int, int, int, int]]
            A tuple containing a boolean indicating if the update was successful
            and a tuple containing the bounding box of the tracked object.

        """
        data: tuple[bool, tuple[int, int, int, int]] = self._tracker.update(image)
        return data[0], data[1]
