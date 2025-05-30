# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Track representation for multi-object tracking algorithms.

This module provides Track classes that maintain state information
for individual objects being tracked across frames.
"""

from __future__ import annotations

from cv2ext.bboxes import KalmanFilter


class Track:
    """
    A single object track for SORT-based tracking algorithms.

    This class encapsulates the state of a single tracked object including
    its Kalman filter, track ID, and management information.

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        Initial bounding box in format (x1, y1, x2, y2)
    track_id : int
        Unique track identifier
    class_id : int, optional
        Object class identifier, by default -1
    confidence : float, optional
        Initial detection confidence, by default 1.0
    max_age : int, optional
        Maximum number of frames to keep track without detections, by default 30
    min_hits : int, optional
        Minimum number of hits before track is considered valid, by default 3

    """

    __slots__ = (
        "_age",
        "_class_id",
        "_confidence",
        "_history",
        "_hit_streak",
        "_kf",
        "_max_age",
        "_min_hits",
        "_time_since_update",
        "_track_id",
    )

    def __init__(
        self,
        bbox: tuple[int, int, int, int],
        track_id: int,
        class_id: int = -1,
        confidence: float = 1.0,
        max_age: int = 30,
        min_hits: int = 3,
    ) -> None:
        """Initialize a new track."""
        self._track_id = track_id
        self._class_id = class_id
        self._confidence = confidence
        self._max_age = max_age
        self._min_hits = min_hits
        self._kf = KalmanFilter(bbox)
        self._time_since_update = 0
        self._hit_streak = 0
        self._age = 0

        self._history: list[tuple[int, int, int, int]] = []

    @property
    def track_id(self) -> int:
        """Get the track ID."""
        return self._track_id

    @property
    def class_id(self) -> int:
        """Get the class ID."""
        return self._class_id

    @property
    def confidence(self) -> float:
        """Get the confidence."""
        return self._confidence

    @property
    def max_age(self) -> int:
        """Get the maximum age."""
        return self._max_age

    @property
    def min_hits(self) -> int:
        """Get the minimum hits."""
        return self._min_hits

    @property
    def history(self) -> list[tuple[int, int, int, int]]:
        """Get the track history."""
        return self._history

    def predict(self) -> tuple[tuple[int, int, int, int], float, int]:
        """
        Predict the next bounding box position.

        Returns
        -------
        tuple[tuple[int, int, int, int], float, int]
            Predicted detection as (bbox, confidence, class_id)

        """
        predicted_bbox = self._kf.predict()
        self._age += 1

        if self._time_since_update > 0:
            self._hit_streak = 0

        self._time_since_update += 1
        self._history.append(predicted_bbox)

        return (predicted_bbox, self._confidence, self._class_id)

    def update(
        self,
        detection: tuple[tuple[int, int, int, int], float, int],
        *,
        update_class: bool | None = None,
    ) -> tuple[tuple[int, int, int, int], float, int]:
        """
        Update the track with a new detection.

        Parameters
        ----------
        detection : tuple[tuple[int, int, int, int], float, int]
            New detection as full detection tuple
        update_class : bool, optional
            Whether to update class_id from detection, by default None

        Returns
        -------
        tuple[tuple[int, int, int, int], float, int]
            Updated detection as (bbox, confidence, class_id)

        """
        bbox, self._confidence, class_id = detection

        if update_class:
            self._class_id = class_id

        self._time_since_update = 0
        self._hit_streak += 1

        updated_bbox = self._kf.update(bbox)
        self._history.append(updated_bbox)

        return (updated_bbox, self._confidence, self._class_id)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """
        Get the current bounding box estimate.

        Returns
        -------
        tuple[int, int, int, int]
            Current bounding box (x1, y1, x2, y2)

        """
        return self._kf.bbox

    @property
    def detection(self) -> tuple[tuple[int, int, int, int], float, int]:
        """
        Get the current detection as a complete tuple.

        Returns
        -------
        tuple[tuple[int, int, int, int], float, int]
            Current detection as (bbox, confidence, class_id)

        """
        return (self.bbox, self._confidence, self._class_id)

    @property
    def is_confirmed(self) -> bool:
        """
        Check if track is confirmed (has enough hits).

        Returns
        -------
        bool
            True if track has enough hits to be considered confirmed

        """
        return self._hit_streak >= self._min_hits

    @property
    def is_deleted(self) -> bool:
        """
        Check if track should be deleted.

        Returns
        -------
        bool
            True if track has been too long without updates

        """
        return self._time_since_update > self._max_age

    def __repr__(self) -> str:
        return f"Track(id={self.track_id}, class_id={self.class_id}, bbox={self.bbox})"
