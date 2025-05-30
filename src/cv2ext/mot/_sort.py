# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from ._core import associate_tracks_to_detections
from ._track import Track
from ._tracker import Tracker


class SORT(Tracker):
    """
    SORT (Simple Online and Realtime Tracking) tracker.

    This class implements the SORT algorithm for multi-object tracking
    using Kalman filters for motion prediction and Hungarian algorithm
    for data association.

    Parameters
    ----------
    iou_threshold : float, optional
        IoU threshold for track-detection association, by default 0.3
    max_age : int, optional
        Maximum number of frames to keep track without detections, by default 30
    min_hits : int, optional
        Minimum number of hits before track is considered valid, by default 3

    Example
    -------
    >>> tracker = SORT()
    >>> detections = [((100, 50, 200, 150), 0.9, 1), ((300, 100, 400, 200), 0.8, 1)]
    >>> tracks = tracker.update(detections)
    >>> for track in tracks:
    ...     print(f"Track {track.track_id}: {track.bbox}")

    """

    __slots__ = (
        "_frame_count",
        "_iou_threshold",
        "_max_age",
        "_min_hits",
        "_next_track_id",
        "_tracks",
    )

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
    ) -> None:
        """Initialize SORT tracker."""
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._min_hits = min_hits
        self._tracks: list[Track] = []
        self._next_track_id = 1
        self._frame_count = 0

    @property
    def tracks(self, *, copy: bool | None = None) -> list[Track]:
        """
        Get all tracks.

        Parameters
        ----------
        copy : bool, optional
            Whether to return a copy of the tracks, by default None

        Returns
        -------
        list[Track]
            All tracks

        """
        return self._tracks.copy() if copy else self._tracks

    def reset(self) -> None:
        """Reset the tracker."""
        self._tracks = []
        self._next_track_id = 1
        self._frame_count = 0

    def predict(self) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """
        Predict the next state of all tracks.

        Returns
        -------
        list[tuple[tuple[int, int, int, int], float, int]]
            List of predicted detections

        """
        return [track.predict() for track in self._tracks]

    def update(
        self,
        detections: list[tuple[tuple[int, int, int, int], float, int]],
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """
        Update the tracker with new detections.

        Parameters
        ----------
        detections : list[tuple[tuple[int, int, int, int], float, int]]
            List of detections in format [(bbox, confidence, class_id), ...]
            where bbox is (x1, y1, x2, y2)

        Returns
        -------
        list[tuple[tuple[int, int, int, int], float, int]]
            List of confirmed detections

        """
        self._frame_count += 1

        self.predict()

        track_bboxes = [track.bbox for track in self._tracks]

        matched_tracks, matched_detections, _ = associate_tracks_to_detections(
            track_bboxes,
            detections,
            self._iou_threshold,
        )

        for track_idx, det_idx in zip(matched_tracks, matched_detections):
            self._tracks[track_idx].update(detections[det_idx])

        unmatched_detection_indices = set(range(len(detections))) - set(
            matched_detections,
        )
        for det_idx in unmatched_detection_indices:
            bbox, confidence, class_id = detections[det_idx]
            new_track = Track(
                bbox=bbox,
                track_id=self._next_track_id,
                class_id=class_id,
                confidence=confidence,
                max_age=self._max_age,
                min_hits=self._min_hits,
            )
            self._tracks.append(new_track)
            self._next_track_id += 1

        self._tracks = [track for track in self._tracks if not track.is_deleted]

        return [track.detection for track in self._tracks if track.is_confirmed]
