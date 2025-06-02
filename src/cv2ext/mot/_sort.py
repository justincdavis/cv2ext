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

    Parameters
    ----------
    iou_threshold : float, optional
        IoU threshold for track-detection association, by default 0.3
    max_age : int, optional
        Maximum number of frames to keep track without detections, by default 30
    min_hits : int, optional
        Minimum number of hits before track is considered valid, by default 3

    """

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
        self._next_track_id = 0
        self._frame_count = 0

    def reset(self) -> None:
        """Reset the tracker."""
        self._tracks = []
        self._next_track_id = 1
        self._frame_count = 0

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

        # predict for all the tracks
        for track in self._tracks:
            track.predict()

        # # assess the valid tracks
        # track_valid: list[bool] = []
        # valid_to_all: list[int] = []
        # for i, track in enumerate(self._tracks):
        #     if track.detection[0] != (-1, -1, -1, -1):
        #         track_valid.append(True)
        #         valid_to_all.append(i)
        #     else:
        #         track_valid.append(False)

        # drop tracks that are invalid
        self._tracks = [t for t in self._tracks if t.detection[0] != (-1, -1, -1, -1)]

        # generate the matches
        matches, unmatched_dets, _ = associate_tracks_to_detections(
            detections,
            # [track.detection for i, track in enumerate(self._tracks) if track_valid[i]],
            [track.detection for track in self._tracks],
            self._iou_threshold,
        )

        # update any of the matched tracks with assigned detctions
        for track_sub_idx, det_id in matches:
            # track_id = valid_to_all[track_sub_idx]
            track_id = track_sub_idx
            self._tracks[track_id].update(detections[det_id])

        # create new tracks for any unmatched detections
        for det_idx in unmatched_dets:
            track = Track(detections[det_idx], self._next_track_id)
            self._tracks.append(track)
            self._next_track_id += 1

        # generate the final detections from the tracks
        detections = []
        to_remove = []
        for i, track in enumerate(self._tracks):
            _, _, tsu, hst = track.state

            # add the tracks detection if in valid state
            if (tsu < 1) and (
                hst >= self._min_hits or self._frame_count <= self._min_hits
            ):
                detections.append(track.detection)

            # if the track has been updated then remove
            if tsu > self._max_age:
                to_remove.append(i)

        # remove stale tracks
        for i in reversed(to_remove):
            self._tracks.pop(i)

        return detections
