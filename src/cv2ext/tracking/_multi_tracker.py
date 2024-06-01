# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self

    from ._interface import TrackerInterface


def _thread_target(
    in_queue: Queue[np.ndarray],
    out_queue: Queue[tuple[int, int, int, int]],
    tracker: TrackerInterface,
) -> None:
    while True:
        image = in_queue.get()
        bbox = tracker.update(image)
        out_queue.put(bbox)


class MultiTracker:
    """Handles multiple trackers for tracking multiple objects in a video."""

    def __init__(
        self: Self,
        tracker_type: type[TrackerInterface],
        *,
        use_threads: bool | None = None,
    ) -> None:
        """
        Create a new MultiTracker object.

        Parameters
        ----------
        tracker_type : type[TrackerInterface]
            The type of tracker to use for tracking objects.
        use_threads : bool, optional
            Whether to use threading for tracking, by default None.
            If None, the tracker will use threading.

        """
        if use_threads is None:
            use_threads = True

        self._tracker_type = tracker_type
        self._trackers: list[TrackerInterface] = []
        self._use_threads = use_threads
        self._threads: list[Thread] = []
        self._in_queues: list[Queue[np.ndarray]] = []
        self._out_queues: list[Queue[tuple[int, int, int, int]]] = []

    def init(
        self: Self, image: np.ndarray, bboxes: list[tuple[int, int, int, int]]
    ) -> None:
        """
        Initialize the trackers with the initial bounding boxes.

        Parameters
        ----------
        image : np.ndarray
            The first frame of the video.
        bboxes : list[tuple[int, int, int, int]]
            The initial bounding boxes of the targets.
            Each bbox is represented as (x1, y1, x2, y2).

        """
        for bbox in bboxes:
            tracker = self._tracker_type()
            tracker.init(image, bbox)
            self._trackers.append(tracker)

        if self._use_threads:
            self._in_queues = [Queue(maxsize=1) for _ in self._trackers]
            self._out_queues = [Queue(maxsize=1) for _ in self._trackers]
            self._threads = [
                Thread(
                    target=_thread_target,
                    args=(in_queue, out_queue, tracker),
                    daemon=True,
                )
                for in_queue, out_queue, tracker in zip(
                    self._in_queues, self._out_queues, self._trackers
                )
            ]

    def update(self: Self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Update the trackers with the next frame of the video.

        Parameters
        ----------
        image : np.ndarray
            The next frame of the video.

        Returns
        -------
        list[tuple[int, int, int, int]]
            The updated bounding boxes of the targets.
            Each bbox is represented as (x1, y1, x2, y2).

        """
        if not self._use_threads:
            bboxes = []
            for tracker in self._trackers:
                bbox = tracker.update(image)
                bboxes.append(bbox)
            return bboxes

        for in_queue in self._in_queues:
            in_queue.put(image)

        return [out_queue.get() for out_queue in self._out_queues]
