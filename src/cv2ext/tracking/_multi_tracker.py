# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import atexit
import contextlib
from queue import Empty, Queue
from threading import Event, Thread
from typing import TYPE_CHECKING

from ._interface import AbstractMultiTracker, AbstractTracker
from ._tracker_type import TrackerType

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self


class MultiTracker(AbstractMultiTracker):
    """Handles multiple trackers for tracking multiple objects in a video."""

    def __init__(
        self: Self,
        tracker_type: TrackerType | type[AbstractTracker] = TrackerType.KCF,
        *,
        use_threads: bool | None = None,
    ) -> None:
        """
        Create a new MultiTracker object.

        Parameters
        ----------
        tracker_type : TrackerType | type[AbstractTracker]
            The type of tracker to use for tracking objects.
        use_threads : bool, optional
            Whether to use threading for tracking, by default None.
            If None, the tracker will use threading.

        """
        if use_threads is None:
            use_threads = True

        self._tracker: AbstractMultiTracker = (
            _ThreadedMultiTracker(tracker_type)
            if use_threads
            else _SerialMultiTracker(tracker_type)
        )

    def init(
        self: Self,
        image: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
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
        self._tracker.init(image, bboxes)

    def update(
        self: Self,
        image: np.ndarray,
    ) -> list[tuple[bool, tuple[int, int, int, int]]]:
        """
        Update the trackers with the next frame of the video.

        Parameters
        ----------
        image : np.ndarray
            The next frame of the video.

        Returns
        -------
        list[tuple[bool, tuple[int, int, int, int]]]
            The updated success values and bounding boxes of the targets.
            Each bbox is represented as (x1, y1, x2, y2).

        """
        return self._tracker.update(image)


class _ThreadedMultiTracker(AbstractMultiTracker):
    """Threaded version of MultiTracker."""

    def __init__(self: Self, tracker_type: TrackerType | type[AbstractTracker]) -> None:
        """
        Create a new ThreadedMultiTracker object.

        Parameters
        ----------
        tracker_type : TrackerType | type[AbstractTracker]
            The type of tracker to use for tracking objects.

        """
        self._tracker_type = (
            tracker_type.value
            if isinstance(tracker_type, TrackerType)
            else tracker_type
        )
        self._trackers: list[AbstractTracker] = []
        self._stop_event = Event()
        self._threads: list[Thread] = []
        self._in_queues: list[Queue[np.ndarray]] = []
        self._out_queues: list[Queue[tuple[bool, tuple[int, int, int, int]]]] = []

        atexit.register(self._reset)

    def _reset(self: Self) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join()
        self._stop_event.clear()
        self._trackers = []
        self._in_queues = []
        self._out_queues = []
        self._threads = []

    def _thread_target(
        self: Self,
        initial_image: np.ndarray,
        initial_bbox: tuple[int, int, int, int],
        thread_id: int,
    ) -> None:
        tracker = self._tracker_type()
        tracker.init(initial_image, initial_bbox)
        while not self._stop_event.is_set():
            with contextlib.suppress(Empty):
                image = self._in_queues[thread_id].get(timeout=0.1)
                data = tracker.update(image)
                self._out_queues[thread_id].put_nowait(data)

    def init(
        self: Self,
        image: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
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
        self._reset()
        for idx, bbox in enumerate(bboxes):
            in_queue: Queue[np.ndarray] = Queue()
            out_queue: Queue[tuple[bool, tuple[int, int, int, int]]] = Queue()
            self._in_queues.append(in_queue)
            self._out_queues.append(out_queue)
            thread = Thread(
                target=self._thread_target,
                args=(image, bbox, idx),
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def update(
        self: Self,
        image: np.ndarray,
    ) -> list[tuple[bool, tuple[int, int, int, int]]]:
        """
        Update the trackers with the next frame of the video.

        Parameters
        ----------
        image : np.ndarray
            The next frame of the video.

        Returns
        -------
        list[tuple[bool, tuple[int, int, int, int]]]
            The updated success values and bounding boxes of the targets.
            Each bbox is represented as (x1, y1, x2, y2).

        """
        for in_queue in self._in_queues:
            in_queue.put(image)

        return [out_queue.get() for out_queue in self._out_queues]


class _SerialMultiTracker(AbstractMultiTracker):
    """Serial version of MultiTracker."""

    def __init__(self: Self, tracker_type: TrackerType | type[AbstractTracker]) -> None:
        """
        Create a new SerialMultiTracker object.

        Parameters
        ----------
        tracker_type : TrackerType | type[AbstractTracker]
            The type of tracker to use for tracking objects.

        """
        self._tracker_type = (
            tracker_type.value
            if isinstance(tracker_type, TrackerType)
            else tracker_type
        )
        self._trackers: list[AbstractTracker] = []

    def init(
        self: Self,
        image: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
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
        self._trackers = [self._tracker_type() for _ in bboxes]
        for tracker, bbox in zip(self._trackers, bboxes):
            tracker.init(image, bbox)

    def update(
        self: Self,
        image: np.ndarray,
    ) -> list[tuple[bool, tuple[int, int, int, int]]]:
        """
        Update the trackers with the next frame of the video.

        Parameters
        ----------
        image : np.ndarray
            The next frame of the video.

        Returns
        -------
        list[tuple[bool, tuple[int, int, int, int]]]
            The updated success values and bounding boxes of the targets.
            Each bbox is represented as (x1, y1, x2, y2).

        """
        return [tracker.update(image) for tracker in self._trackers]
