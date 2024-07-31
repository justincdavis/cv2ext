# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Thread

import cv2
import numpy as np
from typing_extensions import Self

_log = logging.getLogger(__name__)


class IterableVideo:
    def __init__(
        self: Self,
        filename: Path | str,
        channels: int = 3,
        buffersize: int = 8,
        *,
        use_thread: bool | None = None,
    ) -> None:
        """
        Create a new instance of the video.

        Parameters
        ----------
        filename : Path | str
            Path to the video file.
        channels : int
            The number of channels in the video.
            This defaults to 3, and is used to pre-allocate a frame,
            such that the first frame is not empty.
            Use 1 for grayscale videos.
        buffersize : int
            The size of the buffer for the thread.
            This is only used if `use_thread` is True.
            Defaults to 8.
        use_thread : bool
            If True, the frames will be loaded in a separate thread.
            This can help speedup iteration times.
            Defaults to None, in which case the thread is used.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.

        """
        if isinstance(filename, Path):
            filename = str(filename.resolve())
        if not Path(filename).exists():
            err_msg = f"File {filename} does not exist."
            raise FileNotFoundError(err_msg)
        self._cap = cv2.VideoCapture(filename)
        self._frame_num = 0
        self._consumed = 0
        self._got = False
        self._length = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._channels = channels
        self._buffersize = buffersize
        self._frame: np.ndarray = np.zeros(
            (self._height, self._width, self._channels),
            dtype=np.uint8,
        )

        # info for the thread
        if use_thread is None:
            use_thread = True
        self._thread_loads = use_thread
        if self._thread_loads:
            self._thread = Thread(target=self._run, daemon=True)
            self._queue: Queue[tuple[int, bool, np.ndarray]] = Queue(
                maxsize=self._buffersize,
            )
            self._closed = False
            self._thread.start()

    def _run(self: Self) -> None:
        """Read the VideoCapture object."""
        while not self._closed:
            if self._frame_num == self._length:
                break
            got, frame = self._cap.read()
            if not got:
                self._queue.put(
                    (
                        self._frame_num,
                        False,
                        np.zeros(
                            (self._height, self._width, self._channels),
                            dtype=np.uint8,
                        ),
                    ),
                )
                break
            while not self._closed:
                with contextlib.suppress(Full):
                    self._queue.put((self._frame_num, got, frame), timeout=0.1)
                    self._frame_num += 1
                    break
            if self._closed:
                return
        self._closed = True

    @property
    def frame(self: Self) -> np.ndarray:
        """
        Get the current frame.

        When using threading this value will be out of sync from the iterator.

        Returns
        -------
        numpy.ndarray
            The current frame.

        """
        return self._frame

    @property
    def frame_num(self: Self) -> int:
        """
        Get the current frame number.

        When using threading this value will be out of sync from the iterator.

        Returns
        -------
        int
            The current frame number.

        """
        return self._frame_num

    @property
    def success(self: Self) -> bool:
        """
        Get the success of the last frame read.

        When using threading this value will be out of sync from the iterator.

        Returns
        -------
        bool
            True if the frame was successfully loaded.

        """
        return self._got

    @property
    def length(self: Self) -> int:
        """
        Get the length of the video.

        Returns
        -------
        int
            The number of frames in the video.

        """
        return self._length

    @property
    def fps(self: Self) -> float:
        """
        Get the frames per second of the video.

        Returns
        -------
        float
            The frames per second of the video.

        """
        return self._fps

    @property
    def size(self: Self) -> tuple[int, int]:
        """
        Get the size of the video.

        Returns
        -------
        tuple
            The width and height of the video.

        """
        return (self._width, self._height)

    @property
    def channels(self: Self) -> int:
        self._consumed = 0
        """
        Get the number of channels in the video.

        Returns
        -------
        int
            The number of channels in the video.

        """
        return self._channels

    @property
    def width(self: Self) -> int:
        """
        Get the width of the video.

        Returns
        -------
        int
            The width of the video.

        """
        return self._width

    @property
    def height(self: Self) -> int:
        """
        Get the height of the video.

        Returns
        -------
        int
            The height of the video.

        """
        return self._height

    def __len__(self: Self) -> int:
        """
        Get the length of the video.

        Returns
        -------
        int
            The number of frames in the video.

        """
        return self.length

    def __iter__(self: Self) -> Self:
        """
        Get the iterator.

        Returns
        -------
        IterableVideo
            The current instance.

        """
        return self

    def __next__(self: Self) -> tuple[int, np.ndarray]:
        """
        Read the next frame from the video.

        Returns
        -------
        bool
            True if the frame was successfully loaded.
        numpy.ndarray
            The current frame.

        Raises
        ------
        StopIteration
            If the video has ended

        """
        if not self._thread_loads:
            self._got, self._frame = self._cap.read()
            num = self._frame_num
            self._frame_num += 1
            if not self._got:
                self._stop()
                raise StopIteration
            return num, self._frame
        # otherwise use threading
        if self._consumed == self._length:
            self._stop()
            raise StopIteration
        num, got, frame = self._queue.get()
        self._consumed += 1
        if not got:
            self._stop()
            raise StopIteration
        return num, frame

    def _stop(self: Self) -> None:
        """Stop the video."""
        if self._thread_loads:
            self._closed = True
            for _ in range(self._buffersize):
                with contextlib.suppress(Empty):
                    self._queue.get_nowait()
            self._thread.join()
            self._cap.release()
        else:
            self._cap.release()

    def stop(self: Self) -> None:
        """Stop the video."""
        self._stop()

    def read(self: Self) -> tuple[bool, np.ndarray]:
        """
        Read the next frame from the video.

        Returns
        -------
        bool
            True if the frame was successfully loaded.
        numpy.ndarray
            The current frame.

        """
        if not self._thread_loads:
            self._got, self._frame = self._cap.read()
            self._frame_num += 1
            return self._got, self._frame
        # otherwise use threading
        try:
            _, frame = next(self)
        except StopIteration:
            return False, np.zeros(
                (self._height, self._width, self._channels),
                dtype=np.uint8,
            )
        else:
            return True, frame
