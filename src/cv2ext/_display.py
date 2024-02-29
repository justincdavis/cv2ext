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

import contextlib
from queue import Empty, Queue
from threading import Condition, Thread
from typing import TYPE_CHECKING

import cv2  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self


class Display:
    """Class for displaying images using a separate thread."""

    def __init__(
        self: Self,
        windowname: str,
        buffersize: int = 8,
        *,
        show: bool | None = None,
    ) -> None:
        """
        Create a new display.

        Parameters
        ----------
        windowname : str
            The name of the window to display the images in.
        buffersize : int
            The size of the buffer for the thread.
            By default, this is 8.
        show : bool | None
            If True, the window will be shown.
            If False, the window will not be shown.
            If None, the window will be shown.
            Primarily used for debugging purposes, with show being
            False, the display class does not do anything except
            store the current image.

        """
        if show is None:
            show = True
        self._windowname = windowname
        self._show = show
        self._buffersize = buffersize
        # alloocate a dummy image, size does not matter
        self._image: np.ndarray = np.zeros((100, 100, 3), dtype=np.uint8)
        self._frameid = -1  # no frame yet
        self._running = True
        self._queue: Queue[np.ndarray] = Queue(maxsize=self._buffersize)
        self._stopcond = Condition()
        self._thread = Thread(target=self._display, daemon=True)
        self._thread.start()

    @property
    def frame(self: Self) -> np.ndarray:
        """The most recent frame."""
        return self._image

    @property
    def frameid(self: Self) -> int:
        """The current frame id."""
        return self._frameid

    def __call__(self: Self, frame: np.ndarray, timeout: float | None = None) -> None:
        """
        Update the frame being displayed.

        Parameters
        ----------
        frame : np.ndarray
            The frame to display.
        timeout : float | None
            The timeout for the queue.
            Optional, defaults to None.
            If None, no timeout is used.

        Raises
        ------
        queue.Full
            If timeout is provided, and the queue if full at the end of timeout.

        """
        self.update(frame, timeout)

    def __del__(self: Self) -> None:
        self._stop()

    def update(self: Self, frame: np.ndarray, timeout: float | None = None) -> None:
        """
        Update the frame being displayed.

        Parameters
        ----------
        frame : np.ndarray
            The frame to display.
        timeout : float | None
            The timeout for the queue.
            Optional, defaults to None.
            If None, no timeout is used.

        Raises
        ------
        queue.Full
            If timeout is provided, and the queue if full at the end of timeout.

        """
        self._image = frame
        self._frameid += 1
        self._queue.put(frame, timeout=timeout)

    def _stop(self: Self) -> None:
        """Stop the display."""
        self._running = False
        with self._stopcond:
            self._stopcond.wait()
        cv2.destroyWindow(self._windowname)

    def _display(self: Self) -> None:
        while self._running:
            with contextlib.suppress(Empty):
                image = self._queue.get(timeout=0.1)
                if self._show:
                    cv2.imshow(self._windowname, image)
                    cv2.waitKey(1)
        with self._stopcond:
            self._stopcond.notify_all()
