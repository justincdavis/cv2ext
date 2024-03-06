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
import logging
from queue import Empty, Full, Queue
from threading import Thread
from typing import TYPE_CHECKING

import cv2
import numpy as np

from cv2ext import _DELOBJ

if TYPE_CHECKING:
    from typing_extensions import Self


_log = logging.getLogger(__name__)


class Display:
    """Class for displaying images using a separate thread."""

    def __init__(
        self: Self,
        windowname: str,
        stopkey: str = "q",
        buffersize: int = 1,
        *,
        show: bool | None = None,
    ) -> None:
        """
        Create a new display.

        Parameters
        ----------
        windowname : str
            The name of the window to display the images in.
        stopkey : str
            The key to press to stop the display.
            By default, this is "q".
        buffersize : int
            The size of the buffer for the display.
            By default, this is 1.
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
        self._stopkey = stopkey
        self._buffersize = buffersize
        self._show = show

        # allocate runtime variables
        self._image: np.ndarray = np.zeros((100, 100, 3), dtype=np.uint8)
        self._frameid = -1  # no frame yet
        self._running = True
        self._queue: Queue[np.ndarray] = Queue(maxsize=self._buffersize)

        # thread allocation
        _DELOBJ.logwindow(self._windowname)
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

    @property
    def stopped(self: Self) -> bool:
        """Whether the display is stopped."""
        return not self._running

    @property
    def is_alive(self: Self) -> bool:
        """Whether the display is running."""
        return self._thread.is_alive()

    def __call__(self: Self, frame: np.ndarray) -> None:
        """
        Update the frame being displayed.

        Parameters
        ----------
        frame : np.ndarray
            The frame to display.

        Raises
        ------
        queue.Full
            If timeout is provided, and the queue if full at the end of timeout.

        """
        self.update(frame)

    def __del__(self: Self) -> None:
        self._stop()

    def _display(self: Self) -> None:
        if self._show:
            cv2.namedWindow(self._windowname, cv2.WINDOW_AUTOSIZE)
            # cv2.startWindowThread()
        while self._running:
            _log.debug(f"Display {self._windowname} thread starting new loop")
            with contextlib.suppress(Empty):
                image = self._queue.get(timeout=0.1)
                if self._show:
                    cv2.imshow(self._windowname, image)
                    if cv2.waitKey(1) & 0xFF == ord(self._stopkey):
                        self._running = False
                        continue
        _log.debug(f"Display {self._windowname} thread stopped")
        # if self._show:
        #     _log.debug(f"Destroying window {self._windowname}")
        #     cv2.destroyWindow(self._windowname)
        #     cv2.waitKey(1)

    def _stop(self: Self) -> None:
        """Stop the display."""
        self._running = False
        while self._thread.is_alive():
            _log.debug(f"Attempting join for display thread {self._windowname}")
            self._thread.join(timeout=0.01)

    def stop(self: Self) -> None:
        """Stop the display."""
        self._stop()

    def update(self: Self, frame: np.ndarray) -> None:
        """
        Update the frame being displayed.

        Parameters
        ----------
        frame : np.ndarray
            The frame to display.

        Raises
        ------
        queue.Full
            If timeout is provided, and the queue if full at the end of timeout.

        """
        self._image = frame
        self._frameid += 1
        with contextlib.suppress(Full):
            self._queue.put_nowait(frame)
