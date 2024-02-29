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

from threading import Condition, Lock, Thread
from typing import TYPE_CHECKING

import cv2  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self


class Display:
    """Class for displaying images using a separate thread."""

    def __init__(self: Self, windowname: str, *, show: bool | None = None) -> None:
        """
        Create a new display.

        Parameters
        ----------
        windowname : str
            The name of the window to display the images in.
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
        # alloocate a dummy image, size does not matter
        self._image: np.ndarray = np.zeros((100, 100, 3), dtype=np.uint8)
        self._consumed = False
        self._frameid = -1  # no frame yet
        self._running = True
        self._cond = Condition()
        self._lock = Lock()
        self._thread = Thread(target=self._display, daemon=True)
        self._thread.start()

    @property
    def frame(self: Self) -> np.ndarray:
        """The current frame being displayed."""
        return self._image

    @frame.setter
    def frame(self: Self, value: np.ndarray) -> None:
        with self._lock:
            self._image = value
            self._frameid += 1
            self._consumed = False
        with self._cond:
            self._cond.notify_all()

    @property
    def frameid(self: Self) -> int:
        """The current frame id."""
        return self._frameid

    def __call__(self: Self, frame: np.ndarray) -> None:
        """
        Update the frame being displayed.

        Parameters
        ----------
        frame : np.ndarray
            The frame to display.

        """
        self.update(frame)

    def __del__(self: Self) -> None:
        self.stop()

    def update(self: Self, frame: np.ndarray) -> None:
        """
        Update the frame being displayed.

        Parameters
        ----------
        frame : np.ndarray
            The frame to display.

        """
        self.frame = frame

    def stop(self: Self) -> None:
        """Stop the display."""
        self._running = False
        with self._cond:
            self._cond.notify()
        cv2.destroyWindow(self._windowname)

    def _display(self: Self) -> None:
        while self._running:
            if not self._consumed:
                with self._lock:
                    image = self._image
                    self._consumed = True
            else:
                with self._cond:
                    self._cond.wait()
                with self._lock:
                    image = self._image
                    self._consumed = True
            if self._show:
                cv2.imshow(self._windowname, image)
                cv2.waitKey(1)
