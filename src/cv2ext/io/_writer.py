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

from typing import TYPE_CHECKING

import cv2

from ._fourcc import Fourcc

if TYPE_CHECKING:
    from types import TracebackType

    import numpy as np
    from typing_extensions import Self


class VideoWriter:
    def __init__(
        self: Self,
        filename: str,
        fourcc: Fourcc = Fourcc.mp4v,
        fps: float = 30.0,
        frame_size: tuple[int, int] | None = None,
    ) -> None:
        """
        Create a new video writer.

        Parameters
        ----------
        filename : str
            The name of the file to write to.
        fourcc : Fourcc
            The fourcc codec to use.
            Defaults to MP4V.
        fps : float
            The frames per second of the video.
            Defaults to 30.0.
        frame_size : tuple[int, int] | None
            The size of the frames.
            If None, the first frame written will determine the size.
            Defaults to None.

        """
        self._filename = filename
        self._fourcc = fourcc
        self._fps = fps
        self._frame_size = frame_size

        # allocate writer once first frame is written
        self._writer: cv2.VideoWriter | None = None

    def __enter__(self: Self) -> Self:
        return self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    def write(self: Self, frame: np.ndarray) -> None:
        """
        Write a new frame to the video.

        Parameters
        ----------
        frame : np.ndarray
            The frame to write.

        """
        if self._writer is None:
            if self._frame_size is None:
                width, height = frame.shape[:2][::-1]
                self._frame_size = (width, height)
            self._writer = cv2.VideoWriter(
                self._filename,
                self._fourcc.value,
                self._fps,
                self._frame_size,
            )
        self._writer.write(frame)

    def release(self: Self) -> None:
        """Release the video writer."""
        if self._writer is None:
            return
        self._writer.release()
