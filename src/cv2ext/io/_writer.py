# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
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
