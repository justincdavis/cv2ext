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

from pathlib import Path

import numpy as np
from cv2ext import IterableVideo, VideoWriter, Fourcc


def test_video_creation():
    video = IterableVideo(Path("data") / "testvid.mp4")
    with VideoWriter("output.mp4") as writer:
        for idx, frame in video:
            writer.write(frame)

    assert Path.exists(Path("output.mp4"))


def test_video_length():
    video = IterableVideo(Path("data") / "testvid.mp4")
    with VideoWriter("output.mp4") as writer:
        for idx, frame in video:
            writer.write(frame)

    video1 = IterableVideo(Path("data") / "testvid.mp4")
    video2 = IterableVideo(Path("output.mp4"))

    assert len(video1) == len(video2)


def test_frame_contents():
    video = IterableVideo(Path("data") / "testvid.mp4")
    with VideoWriter("output.mp4") as writer:
        for idx, frame in video:
            writer.write(frame)

    video1 = IterableVideo(Path("data") / "testvid.mp4")
    video2 = IterableVideo(Path("output.mp4"))

    for (idx1, frame1), (idx2, frame2) in zip(video1, video2):
        assert idx1 == idx2
        assert frame1.shape == frame2.shape
        assert frame1.dtype == frame2.dtype
        assert frame1.size == frame2.size

        # majority of pixel differences after read/write should be small
        assert np.median(frame1 - frame2) < 2.1
