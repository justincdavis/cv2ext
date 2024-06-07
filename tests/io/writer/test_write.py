# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import platform
from pathlib import Path

import numpy as np
from cv2ext import IterableVideo, VideoWriter


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

    pixel_diff = 3.1 if platform.system() == "Darwin" else 2.1

    for (idx1, frame1), (idx2, frame2) in zip(video1, video2):
        assert idx1 == idx2
        assert frame1.shape == frame2.shape
        assert frame1.dtype == frame2.dtype
        assert frame1.size == frame2.size

        # majority of pixel differences after read/write should be small
        assert np.median(frame1 - frame2) < pixel_diff
