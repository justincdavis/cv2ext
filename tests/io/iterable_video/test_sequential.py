# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

from cv2ext import IterableVideo


def test_sequential():
    # get video from dump dir
    video = IterableVideo(Path("data") / "testvid.mp4", use_thread=False)

    prev_id = -1
    for frame_id, _ in video:
        assert prev_id + 1 == frame_id
        prev_id = frame_id


def test_sequential_thread():
    # get video from dump dir
    video = IterableVideo(Path("data") / "testvid.mp4", use_thread=True)

    prev_id = -1
    for frame_id, _ in video:
        assert prev_id + 1 == frame_id
        prev_id = frame_id
