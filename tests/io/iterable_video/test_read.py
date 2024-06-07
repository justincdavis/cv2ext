# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

from cv2ext import IterableVideo


def test_read():
    video = IterableVideo(Path("data") / "testvid.mp4", use_thread=False)

    got = True
    counter = 0
    while got:
        got, frame = video.read()
        if got:
            counter += 1

    assert counter == len(video)


def test_read_thread():
    video = IterableVideo(Path("data") / "testvid.mp4", use_thread=True)

    got = True
    counter = 0
    while got:
        got, frame = video.read()
        if got:
            counter += 1

    assert counter == len(video)
