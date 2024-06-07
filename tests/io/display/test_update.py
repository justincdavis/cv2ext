# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

from cv2ext import IterableVideo, Display
import numpy as np


def test_update():
    video = IterableVideo(Path("data") / "testvid.mp4")
    display = Display("test", show=False)

    for _, frame in video:
        display.update(frame)

        assert np.all(frame == display.frame)


def test_id():
    video = IterableVideo(Path("data") / "testvid.mp4")
    display = Display("test", show=False)

    for frame_id, frame in video:
        display.update(frame)

        assert np.all(frame == display.frame)
        assert frame_id == display.frameid


def test_call():
    video = IterableVideo(Path("data") / "testvid.mp4")
    display1 = Display("test1", show=False)
    display2 = Display("test2", show=False)

    for _, frame in video:
        display1.update(frame)
        display2(frame)

        assert np.all(frame == display1.frame)
        assert np.all(frame == display2.frame)
        assert np.all(display1.frame == display2.frame)
