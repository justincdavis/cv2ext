# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

import numpy as np
from cv2ext import IterableVideo


def test_video_same():
    # get video from dump dir
    video = IterableVideo(Path("data") / "testvid.mp4", use_thread=False)
    video_thread = IterableVideo(Path("data") / "testvid.mp4", use_thread=True)

    assert len(video) == len(video_thread)
    assert video.fps == video_thread.fps
    assert video.size == video_thread.size
    assert video.length == video_thread.length
    assert video.width == video_thread.width
    assert video.height == video_thread.height

    for (frame_id, frame), (frame_id_thread, frame_thread) in zip(video, video_thread):
        assert frame_id == frame_id_thread
        assert frame.shape == frame_thread.shape
        assert frame.dtype == frame_thread.dtype
        assert frame.size == frame_thread.size

        assert np.all(frame == frame_thread)
