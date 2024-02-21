import os

import numpy as np
from cv2tools import IterableVideo

from ._utils import VID_LINK, download_youtube_video


def test_video_same():
    if not os.path.exists("video.mp4"):
        download_youtube_video(VID_LINK, "video.mp4")

    # get video from dump dir
    video = IterableVideo("video.mp4")
    video_thread = IterableVideo("video.mp4", use_thread=True)

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
