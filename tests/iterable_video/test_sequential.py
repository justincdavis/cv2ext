import os

from cv2tools import IterableVideo

from ._utils import VID_LINK, download_youtube_video


def test_sequential():
    if not os.path.exists("video.mp4"):
        download_youtube_video(VID_LINK, "video.mp4")

    # get video from dump dir
    video = IterableVideo("video.mp4")

    prev_id = -1
    for frame_id, _ in video:
        assert prev_id + 1 == frame_id
        prev_id = frame_id

def test_sequential_thread():
    if not os.path.exists("video.mp4"):
        download_youtube_video(VID_LINK, "video.mp4")

    # get video from dump dir
    video = IterableVideo("video.mp4", use_thread=True)

    prev_id = -1
    for frame_id, _ in video:
        assert prev_id + 1 == frame_id
        prev_id = frame_id
