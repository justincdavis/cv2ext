import os

from cv2ext import IterableVideo

from ._utils import VID_LINK, download_youtube_video


def test_read():
    if not os.path.exists("video.mp4"):
        download_youtube_video(VID_LINK, "video.mp4")

    video = IterableVideo("video.mp4")

    got = True
    counter = 0
    while got:
        got, frame = video.read()
        if got:
            counter += 1

    assert counter == len(video)


def test_read_thread():
    if not os.path.exists("video.mp4"):
        download_youtube_video(VID_LINK, "video.mp4")

    video = IterableVideo("video.mp4", use_thread=True)

    got = True
    counter = 0
    while got:
        got, frame = video.read()
        if got:
            counter += 1

    assert counter == len(video)
