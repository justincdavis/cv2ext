# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from pathlib import Path

from cv2ext import IterableVideo, Display


def test_stress():
    for _ in range(3):
        display = Display("test", show=False)

        video = IterableVideo(Path("data") / "testvid.mp4")

        for fid, frame in video:
            display(frame)


if __name__ == "__main__":
    test_stress()
