# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing how to use the IterableVideo class."""

from __future__ import annotations

from cv2ext import Display, IterableVideo, set_log_level

if __name__ == "__main__":
    set_log_level("DEBUG")
    # create an IterableVideo object
    video = IterableVideo("video.mp4")
    display = Display("example")

    # iterate over the video
    for frame_id, frame in video:
        display(frame)
        print(f"Frame {frame_id}: {frame.shape}")

    display.stop()

    # OR
    # use with context manager

    video = IterableVideo("video.mp4")
    with Display("example") as display:
        for frame_id, frame in video:
            display(frame)
            print(f"Frame {frame_id}: {frame.shape}")
