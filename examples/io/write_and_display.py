# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing how to write and display the same video."""

from __future__ import annotations

from cv2ext import IterableVideo, VideoWriter, set_log_level

if __name__ == "__main__":
    set_log_level("DEBUG")
    # create an IterableVideo object
    video = IterableVideo("video.mp4")

    with VideoWriter("output.mp4", show=True) as writer:
        # iterate over the video
        for frame_id, frame in video:
            writer.write(frame)
            print(f"Frame {frame_id}: {frame.shape}")
