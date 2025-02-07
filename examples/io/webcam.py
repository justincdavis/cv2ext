# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing how to use the IterableVideo class."""

from __future__ import annotations

import cv2ext

if __name__ == "__main__":
    cv2ext.set_log_level("DEBUG")
    video = cv2ext.IterableVideo(0)

    # iterate over the video
    with cv2ext.Display("webcam", stopkey="q") as display:
        for frame_id, frame in video:
            if display.stopped:
                break

            display(cv2ext.image.draw.text(frame, str(frame_id), (10, 30)))
