# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing IoU calculation for bounding boxes."""

from __future__ import annotations

import time

from cv2ext import Display, IterableVideo
from cv2ext.detection import AnnealingFramePacker

if __name__ == "__main__":
    video = IterableVideo("data/testvid.mp4")

    packer = AnnealingFramePacker(
        (video.height, video.width),
    )

    with Display("Frame Packing Example") as display:
        for _, frame in video:
            if display.stopped:
                break

            packed, transform = packer.pack(frame, exclude=[])
            display(packed)

            time.sleep(0.013)  # for viewing purposes
