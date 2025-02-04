# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing IoU calculation for bounding boxes."""

from __future__ import annotations

import cv2ext
from cv2ext.detection import AnnealingFramePacker

if __name__ == "__main__":
    video = cv2ext.IterableVideo("data/testvid.mp4")

    packer = AnnealingFramePacker(
        (video.width, video.height),
    )

    with cv2ext.Display("Frame Packing Example", nextkey="n") as display:
        for _, frame in video:
            if display.stopped:
                break

            packed, transform = packer.pack(frame, exclude=[], method="smart")
            display(packed)

            display.wait(timeout=1)
