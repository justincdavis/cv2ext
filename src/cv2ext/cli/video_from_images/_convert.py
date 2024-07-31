# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import argparse

from cv2ext.video import video_from_images


def video_from_images_cli() -> None:
    """Create a video from a directory of images."""
    parser = argparse.ArgumentParser(
        description="Create a video from a directory of images.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="The directory containing the images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The output video file.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="The frames per second of the output video.",
    )
    args = parser.parse_args()

    video_from_images(
        directory=args.dir,
        output=args.output,
        fps=args.fps,
    )
