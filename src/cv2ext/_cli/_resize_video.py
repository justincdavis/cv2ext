# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

import argparse

import cv2
from tqdm import tqdm

from cv2ext import IterableVideo


def resize_video() -> None:
    parser = argparse.ArgumentParser(description="Resize a video file.")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="The video file to resize.",
    )
    parser.add_argument("--output", type=str, required=True, help="The output file.")
    parser.add_argument(
        "--size",
        type=str,
        required=True,
        help="The size to resize the video to. Example: [640, 480]",
    )
    parser.add_argument(
        "--fast",
        type=bool,
        default=False,
        required=False,
        help="Use a faster resize method.",
    )
    args = parser.parse_args()

    # parse size
    size = tuple(map(int, args.size.strip("[]").split(",")))

    # open video
    video = IterableVideo(args.video, use_thread=True)
    extension = args.output.split(".")[-1]
    fourcc = None
    if extension == "mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    elif extension == "avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore[attr-defined]
    if fourcc is None:
        err_msg = f"Unsupported file extension: {extension}"
        raise ValueError(err_msg)
    writer = cv2.VideoWriter(args.output, fourcc, video.fps, size)

    for _, frame in tqdm(video):
        if args.fast:
            resized_frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
        else:
            resized_frame = cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)
        writer.write(resized_frame)

    writer.release()
