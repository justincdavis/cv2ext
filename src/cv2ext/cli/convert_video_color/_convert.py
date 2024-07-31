# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from cv2ext.io import IterableVideo, VideoWriter


def convert_video_color_cli() -> None:
    """
    Convert the color of a video from the command line.

    Raises
    ------
    FileNotFoundError
        If the input file is not found.
    IsADirectoryError
        If the input path is not a file.
    ValueError
        If no color space is selected
    ValueError
        More than one color space is selected
    ValueError
        An unknown color space is found
    ValueError
        No conversion needed between the two color spaces

    """
    parser = argparse.ArgumentParser("Convert the color of a video.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="The path to the input video.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="The path to the output video.",
    )
    parser.add_argument(
        "--color",
        "-c",
        type=str,
        choices=["bgr", "gray", "rgb"],
        default="bgr",
        help="The color space to convert the video to.",
    )
    parser.add_argument(
        "--bgr",
        action="store_true",
        help="Convert the video to BGR color space.",
    )
    parser.add_argument(
        "--gray",
        action="store_true",
        help="Convert the video to grayscale.",
    )
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Convert the video to RGB color space.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        err_msg = f"Input file not found: {input_path}"
        raise FileNotFoundError(err_msg)
    if not input_path.is_file():
        err_msg = f"Input path is not a file: {input_path}"
        raise IsADirectoryError(err_msg)
    output_path = Path(args.output)
    start_color = args.color
    use_bgr = args.bgr
    use_gray = args.gray
    use_rgb = args.rgb
    options_selected = sum([use_bgr, use_gray, use_rgb])
    if options_selected == 0:
        err_msg = "Must specify a color space to convert to."
        raise ValueError(err_msg)
    if options_selected > 1:
        err_msg = "Can only specify one color space to convert to."
        raise ValueError(err_msg)
    swap_color = "bgr"
    if use_gray:
        swap_color = "gray"
    if use_rgb:
        swap_color = "rgb"

    # create the cv2 color convert
    if start_color == "bgr":
        if use_bgr:
            convert_color = None
        elif use_gray:
            convert_color = cv2.COLOR_BGR2GRAY
        elif use_rgb:
            convert_color = cv2.COLOR_BGR2RGB
    elif start_color == "gray":
        if use_bgr:
            convert_color = cv2.COLOR_GRAY2BGR
        elif use_gray:
            convert_color = None
        elif use_rgb:
            convert_color = cv2.COLOR_GRAY2RGB
    elif start_color == "rgb":
        if use_bgr:
            convert_color = cv2.COLOR_RGB2BGR
        elif use_gray:
            convert_color = cv2.COLOR_RGB2GRAY
        elif use_rgb:
            convert_color = None
    else:
        err_msg = f"Unknown color space: {start_color}"
        raise ValueError(err_msg)

    if convert_color is None:
        err_msg = f"No conversion needed between: {start_color} and {swap_color}."
        raise ValueError(err_msg)

    video = IterableVideo(input_path)
    with VideoWriter(output_path, fps=video.fps) as writer:
        for _, frame in video:
            new_frame = cv2.cvtColor(frame, convert_color)
            writer.write(new_frame)
