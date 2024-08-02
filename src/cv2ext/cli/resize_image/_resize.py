# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def resize_image_cli() -> None:
    """
    Resize an image.

    Raises
    ------
    FileNotFoundError
        If the input image is not found.
    IsADirectoryError
        If the input path is not a file.

    """
    parser = argparse.ArgumentParser(description="Resize an image.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="The path to the input image.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="The path to the output image.",
    )
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="The width of the output image.",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="The height of the output image.",
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
    width = int(args.width)
    height = int(args.height)

    image = cv2.imread(str(input_path))
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_path), resized_image)
