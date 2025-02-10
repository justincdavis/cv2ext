# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Simple webcam program using cv2ext."""

from __future__ import annotations

import argparse
import contextlib
from typing import TYPE_CHECKING

import cv2ext

if TYPE_CHECKING:
    from argparse import Namespace


def _main(args: Namespace) -> None:
    source = args.source
    with contextlib.suppress(ValueError):
        source = int(source)

    with cv2ext.Display("webcam", stopkey="q") as display:
        for _, frame in cv2ext.IterableVideo(source):
            if display.stopped:
                break

            display(frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Simple webcam app.")
    parser.add_argument(
        "--source",
        "-s",
        required=True,
        help="The source of the webcam.",
    )
    args = parser.parse_args()
    _main(args)
