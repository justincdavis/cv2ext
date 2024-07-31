# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

from cv2ext.io import Fourcc, VideoWriter

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def video_from_images(
    directory: Path,
    output: Path,
    fps: float = 30.0,
    extensions: Sequence[str] = ("png", "jpg", "jpeg"),
    fourcc: Fourcc = Fourcc.mp4v,
) -> Path:
    """
    Create a video from a directory of images.

    The images are assumed to be named such that,
    when sorted, the video is constructed in the correct
    order.

    Parameters
    ----------
    directory : Path
        The directory containing the images.
    output : Path
        The output video file.
        The extension is assumed to match the codec, given
        by the fourcc parameter.
    fps : float
        The frames per second of the video.
        Defaults to 30.0.
    extensions : Sequence[str]
        The extensions of the image files.
        Defaults to ("png", "jpg", "jpeg").
        Extensions should not have a dot prefix.
    fourcc : Fourcc
        The fourcc codec to use.
        Defaults to MP4V.

    Returns
    -------
    Path
        The path to the output video file.

    """
    image_files = sorted([i for i in directory.iterdir() if i.suffix[1:] in extensions])

    with VideoWriter(str(output), fourcc, fps) as writer:
        for image_path in image_files:
            image = cv2.imread(str(image_path))
            writer.write(image)

    return output
