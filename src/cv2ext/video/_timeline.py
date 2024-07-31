# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from cv2ext.io import IterableVideo

if TYPE_CHECKING:
    from pathlib import Path


def create_timeline(
    videopath: Path,
    output: Path | None = None,
    bboxes: list[tuple[int, int, int, int]] | None = None,
    offset: int = 20,
    slices: int = 6,
    img_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Create a timeline image of a video.

    Parameters
    ----------
    videopath : Path
        Path to the video file.
    output : Path, optional
        Path to the output image.
        If None, the image will not be saved to disk.
    bboxes : list[tuple[int, int, int, int]], optional
        List of bounding boxes for each frame.
    offset : int, optional
        Number of pixels to offset the bounding boxes.
    slices : int, optional
        Number of frames to include in the timeline.
    img_size : tuple[int, int], optional
        Size of the images in the timeline.

    Returns
    -------
    np.ndarray
        The timeline image.

    Raises
    ------
    ValueError
        If slices and bboxes are not the same length.

    """
    if bboxes is not None and len(bboxes) != slices:
        err_msg = (
            f"Length of slices ({slices}) and bboxes ({len(bboxes)}) must be equal."
        )
        raise ValueError(err_msg)

    video = IterableVideo(str(videopath), use_thread=True)
    frames = []
    for i, frame in video:
        if i % (len(video) // slices) == 0:
            frames.append(frame)
    if bboxes is not None:
        new_frames = []
        for frame, bbox in zip(frames, bboxes):
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1 - offset)
            y1 = max(0, y1 - offset)
            x2 = min(frame.shape[1], x2 + offset)
            y2 = min(frame.shape[0], y2 + offset)
            cropped = frame[y1:y2, x1:x2]
            if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                new_frames.append(frame)
            else:
                new_frames.append(cropped)
        frames = new_frames
    if img_size is not None:
        frames = [
            cv2.resize(frame, img_size, interpolation=cv2.INTER_CUBIC)
            for frame in frames
        ]
    samesize = True
    for idx, frame in enumerate(frames):
        if idx == len(frames) - 1:
            break
        if frame.shape != frames[idx + 1].shape:
            samesize = False
            break
    if not samesize:
        framesize = frames[0].shape[0::2]
        frames = [
            cv2.resize(frame, framesize, interpolation=cv2.INTER_CUBIC)
            for frame in frames
        ]
    timeline: np.ndarray = np.concatenate(frames, axis=1)

    if output is not None:
        cv2.imwrite(str(output), timeline)

    return timeline
