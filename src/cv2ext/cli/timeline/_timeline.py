# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from cv2ext import IterableVideo


def timeline_cli() -> None:
    parser = argparse.ArgumentParser(description="Create a timeline image of a video.")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the video file.",
    )
    parser.add_argument("--bboxes", type=str, help="Path to the annotation file.")
    parser.add_argument(
        "--offset",
        type=int,
        default=20,
        help="Number of pixels to offset the bounding boxes.",
    )
    parser.add_argument("--output", type=str, help="Path to the output image.")
    parser.add_argument(
        "--slices",
        type=int,
        default=6,
        help="Number of frames to include in the timeline.",
    )
    parser.add_argument(
        "--img_size",
        type=str,
        help="Size of the images in the timeline.",
    )

    args = parser.parse_args()

    videopath = Path(args.video)
    if not videopath.exists():
        err_msg = f"File not found: {videopath}"
        raise FileNotFoundError(err_msg)

    output_path = Path(args.output) if args.output is not None else None
    img_size: tuple[int, int] | None = (
        tuple(map(int, args.img_size.strip("[]").split(",")))  # type: ignore[assignment]
        if args.img_size is not None
        else None
    )
    max_imgsize = 2
    if img_size is not None and len(img_size) != max_imgsize:
        err_msg = "Image size must be in the format: [width, height]"
        raise ValueError(err_msg)

    bboxes: list[tuple[int, int, int, int]] | None = None
    if args.bboxes is not None:
        bboxespath = Path(args.bboxes)
        if not bboxespath.exists():
            err_msg = f"File not found: {bboxespath}"
            raise FileNotFoundError(err_msg)

        with bboxespath.open() as f:
            try:
                bboxes = [
                    tuple(map(int, line.strip().split(",")))  # type: ignore[misc]
                    for line in f.readlines()
                ]
            except ValueError as err:
                err_msg = "Bounding boxes must be in the format: x1,y1,x2,y2"
                raise ValueError(err_msg) from err

        bbox_size = 4
        for bbox in bboxes:
            if len(bbox) != bbox_size:
                err_msg = f"Bounding box must have {bbox_size} values."
                raise ValueError(err_msg)

    timeline(videopath, output_path, bboxes, args.offset, args.slices, img_size)


def timeline(
    videopath: Path,
    output_path: Path | None = None,
    bboxes: list[tuple[int, int, int, int]] | None = None,
    offset: int = 20,
    slices: int = 6,
    img_size: tuple[int, int] | None = None,
) -> None:
    """
    Create a timeline image of a video.

    Parameters
    ----------
    videopath : Path
        Path to the video file.
    output_path : Path, optional
        Path to the output image.
    bboxes : list[tuple[int, int, int, int]], optional
        List of bounding boxes for each frame.
    offset : int, optional
        Number of pixels to offset the bounding boxes.
    slices : int, optional
        Number of frames to include in the timeline.
    img_size : tuple[int, int], optional
        Size of the images in the timeline.

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
    timeline = np.concatenate(frames, axis=1)

    if output_path is not None:
        cv2.imwrite(str(output_path), timeline)
    else:
        pathstr = str(videopath)
        output_path = Path(pathstr[: pathstr.rfind(".")] + "_timeline.png")
        cv2.imwrite(str(output_path), timeline)
