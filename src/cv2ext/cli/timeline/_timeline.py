# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import argparse
from pathlib import Path

from cv2ext.video import create_timeline


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

    if output_path is None:
        pathstr = str(videopath)
        output_path = Path(pathstr[: pathstr.rfind(".")] + "_timeline.png")

    create_timeline(videopath, output_path, bboxes, args.offset, args.slices, img_size)
