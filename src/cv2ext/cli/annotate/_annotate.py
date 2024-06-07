# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from cv2ext import IterableVideo


def _write_csv(
    outputpath: Path,
    bboxes: list[list[tuple[int, int, int, int]]],
    formatarg: str,
) -> None:
    fieldnames = ["frame", "bid", "x1", "y1", "x2", "y2"]
    if formatarg == "xywh":
        fieldnames = ["frame", "bid", "x", "y", "w", "h"]
    with Path(outputpath).open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx1, local_bboxes in enumerate(bboxes):
            if len(local_bboxes) == 0:
                continue
            for idx2, bbox in enumerate(local_bboxes):
                row = {"frame": idx1, "bid": idx2}
                if formatarg == "xywh":
                    row["x"] = bbox[0]
                    row["y"] = bbox[1]
                    row["w"] = bbox[2]
                    row["h"] = bbox[3]
                else:
                    row["x1"] = bbox[0]
                    row["y1"] = bbox[1]
                    row["x2"] = bbox[2]
                    row["y2"] = bbox[3]
                writer.writerow(row)


def _write_json(
    outputpath: Path,
    bboxes: list[list[tuple[int, int, int, int]]],
    formatarg: str,
) -> None:
    dictdata: dict[str, dict[str, dict[str, int]]] = {}
    for idx1, local_bboxes in enumerate(bboxes):
        localdictdata: dict[str, dict[str, int]] = {}
        if len(local_bboxes) == 0:
            continue
        for idx2, bbox in enumerate(local_bboxes):
            bboxdict: dict[str, int] = {}
            if formatarg == "xywh":
                bboxdict = {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]}
            else:
                bboxdict = {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}
            localdictdata[str(idx2)] = bboxdict
        dictdata[str(idx1)] = localdictdata
    with Path(outputpath).open("w", encoding="utf-8") as f:
        json.dump(dictdata, f, indent=4)


def annotate() -> None:
    parser = argparse.ArgumentParser(description="Annotate a video with a template.")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="The video file to annotate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The output file to save the annotations to, valid types are: [json, csv]",
    )
    parser.add_argument(
        "--format",
        type=str,
        required=False,
        default="xyxy",
        help="The format to annotations as: [xyxy, xywh]",
    )
    args = parser.parse_args()

    videopath = Path(args.video)
    if not videopath.exists():
        err_msg = f"File not found: {videopath!s}"
        raise FileNotFoundError(err_msg)

    outputpath = Path(args.output)
    if outputpath.suffix not in [".json", ".csv"]:
        err_msg = f"Unsupported file extension: {outputpath.suffix}"
        raise ValueError(err_msg)

    if args.format not in ["xyxy", "xywh"]:
        err_msg = f"Unsupported format: {args.format}"
        raise ValueError(err_msg)

    video = IterableVideo(videopath)

    bboxes: list[list[tuple[int, int, int, int]]] = []
    for _, frame in video:
        local_bboxes = []
        while True:
            x, y, h, w = cv2.selectROI("Select ROI", frame)
            bbox = (x, y, h, w)
            if bbox == (0, 0, 0, 0):
                break
            if args.format == "xywh":
                fbbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            else:
                fbbox = (
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[0] + bbox[2]),
                    int(bbox[1] + bbox[3]),
                )
            local_bboxes.append(fbbox)
        bboxes.append(local_bboxes)

    if outputpath.suffix == ".json":
        _write_json(outputpath, bboxes, args.format)
    elif outputpath.suffix == ".csv":
        _write_csv(outputpath, bboxes, args.format)
    else:
        err_msg = f"Unsupported file extension: {outputpath.suffix}"
        err_msg += " This should not happen. Please report this issue."
        raise RuntimeError(err_msg)
