# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S311
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2

from cv2ext import IterableVideo


def _read_csv(
    csvfile: Path,
) -> tuple[list[list[tuple[int, int, int, int]]], str]:
    bboxes: list[list[tuple[int, int, int, int]]] = []
    formatstr = "xywh"
    with Path(csvfile).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            if len(bboxes) <= frame:
                bboxes.append([])
            try:
                x = int(row["x"])
                y = int(row["y"])
                w = int(row["w"])
                h = int(row["h"])
                formatstr = "xywh"
                bboxes[frame].append((x, y, w, h))
            except KeyError:
                x1 = int(row["x1"])
                y1 = int(row["y1"])
                x2 = int(row["x2"])
                y2 = int(row["y2"])
                formatstr = "xyxy"
                bboxes[frame].append((x1, y1, x2, y2))
    return bboxes, formatstr


def _read_json(
    jsonfile: Path,
) -> tuple[list[list[tuple[int, int, int, int]]], str]:
    with Path(jsonfile).open("r", encoding="utf-8") as f:
        dictdata: dict[str, dict[str, dict[str, int]]] = json.load(f)
    bboxes: list[list[tuple[int, int, int, int]]] = []
    formatstr = "xywh"
    found_format = False
    for bids in dictdata.values():
        local_bboxes: list[tuple[int, int, int, int]] = []
        for bbox in bids.values():
            if not found_format:
                if "x1" in bbox:
                    formatstr = "xyxy"
                found_format = True
            if formatstr == "xywh":
                x = bbox["x"]
                y = bbox["y"]
                w = bbox["w"]
                h = bbox["h"]
                formatstr = "xywh"
                local_bboxes.append((x, y, w, h))
            else:
                x1 = bbox["x1"]
                y1 = bbox["y1"]
                x2 = bbox["x2"]
                y2 = bbox["y2"]
                formatstr = "xyxy"
                local_bboxes.append((x1, y1, x2, y2))
        bboxes.append(local_bboxes)
    return bboxes, formatstr


def _write_yolo(
    bboxes: list[list[tuple[int, int, int, int]]],
    output_dir: Path,
    input_video: Path,
    formatstr: str,
    split: float = 0.8,
    classid: int = 0,
    classname: str = "object",
) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    else:
        err_msg = f"Output directory {output_dir} already exists."
        raise FileExistsError(err_msg)
    val_dir = output_dir / "val"
    train_dir = output_dir / "train"
    val_image_dir = val_dir / "images"
    train_image_dir = train_dir / "images"
    val_label_dir = val_dir / "labels"
    train_label_dir = train_dir / "labels"
    val_image_dir.mkdir(parents=True)
    train_image_dir.mkdir(parents=True)
    val_label_dir.mkdir(parents=True)
    train_label_dir.mkdir(parents=True)

    vidname = input_video.stem

    ids = [idx for idx, b in enumerate(bboxes) if len(b) > 0]
    val_ids = random.choices(ids, k=int(len(ids) * (1 - split)))

    for boxes, (fidx, frame) in zip(bboxes, IterableVideo(input_video)):
        if len(boxes) == 0:
            continue
        imagename = f"{vidname}_{fidx}.jpg"
        labelname = f"{vidname}_{fidx}.txt"

        imagepath = train_image_dir / imagename
        labelpath = train_label_dir / labelname
        if fidx in val_ids:
            imagepath = val_image_dir / imagename
            labelpath = val_label_dir / labelname

        cv2.imwrite(str(imagepath), frame)
        with labelpath.open("w") as f:
            for box in boxes:
                x1, y1, e1, e2 = box
                if formatstr == "xyxy":
                    w1 = e1 - x1
                    h1 = e2 - y1
                else:
                    w1 = e1
                    h1 = e2
                x = x1 / frame.shape[1]
                y = y1 / frame.shape[0]
                w = w1 / frame.shape[1]
                h = h1 / frame.shape[0]
                f.write(f"{classid} {x} {y} {w} {h}\n")

    yamlpath = output_dir / "data.yaml"
    with yamlpath.open("r") as f:
        f.write(f"train: {train_image_dir!s}\n")
        f.write(f"val: {val_image_dir!s}\n")
        f.write("nc: 1\n")
        f.write(f"names: ['{classname}']\n")


def convertannotations() -> None:
    parser = argparse.ArgumentParser(
        description="Convert annotations to other formats.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="The input file to convert.",
    )
    parser.add_argument(
        "--format",
        required=True,
        type=str,
        options=["yolo"],
        help="The format to convert to.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        default=None,
        help="The output file to write.",
    )
    parser.add_argument(
        "--input_video",
        type=Path,
        default=None,
        help="The input video file used for annotations.",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="The split ratio for the train/test split.",
    )
    parser.add_argument(
        "--classid",
        type=int,
        default=0,
        help="The class id to use for the annotations.",
    )
    parser.add_argument(
        "--classname",
        type=str,
        default="object",
        help="The class name to use.",
    )
    args = parser.parse_args()

    inputfile = args.input
    if not inputfile.exists():
        err_msg = f"Input file {inputfile} does not exist."
        raise FileNotFoundError(err_msg)

    bboxes: list[list[tuple[int, int, int, int]]]
    formatstr: str = "xyxy"
    if inputfile.suffix == ".csv":
        bboxes, formatstr = _read_csv(inputfile)
    elif inputfile.suffix == ".json":
        bboxes, formatstr = _read_json(inputfile)
    else:
        err_msg = f"Unknown file type: {inputfile.suffix}"
        raise ValueError(err_msg)

    if args.format == "yolo":
        if args.output_dir is None:
            err_msg = "Output directory is required for YOLO format."
            raise ValueError(err_msg)
        if args.input_video is None:
            err_msg = (
                "Input video is for annotation source is required for YOLO format."
            )
            raise ValueError(err_msg)
        _write_yolo(
            bboxes,
            args.output_dir,
            args.input_video,
            formatstr,
            args.split,
            args.classid,
            args.classname,
        )
    else:
        err_msg = f"Unknown format for conversion: {args.format}"
        raise ValueError(err_msg)
