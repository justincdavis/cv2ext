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
    with Path(csvfile).open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            frame = int(row["frame"])
            if len(bboxes) <= frame:
                bboxes.append([])
            try:
                x = int(row["x"])
                y = int(row["y"])
                w = int(row["w"])
                h = int(row["h"])
                bboxes[frame].append((x, y, w, h))
            except KeyError:
                x1 = int(row["x1"])
                y1 = int(row["y1"])
                x2 = int(row["x2"])
                y2 = int(row["y2"])
                bboxes[frame].append((x1, y1, x2, y2))
    return bboxes


def _read_json(
    jsonfile: Path,
) -> tuple[list[list[tuple[int, int, int, int]]], str]:
    with Path(jsonfile).open("r") as jsonfile:
        dictdata: dict[str, dict[str, dict[str, int]]] = json.load(jsonfile)
    bboxes: list[list[tuple[int, int, int, int]]] = []
    formatstr = "xywh"
    for frame, bids in dictdata.items():
        if len(bboxes) <= frame:
            bboxes.append([])
        for bbox in bids.values():
            try:
                x = bbox["x"]
                y = bbox["y"]
                w = bbox["w"]
                h = bbox["h"]
                formatstr = "xywh"
                bboxes.append((x, y, w, h))
            except KeyError:
                x1 = bbox["x1"]
                y1 = bbox["y1"]
                x2 = bbox["x2"]
                y2 = bbox["y2"]
                formatstr = "xyxy"
                bboxes.append((x1, y1, x2, y2))
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
                    w = e1 - x1
                    h = e2 - y1
                else:
                    w = e1
                    h = e2
                x = x1 / frame.shape[1]
                y = y1 / frame.shape[0]
                w = w / frame.shape[1]
                h = h / frame.shape[0]
                f.write(f"{classid} {x} {y} {w} {h}\n")

    yamlpath = output_dir / "data.yaml"
    with yamlpath.open("r") as f:
        f.write(f"train: {train_image_dir!s}\n")
        f.write(f"val: {val_image_dir!s}\n")
        f.write("nc: 1\n")
        f.write(f"names: ['{classname}']\n")


def convertannotations() -> None:
    parser = argparse.ArgumentParser(description="Convert annotations to other formats.")
    parser.add_argument("--input", required=True, type=Path, help="The input file to convert.")
    parser.add_argument("--format", required=True, type=str, options=["yolo"], help="The format to convert to.")
    parser.add_argument("--output_dir", type=Path, default=None, help="The output file to write.")
    parser.add_argument("--input_video", type=Path, default=None, help="The input video file used for annotations.")
    parser.add_argument("--split", type=float, default=0.8, help="The split ratio for the train/test split.")
    parser.add_argument("--classid", type=int, default=0, help="The class id to use for the annotations.")
    parser.add_argument("--classname", type=str, default="object", help="The class name to use.")
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
            err_msg = "Input video is for annotation source is required for YOLO format."
            raise ValueError(err_msg)
        _write_yolo(bboxes, args.output_dir, args.input_video, formatstr, args.split, args.classid, args.classname)
    else:
        err_msg = f"Unknown format for conversion: {args.format}"
        raise ValueError(err_msg)
