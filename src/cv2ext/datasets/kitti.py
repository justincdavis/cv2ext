# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule for interacting with the KITTI dataset.

Functions
---------
:fund:`get_kitti_files`
    Gets the train and test file paths for KITTI

"""

from __future__ import annotations

from pathlib import Path

from cv2ext.io._fourcc import Fourcc
from cv2ext.video._images import video_from_images

from .coco import ID_MAP

_TO_COCO = {
    "Car": "car",
    "Truck": "truck",
    "Van": "car",
    "Pedestrian": "person",
    "Cyclist": "person",
}


def get_kitti_files(
    kitti_dir: str | Path,
) -> tuple[
    list[tuple[Path, Path, Path]],
    list[tuple[Path, Path]],
]:
    """
    Get the list of KITTI video/label files.

    Expects the same structure as is downloaded from the KITTI website.

    Parameters
    ----------
    kitti_dir : Path | str | None
        The directory to search for the train/test directories
        The directory structure should be:
        mot_dir
        -----> images
        ----------> testing
        ---------------> image_02
        --------------------> 0000
        -------------------------> 000000.png
        -------------------------> 000001.png
        ----------> training
        ---------------> image_02
        --------------------> 0000
        -------------------------> 000000.png
        -------------------------> 000001.png
        -----> labels
        ----------> training
        ---------------> label_02
        --------------------> 0000.txt
        --------------------> 0001.txt

    Returns
    -------
    tuple[
        list[tuple[Path, Path, Path]],
        list[tuple[Path, Path]],
    ]
        The training subdirectory, video.mp4, and label.txt
        The testing subdirectory and video.mp4

    Raises
    ------
    FileNotFoundError
        If a label file could not be found for training sequence

    """
    # ensure kitti_dir is a Path
    if isinstance(kitti_dir, str):
        kitti_dir = Path(kitti_dir)

    image_subdir = kitti_dir / "images"
    labels_subdir = kitti_dir / "labels"

    train_image_subdir = image_subdir / "training" / "image_02"
    test_image_subdir = image_subdir / "testing" / "image_02"
    train_label_subdir = labels_subdir / "training" / "label_02"

    # handle training files
    train_labels: list[tuple[Path, Path, Path]] = []
    subdirs = sorted(train_image_subdir.iterdir())
    for subdirectory in subdirs:
        label = train_label_subdir / subdirectory.stem / f"{subdirectory.stem}.txt"
        if not label.exists():
            err_msg = f"Could not find label file for video: {subdirectory.stem}"
            raise FileNotFoundError(err_msg)

        video = subdirectory / "video.mp4"
        if not video.exists():
            video_from_images(
                subdirectory,
                video,
                fourcc=Fourcc.mp4v,
            )

        train_labels.append((subdirectory, video, label))

    # handle testing files
    test_labels: list[tuple[Path, Path]] = []
    subdirs = sorted(test_image_subdir.iterdir())
    for subdirectory in subdirs:
        video = subdirectory / "video.mp4"
        if not video.exists():
            video_from_images(
                subdirectory,
                video,
                fourcc=Fourcc.mp4v,
            )

        test_labels.append((subdirectory, video))

    return train_labels, test_labels


def read_kitti_label(
    label: Path,
) -> list[list[tuple[tuple[int, int, int, int], int]]]:
    """
    Read the ground truth KITTI label file.

    Parameters
    ----------
    label : Path
        The path to the ground truth label file.

    Returns
    -------
    list[list[tuple[tuple[int, int, int, int], int]]]
        The list of ground truth labels.
        Each label is a list of bounding boxes.

    Raises
    ------
    ValueError
        If a label entry has insufficient elements

    """
    with label.open("r") as f:
        lines = f.readlines()

    seqlen = int(lines[-1].split(" ")[0]) + 1

    data: list[list[tuple[tuple[int, int, int, int], int]]] = [
        [] for _ in range(seqlen)
    ]

    # each object ID has frames
    for idx, line in enumerate(lines):
        values = line.strip().split(" ")
        if len(values) < 10:
            err_msg = (
                f"Invalid number of entries in {label.name} found on line: {idx + 1}"
            )
            raise ValueError(err_msg)

        fid, _, cid, _, _, _, x, y, w, h = values[:10]

        # if class is marked dont care then skip
        if cid == "DontCare":
            continue

        cid = _TO_COCO[cid]
        oid = ID_MAP[cid]

        # convert to correct object/frame ids
        fid = int(fid)
        x, y, w, h = map(float, (x, y, w, h))
        data[fid - 1].append(((int(x), int(y), int(x + w), int(y + h)), oid))

    return data
