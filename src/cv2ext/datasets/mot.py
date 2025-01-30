# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule for interacting with the MOT17(det)/20(det) datasets.

Classes
-------
:class:`MOTSequence`

:class:`MOTReader`

:class:`MOTWriter`


Functions
---------
:func:`get_mot_files`
    Gets the train and test file paths for MOT17(det)/20(det)
:func:`read_gt_det_mot_label`
    Read a det.txt or gt.txt file from a MOT sequence.

"""

from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path
from typing import TYPE_CHECKING

from cv2ext.io._fourcc import Fourcc
from cv2ext.io._iterablevideo import IterableVideo
from cv2ext.video._images import video_from_images

if TYPE_CHECKING:
    from types import TracebackType
    from typing import Generator

    import numpy as np
    from typing_extensions import Self


def _get_fps(seqinfo: Path) -> float:
    config = ConfigParser()
    config.read(seqinfo)
    return float(config["Sequence"]["frameRate"])


def _create_vid(subdir: Path, seqinfo: Path) -> None:
    vid_path = subdir / "video.mp4"
    img_dir = subdir / "img1"
    video_from_images(
        img_dir,
        vid_path,
        fps=_get_fps(seqinfo),
        fourcc=Fourcc.mp4v,
    )


def get_mot_files(
    mot_dir: str | Path,
) -> tuple[
    list[tuple[Path, Path, Path, Path, Path | None]],
    list[tuple[Path, Path, Path, Path | None]],
]:
    """
    Get the list of MOT label files.

    Parameters
    ----------
    mot_dir : Path | str | None
        The directory to search for the train/test directories
        The directory structure should be:
        mot_dir
        -----> test
        ----------> MOT_VIDEO_1
        ----------> MOT_VIDEO_4
        -----> train
        ----------> MOT_VIDEO_2
        ----------> MOT_VIDEO_3

    Returns
    -------
    tuple[
        list[tuple[Path, Path, Path, Path, Path | None]],
        list[tuple[Path, Path, Path, Path | None]],
    ]
        The training subdirectory, seqinfo.ini, video.mp4, gt.txt, and det.txt if it exists
        The testing subdirectory, seqinfo.ini, video.mp4, and det.txt if it exists

    Raises
    ------
    FileNotFoundError
        If the gt.txt, seqinfo.ini could not be resolved

    """
    # ensure mot_dir is a Path
    if isinstance(mot_dir, str):
        mot_dir = Path(mot_dir)

    train_subdir = mot_dir / "train"
    test_subdir = mot_dir / "test"

    # handle training files
    train_labels: list[tuple[Path, Path, Path, Path, Path | None]] = []
    subdirs = sorted(train_subdir.iterdir())
    for subdirectory in subdirs:
        # try to resolve det.txt
        det_label = subdirectory / "det" / "det.txt"
        if not det_label.exists():
            det_label = None

        # resolve gt.txt
        gt_label = subdirectory / "gt" / "gt.txt"
        if not gt_label.exists():
            err_msg = f"Could not find gt.txt in: {subdirectory}"
            raise FileNotFoundError(err_msg)

        # resolve seginfo.txt
        seqinfo = subdirectory / "seqinfo.ini"
        if not seqinfo.exists():
            err_msg = f"Could not find seqinfo.ini in: {subdirectory}"
            raise FileNotFoundError(err_msg)

        video = subdirectory / "video.mp4"
        if not video.exists():
            _create_vid(subdirectory, seqinfo)

        train_labels.append((subdirectory, seqinfo, video, gt_label, det_label))

    # handle testing files
    test_labels: list[tuple[Path, Path, Path, Path | None]] = []
    subdirs = sorted(test_subdir.iterdir())
    for subdirectory in subdirs:
        # try to resolve det.txt
        det_label = subdirectory / "det" / "det.txt"
        if not det_label.exists():
            det_label = None

        # resolve seginfo.txt
        seqinfo = subdirectory / "seqinfo.ini"
        if not seqinfo.exists():
            err_msg = f"Could not find seqinfo.ini in: {subdirectory}"
            raise FileNotFoundError(err_msg)

        video = subdirectory / "video.mp4"
        if not video.exists():
            _create_vid(subdirectory, seqinfo)

        test_labels.append((subdirectory, seqinfo, video, det_label))

    return train_labels, test_labels


def read_gt_det_mot_label(
    label: Path,
    seqlen: int,
) -> list[list[tuple[tuple[int, int, int, int], int]]]:
    """
    Read the ground truth MOT label file.

    Parameters
    ----------
    label : Path
        The path to the ground truth label file.
    seqlen : int
        The number of frames in the video.

    Returns
    -------
    list[list[tuple[tuple[int, int, int, int], int]]]
        The list of ground truth labels.
        Each label is a list of bounding boxes.

    """
    data: list[list[tuple[tuple[int, int, int, int], int]]] = [
        [] for _ in range(seqlen)
    ]

    with label.open("r") as f:
        lines = f.readlines()

    # each object ID has frames
    for idx, line in enumerate(lines):
        values = line.strip().split(",")
        if len(values) < 7:
            err_msg = (
                f"Invalid number of entries in {label.name} found on line: {idx + 1}"
            )
            raise ValueError(err_msg)

        # 7 elements per label
        fid, oid, x, y, w, h, conf = values[:7]

        # if marked with 0 conf then they are not used for accuracy assessment
        if int(conf) == 0:
            continue

        # convert to correct object/frame ids
        fid = int(fid)
        oid = int(oid)
        x, y, w, h = map(float, (x, y, w, h))
        data[fid - 1].append(((int(x), int(y), int(x + w), int(y + h)), oid))

    return data


class MOTSequence:
    """Class for iterating over frames in a MOT17(det)/20(det) sequence."""

    def __init__(
        self: Self,
        video: Path,
        dets: Path | None = None,
        gt: Path | None = None,
    ) -> None:
        """
        Initialize the MotReader object.

        Parameters
        ----------
        video : Path
            The path to the video file
        gt : Path, optional
            The path to the ground truths labels, if they exist
        dets : Path, optional
            The path to the detections, if they exist

        """
        self._video_path = video
        self._video = IterableVideo(self._video_path)

        # read dets if they exit
        self._dets: list[list[tuple[tuple[int, int, int, int], int]]] | None = None
        if dets is not None:
            self._dets = read_gt_det_mot_label(dets, len(self._video))

        # read ground truths if they exist
        self._gt: list[list[tuple[tuple[int, int, int, int], int]]] | None = None
        if gt is not None:
            self._gt = read_gt_det_mot_label(gt, len(self._video))

        # counter
        self._idx = 0

    @property
    def name(self: Self) -> str:
        """Get the name of the MOT video."""
        return self._video_path.stem

    @property
    def framesize(self: Self) -> tuple[int, int]:
        """Get the (width, height) of each frame."""
        return (self._video.width, self._video.height)

    def __iter__(self: Self) -> MOTSequence:
        """Return the iterator object."""
        return self

    def __next__(
        self: Self,
    ) -> tuple[
        np.ndarray,
        list[tuple[tuple[int, int, int, int], int]] | None,
        list[tuple[tuple[int, int, int, int], int]] | None,
    ]:
        """Return the next frame, det_label, and gt_label."""
        # will handle the StopIteration call for this object
        _, frame = next(self._video)

        det_label = None if self._dets is None else self._dets[self._idx]
        gt_label = None if self._gt is None else self._gt[self._idx]

        # increment and return
        self._idx += 1
        return frame, det_label, gt_label

    def __len__(self) -> int:
        """Return the number of labels."""
        return len(self._video)


class MOTReader:
    """Wrapper for MotReader, allowing iterating over videos."""

    def __init__(self, mot_dir: Path) -> None:
        """
        Create an instance of MotRunner.

        Parameters
        ----------
        mot_dir : Path
            The directory containing the test/train sequences for
            a MOT17(det)/20(det) dataset.

        """
        self._mot_dir = mot_dir
        self._train, self._test = get_mot_files(self._mot_dir)

    @property
    def test(self: Self) -> Generator[MOTSequence, None, None]:
        """
        Iterate over the test sequences.

        Returns
        -------
        Generator[MOTSequence, None, None]
            The generator of test sequences

        """
        for _, _, video, det in self._test:
            yield MOTSequence(video, dets=det)

    @property
    def train(self: Self) -> Generator[MOTSequence, None, None]:
        """
        Iterate over the train sequences.

        Returns
        -------
        Generator[MOTSequence, None, None]
            The generator of train sequences

        """
        for _, _, video, gt, det in self._train:
            yield MOTSequence(video, dets=det, gt=gt)


class MOTWriter:
    """Write labels from tracker to file for evaluation."""

    def __init__(
        self: Self,
        output_file: str | Path,
        digits: int | None = None,
    ) -> None:
        """
        Create an instance of MOTWriter.

        Parameters
        ----------
        output_file : str, Path
            The location of the output file to write.
            If a str, will be converted to a Path.
        digits : int, optional
            The number of digits to round bounding box values to.
            By default, None so no rounding will occur.

        """
        self._digits = digits
        self._filepath = (
            output_file if isinstance(output_file, Path) else Path(output_file)
        )
        self._file = self._filepath.open("w+")

    def add(
        self: Self,
        labels: list[tuple[int, int, float, float, float, float]],
    ) -> None:
        """
        Write a set of labels to the file.

        Parameters
        ----------
        labels : list[tuple[int, int, float, float, float, float]]
            The labels to write to the file
            Each tuple: frame id, object id, left, top, width, height

        """

        def set_digits(value: float, digits: int = 5) -> float:
            try:
                whole, _ = str(value).split(".")
            except ValueError:
                return value
            whole_len = len(whole)
            decimal_len = max(digits - whole_len, 0)
            return float(f"{value:.{decimal_len}f}")

        for label in labels:
            fid, cid, left, top, width, height = label

            if self._digits is not None:
                left = set_digits(left)
                top = set_digits(top)
                width = set_digits(width)
                height = set_digits(height)

            self._file.write(f"{fid},{cid},{left},{top},{width},{height},-1,-1,-1,-1\n")

    def close(self: Self) -> None:
        """Close the MOTWriter."""
        self._file.close()

    def __enter__(self: Self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the context manager."""
        self.close()
