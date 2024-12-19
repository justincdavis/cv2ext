# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
from pathlib import Path

import cv2
from cv2ext.tracking import MultiTracker, TrackerType


def _create_tracker(use_threads: bool):
    # suppress ImportError, means we are catching the attribute error
    # from not having contrib
    with contextlib.suppress(ImportError):
        for tracker_type in TrackerType:
            tracker = MultiTracker(tracker_type, use_threads=use_threads)
            assert tracker is not None


def test_create_tracker():
    _create_tracker(False)


def test_create_tracker_threads():
    _create_tracker(True)


def _init_no_error(use_threads: bool):
    image = cv2.imread(str(Path("data") / "pictograms.png"))
    init_bbox = (308, 308, 458, 454)

    # suppress ImportError, means we are catching the attribute error
    # from not having contrib
    with contextlib.suppress(ImportError):
        for tracker_type in TrackerType:
            tracker = MultiTracker(tracker_type, use_threads=use_threads)
            tracker.init(image, [init_bbox])


def test_init_no_error():
    _init_no_error(False)    


def test_init_no_error_threads():
    _init_no_error(True)


def _data_cycle(use_threads: bool):
    image = cv2.imread(str(Path("data") / "pictograms.png"))
    init_bbox = (308, 308, 458, 454)

    # suppress RuntimeErorr, means we are catching if the thread
    # cannot open the underlying tracker
    with contextlib.suppress(RuntimeError, ImportError):
        for tracker_type in TrackerType:
            tracker = MultiTracker(tracker_type, use_threads=use_threads)
            tracker.init(image, [init_bbox])
            results = tracker.update(image)

            for success, bbox in results:
                assert success
                assert isinstance(bbox, tuple)


def test_data_cycle():
    _data_cycle(False)


def test_data_cycle_threads():
    _data_cycle(True)


def _many_data_cycle(use_threads: bool):
    image = cv2.imread(str(Path("data") / "pictograms.png"))
    init_bbox = (308, 308, 458, 454)
    init_bboxes = [init_bbox] * 10

    # suppress RuntimeErorr, means we are catching if the thread
    # cannot open the underlying tracker
    with contextlib.suppress(RuntimeError, ImportError):
        for tracker_type in TrackerType:
            tracker = MultiTracker(tracker_type, use_threads=use_threads)
            tracker.init(image, init_bboxes)
            results = tracker.update(image)
            
            assert len(results) == 10
            for success, bbox in results:
                assert success
                assert isinstance(bbox, tuple)


def test_many_data_cycle():
    _many_data_cycle(False)


def test_many_data_cycle_threads():
    _many_data_cycle(True)
