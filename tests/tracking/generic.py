# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path
from typing import Type

import cv2
from cv2ext.io import IterableVideo
from cv2ext.tracking import Tracker, TrackerType, MultiTrackerType


def check_basic_tracking(tracker_type: TrackerType):
    try:
        tracker = Tracker(tracker_type)
    except ImportError:
        return

    image = cv2.imread(str(Path("data") / "pictograms.png"))
    init_bbox = (308, 308, 458, 454)
    tracker.init(image, init_bbox)

    if tracker_type in [TrackerType.TLD, TrackerType.KLT]:
        for _ in range(10):
            success, bbox = tracker.update(image)

            assert success

            for c in bbox:
                assert isinstance(c, int), f"Expected int but got {type(c)}"
            assert 0 <= bbox[0] <= image.shape[1] and 0 <= bbox[1] <= image.shape[0]
            assert 0 <= bbox[2] <= image.shape[1] and 0 <= bbox[3] <= image.shape[0]
    else:
        for _ in range(10):
            success, bbox = tracker.update(image)

            assert success
            
            for c1, c2 in zip(init_bbox, bbox):
                assert isinstance(c2, int), f"Expected int but got {type(c2)}"
                assert abs(c1 - c2) <= 3, f"Expected {c1} but got {c2}"
                init_bbox = bbox
            assert 0 <= bbox[0] <= image.shape[1] and 0 <= bbox[1] <= image.shape[0]
            assert 0 <= bbox[2] <= image.shape[1] and 0 <= bbox[3] <= image.shape[0]


def check_full_tracking(tracker_type: TrackerType, use_gray: bool):
    """Checks that a full run through a video will not crash."""
    try:
        tracker = Tracker(tracker_type)
    except ImportError:
        return True

    started = False
    for frame_id, frame in IterableVideo("data/testvid.mp4"):
        if frame_id < 100:
            continue
        if use_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not started:
            bbox = (149, 66, 69, 49)
            x, y, w, h = bbox
            bbox = (x, y, x + w, y + h)
            tracker.init(frame, bbox)
            started = True
        else:
            _, bbox = tracker.update(frame)

    return None


def check_basic_multi_tracking(tracker_type: MultiTrackerType):
    try:
        tracker = tracker_type.value()
    except ImportError:
        return True

    image = cv2.imread(str(Path("data") / "pictograms.png"))
    init_bboxs = [(308, 308, 458, 454)]
    tracker.init(image, init_bboxs)

    for _ in range(10):
        results = tracker.update(image)

        for i, (success, bbox) in enumerate(results):
            # check basic types
            assert isinstance(success, bool)
            assert isinstance(bbox, tuple)
            assert isinstance(bbox[0], int)

            # check bbox is in bounds
            assert 0 <= bbox[0] <= image.shape[1] and 0 <= bbox[1] <= image.shape[0]
            assert 0 <= bbox[2] <= image.shape[1] and 0 <= bbox[3] <= image.shape[0]

            if tracker_type != MultiTrackerType.KLT:
                # check diff of bboxes
                for c1, c2 in zip(init_bboxs[i], bbox):
                    assert isinstance(c2, int), f"Expected int but got {type(c2)}"
                    assert abs(c1 - c2) <= 3, f"Expected {c1} but got {c2}"


def check_full_multi_tracking(tracker_type: MultiTrackerType, use_gray: bool):
    """Checks that a full run through a video will not crash."""
    try:
        tracker = tracker_type.value()
    except ImportError:
        return True

    started = False
    for frame_id, frame in IterableVideo("data/testvid.mp4"):
        if frame_id < 100:
            continue
        if use_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not started:
            bbox = (149, 66, 69, 49)
            x, y, w, h = bbox
            bbox = (x, y, x + w, y + h)
            tracker.init(frame, [bbox])
            started = True
        else:
            results = tracker.update(frame)
            _, bbox = results[0]

    return None
