# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import cv2

_log = logging.getLogger(__name__)


def find_all_cameras(
    max_cameras: int = 10,
) -> Sequence[int]:
    """
    Find all available local webcams using cv2.VideoCapture.

    This function scans for local webcams by attempting to open each camera index
    up to max_cameras. It uses ThreadPoolExecutor to check cameras in parallel
    for better performance.

    Note: This function only searches for local webcams using integer indices.
    For other video sources like IP cameras, video files, or GStreamer pipelines,
    use IterableVideo directly with the appropriate URL or path.

    Parameters
    ----------
    max_cameras : int, optional
        The maximum number of cameras to scan for.
        Defaults to 10.

    Returns
    -------
    Sequence[int]
        A sequence of valid camera indices.

    """
    valid_cameras = []

    def check_camera(camera_index: int) -> tuple[int, bool]:
        """Check if a camera index is valid."""
        try:
            # try to open a camera
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                return camera_index, False
            ret, _ = cap.read()
        except (cv2.error, OSError, RuntimeError) as e:
            _log.debug(f"Camera {camera_index} is invalid: {e}")
            return camera_index, False
        else:
            cap.release()
            _log.debug(f"Camera {camera_index} is valid: {ret}")
            return camera_index, ret

    # disable opencv log messages
    original_log_level = cv2.getLogLevel()
    cv2.setLogLevel(0)

    try:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(check_camera, i) for i in range(max_cameras)]
            for future in futures:
                camera_index, is_valid = future.result()
                if is_valid:
                    valid_cameras.append(camera_index)
    finally:
        cv2.setLogLevel(original_log_level)

    return sorted(valid_cameras)
