# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging

import cv2
import numpy as np

_log = logging.getLogger(__name__)

def generate_camera_intrinsics(images: list[np.ndarray], chessboard_size: tuple[int, int] = (8, 8)) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate camera intrinsics from a list of images containing a chessboard pattern.

    Parameters
    ----------
    images : list[np.ndarray]
        List of images containing a chessboard pattern.
    chessboard_size : tuple[int, int], optional
        Size of the chessboard pattern, by default (8, 8)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Camera intrinsics and distortion coefficients

    """
    cols, rows = chessboard_size
    obj_points: np.ndarray = np.zeros((cols * rows, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    img_points = []
    for idx, image in enumerate(images):
        img = image
        # convert to grayscale if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find chessboard in the frame
        success, corners = cv2.findChessboardCorners(img, chessboard_size, None)
        if success:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            refined_corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            img_points.append(refined_corners)
        else:
          _log.error(f"Could not identify chessboard in image {idx}")

    count = len(img_points)
    success, k, dist, rvecs, tvecs = cv2.calibrateCamera([obj_points] * count, img_points, chessboard_size, None, None)

    _log.debug(f"Generated calibration data from {count} images")
    _log.debug(f"Camera intrinsic matrix:, {k}")
    _log.debug(f"Distortion coefficients: {dist}")

    error = 0
    for i in range(count):
        proj_points, _ = cv2.projectPoints(obj_points, rvecs[i], tvecs[i], k, dist)
        error += cv2.norm(proj_points, img_points[i], cv2.NORM_L2)

    error_val = error / (count * cols * rows)
    error_str = f"Mean error: {error_val} (should be close to zero)"
    if error_val > 0.2:
        _log.warning(error_str)
    else:
        _log.info(error_str)

    return k, dist
