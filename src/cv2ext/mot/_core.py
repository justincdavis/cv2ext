# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Core functions for multi-object tracking algorithms.

This module provides common functionality used across different tracking algorithms
including association, track management, and detection processing.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from cv2ext._jit import register_jit
from cv2ext.bboxes import iou


@register_jit(nogil=True)
def compute_iou_matrix(
    detections: list[tuple[tuple[int, int, int, int], float, int]],
    tracks: list[tuple[tuple[int, int, int, int], float, int]],
) -> np.ndarray:
    """
    Compute IoU matrix between tracks and detections.

    Parameters
    ----------
    detections: list[tuple[tuple[int, int, int, int], float, int]]
        List of detections in format [(bbox, confidence, class_id), ...]
    tracks: list[tuple[tuple[int, int, int, int], float, int]]
        List of tracks in format [(bbox, confidence, class_id), ...]

    Returns
    -------
    np.ndarray
        IoU matrix of shape (num_tracks, num_detections)

    """
    if not tracks or not detections:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    iou_matrix: np.ndarray = np.zeros((len(tracks), len(detections)), dtype=np.float32)

    for i, (track_bbox, _, _) in enumerate(tracks):
        for j, (det_bbox, _, _) in enumerate(detections):
            iou_matrix[i, j] = iou(track_bbox, det_bbox)

    return iou_matrix


def linear_assignment(
    iou_matrix: np.ndarray,
    invalid_cost: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve linear assignment problem using Hungarian algorithm.

    Uses scipy's optimal linear_sum_assignment implementation for the Hungarian algorithm.

    Parameters
    ----------
    iou_matrix : np.ndarray
        Cost matrix of shape (num_tracks, num_detections)
    invalid_cost : float, optional
        Maximum cost threshold for valid assignments, by default 1.0

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        matches, unmatched_detections, unmatched_tracks

    """
    if iou_matrix.size == 0:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.arange(iou_matrix.shape[0], dtype=np.int32),
        )

    # invert values to linear_assignment minimizes based on iou (a max metric)
    cost_matrix = 1.0 - iou_matrix
    cost_matrix[cost_matrix > invalid_cost] = invalid_cost + 1.0

    # solve
    row_indices: np.ndarray
    col_indices: np.ndarray
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # stack
    matches = np.column_stack((row_indices, col_indices))

    # clear out bad entries which do not meet iou
    keep = cost_matrix[row_indices, col_indices] <= invalid_cost
    return matches[keep], row_indices[keep], col_indices[keep]


def associate_tracks_to_detections(
    detections: list[tuple[tuple[int, int, int, int], float, int]],
    tracks: list[tuple[tuple[int, int, int, int], float, int]],
    iou_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Associate tracks to detections using IoU matching.

    Parameters
    ----------
    detections: list[tuple[tuple[int, int, int, int], float, int]]
        List of detections in format [(bbox, confidence, class_id), ...]
    tracks: list[tuple[tuple[int, int, int, int], float, int]]
        List of tracks in format [(bbox, confidence, class_id), ...]
    iou_threshold : float, optional
        Minimum IoU threshold for association, by default 0.3

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Matches, unmatched detections, unmatched tracks

    """
    if len(tracks) == 0 or len(detections) == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.arange(len(detections), dtype=np.int32),
            np.empty((0, 1), dtype=np.int32),
        )

    iou_matrix = compute_iou_matrix(tracks, detections)

    matches, track_ids, det_ids = linear_assignment(
        iou_matrix, invalid_cost=1.0 - iou_threshold
    )

    # identify the unmatches indices in detections and tracks
    unmatched_dets = np.setdiff1d(
        np.arange(len(detections)), det_ids, assume_unique=True
    )
    unmatched_tracks = np.setdiff1d(
        np.arange(len(tracks)), track_ids, assume_unique=True
    )

    return matches, unmatched_dets, unmatched_tracks
