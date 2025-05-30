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
from scipy.optimize import linear_sum_assignment

from cv2ext._jit import register_jit
from cv2ext.bboxes import iou


@register_jit(nogil=True)
def compute_iou_matrix(
    tracks: list[tuple[int, int, int, int]],
    detections: list[tuple[tuple[int, int, int, int], float, int]],
) -> np.ndarray:
    """
    Compute IoU matrix between tracks and detections.

    Parameters
    ----------
    tracks : list[tuple[int, int, int, int]]
        List of track bounding boxes in format (x1, y1, x2, y2)
    detections : list[tuple[tuple[int, int, int, int], float, int]]
        List of detection bounding boxes in format (x1, y1, x2, y2)

    Returns
    -------
    np.ndarray
        IoU matrix of shape (num_tracks, num_detections)

    """
    if not tracks or not detections:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)

    for i, track_bbox in enumerate(tracks):
        for j, (det_bbox, _, _) in enumerate(detections):
            iou_matrix[i, j] = iou(track_bbox, det_bbox)

    return iou_matrix


def linear_assignment(
    cost_matrix: np.ndarray,
    max_cost: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve linear assignment problem using Hungarian algorithm.

    Uses scipy's optimal linear_sum_assignment implementation for the Hungarian algorithm.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Cost matrix of shape (num_tracks, num_detections)
    max_cost : float, optional
        Maximum cost threshold for valid assignments, by default 1.0

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        matched_tracks, matched_detections, unmatched_tracks

    """
    if cost_matrix.size == 0:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.arange(cost_matrix.shape[0], dtype=np.int32),
        )

    if cost_matrix.max() <= 1.0:
        cost_matrix = 1.0 - cost_matrix

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    valid_assignments = cost_matrix[row_indices, col_indices] <= max_cost
    matched_tracks = row_indices[valid_assignments]
    matched_detections = col_indices[valid_assignments]

    used_tracks = set(matched_tracks)

    unmatched_tracks = np.array(
        [i for i in range(cost_matrix.shape[0]) if i not in used_tracks],
        dtype=np.int32,
    )

    return (
        matched_tracks.astype(np.int32),
        matched_detections.astype(np.int32),
        unmatched_tracks,
    )


def associate_tracks_to_detections(
    tracks: list[tuple[int, int, int, int]],
    detections: list[tuple[tuple[int, int, int, int], float, int]],
    iou_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Associate tracks to detections using IoU matching.

    Parameters
    ----------
    tracks : list[tuple[int, int, int, int]]
        List of track bounding boxes
    detections : list[tuple[tuple[int, int, int, int], float, int]]
        List of detection bounding boxes
    iou_threshold : float, optional
        Minimum IoU threshold for association, by default 0.3

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Matched track indices, matched detection indices, unmatched track indices

    """
    if not tracks or not detections:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.arange(len(tracks), dtype=np.int32),
        )

    iou_matrix = compute_iou_matrix(tracks, detections)

    cost_matrix = 1.0 - iou_matrix
    cost_matrix[iou_matrix < iou_threshold] = 1.0

    return linear_assignment(cost_matrix, max_cost=1.0 - iou_threshold)
