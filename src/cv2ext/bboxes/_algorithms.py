# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from cv2ext._jit import register_jit

from ._iou import _iou_kernel

if TYPE_CHECKING:
    from collections.abc import Sequence


def filter_bboxes_by_region(
    bboxes: Sequence[tuple[int, int, int, int]],
    region: tuple[int, int, int, int],
    overlap: float = 0.6,
) -> list[tuple[int, int, int, int]]:
    """
    Get the bounding boxes contained within a region.

    Parameters
    ----------
    bboxes : Sequence[tuple[int, int, int, int]]
        The bounding boxes in form (x1, y1, x2, y2).
    region : tuple[int, int, int, int]
        The region by which to filter the bounding boxes.
    overlap : float
        The percentage of a bounding boxes area is inside
        the region to be included.

    Returns
    -------
    list[tuple[int, int, int, int]]
        The bounding boxes in the region.

    """
    filtered: list[tuple[int, int, int, int]] = []
    r_x1, r_y1, r_x2, r_y2 = region
    for bbox in bboxes:
        b_x1, b_y1, b_x2, b_y2 = bbox
        area = (b_x2 - b_x1) * (b_y2 - b_y1)

        # Calculate the intersection area
        x1 = max(b_x1, r_x1)
        y1 = max(b_y1, r_y1)
        x2 = min(b_x2, r_x2)
        y2 = min(b_y2, r_y2)
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

        if area == 0:
            continue

        # Calculate the percentage of the bbox area that is inside the region
        overlap_percentage = intersection_area / area
        if overlap_percentage >= overlap:
            filtered.append(bbox)

    return filtered


@register_jit()
def _match_kernel(
    bboxes1: Sequence[
        tuple[int, int, int, int] | tuple[tuple[int, int, int, int], float, int]
    ],
    bboxes2: Sequence[
        tuple[int, int, int, int] | tuple[tuple[int, int, int, int], float, int]
    ],
    iou_threshold: float = 0.5,
    *,
    class_agnostic: bool = False,
) -> list[tuple[int, int]]:
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        err_msg = "Each list of bboxes must have at least length of 1."
        raise ValueError(err_msg)

    matches: list[tuple[int, int]] = []
    used_idx: set[int] = set()

    for idx1, entry1 in enumerate(bboxes1):
        best_iou: float = 0.0
        best_idx: int = -1

        if len(entry1) == 3:
            bbox1, _, cid1 = entry1
        else:
            bbox1 = entry1
            cid1 = -1

        for idx2, entry2 in enumerate(bboxes2):
            if idx2 in used_idx:
                continue

            if len(entry2) == 3:
                bbox2, _, cid2 = entry2
            else:
                bbox2 = entry2
                cid2 = -1

            if not class_agnostic and cid1 != cid2:
                continue

            iou = _iou_kernel(bbox1, bbox2)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx2

        if best_iou >= iou_threshold:
            matches.append((idx1, best_idx))
            used_idx.add(best_idx)

    return matches


def match(
    bboxes1: Sequence[
        tuple[int, int, int, int] | tuple[tuple[int, int, int, int], float, int]
    ],
    bboxes2: Sequence[
        tuple[int, int, int, int] | tuple[tuple[int, int, int, int], float, int]
    ],
    iou_threshold: float = 0.5,
    *,
    class_agnostic: bool = False,
) -> list[tuple[int, int]]:
    """
    Match bounding boxes using a greedy algorithm.

    Parameters
    ----------
    bboxes1 : Sequence[tuple[int, int, int, int] | tuple[tuple[int, int, int, int], float, int]]
        The first Sequence of bounding boxes
    bboxes2 : Sequence[tuple[int, int, int, int] | tuple[tuple[int, int, int, int], float, int]]
        The second Sequence of bounding boxes
    iou_threshold : float, optional
        The IOU threshold which determines whether two bounding boxes are a match.
        By default, 0.5
    class_agnostic : bool, optional
        Whether or not to compare class ID (if present)
        By default, False

    Returns
    -------
    list[tuple[int, int]]
        A list of the matching indices

    """
    return _match_kernel(bboxes1, bboxes2, iou_threshold, class_agnostic=class_agnostic)


def calculate_metrics(
    bboxes1: Sequence[
        tuple[int, int, int, int] | tuple[tuple[int, int, int, int], float, int]
    ],
    bboxes2: Sequence[
        tuple[int, int, int, int] | tuple[tuple[int, int, int, int], float, int]
    ],
    iou_threshold: float = 0.5,
    epsilon: float = 1e-6,
    *,
    class_agnostic: bool = False,
) -> tuple[list[tuple[int, int]], dict[str, float]]:
    """
    Compute accuracy metrics between two Sequences of bounding boxes/detections.

    Bounding boxes are matched using the greedy algolrithm from :func:`match`.

    Parameters
    ----------
    bboxes1 : Sequence[tuple[int, int, int, int] | tuple[tuple[int, int, int, int], float, int]]
        The first Sequence of bounding boxes
    bboxes2 : Sequence[tuple[int, int, int, int] | tuple[tuple[int, int, int, int], float, int]]
        The second Sequence of bounding boxes
    iou_threshold : float, optional
        The IOU threshold which determines whether two bounding boxes are a match.
        By default, 0.5
    epsilon : float, optional
        The minimum/default value to prevent divide by zero errors.
        By default, 1e-6
    class_agnostic : bool, optional
        Whether or not to compare class ID (if present)
        By default, False

    Returns
    -------
    tuple[
        list[tuple[int, int]],
        dict[str, float],
    ]
        The list of matching indices and a dict of the computer metrics
        Metrics are: tp, fp, fn, precision, recall, f1

    """
    matches = match(bboxes1, bboxes2, iou_threshold, class_agnostic=class_agnostic)

    true_positives = len(matches)
    false_positives = len(bboxes1) - true_positives
    false_negatives = len(bboxes2) - true_positives

    default_val = max(int(len(bboxes2) == len(bboxes1)), epsilon)

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else default_val
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else default_val
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if precision + recall > 0
        else default_val
    )

    metrics = {
        "tp": max(true_positives, 0),
        "fp": max(false_positives, 0),
        "fn": max(false_negatives, 0),
        "precision": max(precision, 1e-6),
        "recall": max(recall, 1e-6),
        "f1": max(f1_score, 1e-6),
    }

    return matches, metrics
