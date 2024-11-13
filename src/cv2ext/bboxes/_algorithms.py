# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

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
