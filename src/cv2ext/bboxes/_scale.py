# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations


def scale(
    bbox: tuple[int, int, int, int],
    s1: tuple[int, int],
    s2: tuple[int, int],
) -> tuple[int, int, int, int]:
    """

    Scale a bounding box from one image size to another.

    Takes a bounding box (x1, y1, x2, y2) within the image size
    s1 (width, height) and transform it to be a bounding box (x1, y1, x2, y2)
    for the image size s2 (width, height)

    Parameters
    ----------
    bbox: tuple[int, int, int, int]
        The bounding box (x1, y1, x2, y2) in image size s1 (width, height)
    s1: tuple[int, int]
        The input image size the bbox is based on
    s2: tuple[int, int]
        The output image size to transform the bbox to

    Returns
    -------
    tuple[int, int, int, int]
        A (x1, y1, x2, y2) bounding box in the s2 image size

    """
    x1, y1, x2, y2 = bbox  # Unpack the bounding box coordinates

    # Calculate scaling factors for x and y dimensions
    scale_x = s2[0] / s1[0]
    scale_y = s2[1] / s1[1]

    # Scale the coordinates using the calculated scaling factors
    scaled_x1 = int(x1 * scale_x)
    scaled_y1 = int(y1 * scale_y)
    scaled_x2 = int(x2 * scale_x)
    scaled_y2 = int(y2 * scale_y)

    return (scaled_x1, scaled_y1, scaled_x2, scaled_y2)


def scale_many(
    bboxes: list[tuple[int, int, int, int]],
    s1: tuple[int, int],
    s2: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """

    Scale a bounding box from one image size to another.

    Takes a bounding box (x1, y1, x2, y2) within the image size
    s1 (width, height) and transform it to be a bounding box (x1, y1, x2, y2)
    for the image size s2 (width, height)

    Parameters
    ----------
    bboxes: list[tuple[int, int, int, int]]
        The bounding boxes of form (x1, y1, x2, y2) in image size s1 (width, height)
    s1: tuple[int, int]
        The input image size the bboxes are based on
    s2: tuple[int, int]
        The output image size to transform the bbox to

    Returns
    -------
    list[tuple[int, int, int, int]]
        List of bounding boxes of form (x1, y1, x2, y2) in the s2 image size

    """
    return [scale(bbox, s1, s2) for bbox in bboxes]
