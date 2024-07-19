# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations


def xyxy_to_xywh(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """
    Convert a bounding box from (xmin, ymin, xmax, ymax) to (x, y, w, h).

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        The bounding box to convert.
        Bounding box is format (xmin, ymin, xmax, ymax),
        where (xmin, ymin) is the top-left corner and (xmax, ymax) is the bottom-right corner.

    Returns
    -------
    tuple[int, int, int, int]
        The converted bounding box.
        Bounding box is format (x, y, w, h),
        where (x, y) is the top-left corner and (w, h) is the width and height.

    """
    x1, y1, x2, y2 = bbox
    return x1, y1, x2 - x1, y2 - y1


def xywh_to_xyxy(
    bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """
    Convert a bounding box from (x, y, w, h) to (xmin, ymin, xmax, ymax).

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        The bounding box to convert.
        Bounding box is format (x, y, w, h),
        where (x, y) is the top-left corner and (w, h) is the width and height.

    Returns
    -------
    tuple[int, int, int, int]
        The converted bounding box.
        Bounding box is format (xmin, ymin, xmax, ymax),
        where (xmin, ymin) is the top-left corner and (xmax, ymax) is the bottom-right corner.

    """
    x, y, w, h = bbox
    return x, y, x + w, y + h


def xyxy_to_yolo(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Convert a bounding box from (xmin, ymin, xmax, ymax) to YOLO format.

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        The bounding box to convert.
        Bounding box is format (xmin, ymin, xmax, ymax),
        where (xmin, ymin) is the top-left corner and (xmax, ymax) is the bottom-right corner.
    image_width : int
        The width of the image.
    image_height : int
        The height of the image.

    Returns
    -------
    tuple[float, float, float, float]
        The converted bounding box in YOLO format.

    """
    x1, y1, x2, y2 = bbox
    return (
        x1 / image_width,
        y1 / image_height,
        (x2 - x1) / image_width,
        (y2 - y1) / image_height,
    )


def yolo_to_xyxy(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """
    Convert a bounding box from YOLO format to (xmin, ymin, xmax, ymax).

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        The bounding box to convert in YOLO format.
    image_width : int
        The width of the image.
    image_height : int
        The height of the image.

    Returns
    -------
    tuple[int, int, int, int]
        The converted bounding box in (xmin, ymin, xmax, ymax) format.

    """
    x, y, w, h = bbox
    return (
        int(x * image_width),
        int(y * image_height),
        int((x + w) * image_width),
        int((y + h) * image_height),
    )


def xywh_to_yolo(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Convert a bounding box from (x, y, w, h) to YOLO format.

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        The bounding box to convert.
        Bounding box is format (x, y, w, h),
        where (x, y) is the top-left corner and (w, h) is the width and height.
    image_width : int
        The width of the image.
    image_height : int
        The height of the image.

    Returns
    -------
    tuple[float, float, float, float]
        The converted bounding box in YOLO format.

    """
    x, y, w, h = bbox
    return x / image_width, y / image_height, w / image_width, h / image_height


def yolo_to_xywh(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """
    Convert a bounding box from YOLO format to (x, y, w, h).

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        The bounding box to convert in YOLO format.
    image_width : int
        The width of the image.
    image_height : int
        The height of the image.

    Returns
    -------
    tuple[int, int, int, int]
        The converted bounding box in (x, y, w, h) format.

    """
    x, y, w, h = bbox
    return (
        int(x * image_width),
        int(y * image_height),
        int(w * image_width),
        int(h * image_height),
    )
