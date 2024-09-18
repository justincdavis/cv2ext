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


def xyxy_to_nxywh(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Convert a bounding box from (xmin, ymin, xmax, ymax) to normalized (x, y, w, h) format.

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
        The converted bounding box in normalized (x, y, w, h) format.

    """
    x1, y1, x2, y2 = bbox
    return (
        x1 / image_width,
        y1 / image_height,
        (x2 - x1) / image_width,
        (y2 - y1) / image_height,
    )


def nxywh_to_xyxy(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """
    Convert a bounding box from normalized (x, y, w, h) format to (xmin, ymin, xmax, ymax).

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        The bounding box to convert in normalized (x, y, w, h) format.
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


def xywh_to_nxywh(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Convert a bounding box from (x, y, w, h) to normalized (x, y, w, h) format.

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
        The converted bounding box in normalized (x, y, w, h) format.

    """
    x, y, w, h = bbox
    return x / image_width, y / image_height, w / image_width, h / image_height


def nxywh_to_xywh(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """
    Convert a bounding box from normalized (x, y, w, h) format to (x, y, w, h).

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        The bounding box to convert in normalized (x, y, w, h) format.
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


def nxyxy_to_xyxy(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """
    Convert a normalized (x, y, x, y) bbox to absolute (x, y, x, y) coordinates.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        The normalized bounding box to convert.
        The format is (xmin, ymin, xmax, ymax) where values are normalized between 0 and 1.
    image_width : int
        The width of the image.
    image_height : int
        The height of the image.

    Returns
    -------
    tuple[int, int, int, int]
        The converted bounding box in (xmin, ymin, xmax, ymax) format.

    """
    x1, y1, x2, y2 = bbox
    return (
        int(x1 * image_width),
        int(y1 * image_height),
        int(x2 * image_width),
        int(y2 * image_height),
    )


def nxyxy_to_xywh(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """
    Convert a normalized (x, y, x, y) bbox to absolute (x, y, w, h) coordinates.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        The normalized bounding box to convert.
        The format is (xmin, ymin, xmax, ymax) where values are normalized between 0 and 1.
    image_width : int
        The width of the image.
    image_height : int
        The height of the image.

    Returns
    -------
    tuple[int, int, int, int]
        The converted bounding box in (x, y, w, h) format.

    """
    x1, y1, x2, y2 = bbox
    return (
        int(x1 * image_width),
        int(y1 * image_height),
        int(x2 * image_width - x1 * image_width),
        int(y2 * image_height - y1 * image_height),
    )


def nxyxy_to_nxywh(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Convert a normalized (x, y, x, y) bbox to normalized (x, y, w, h) format.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        The normalized bounding box to convert.
        The format is (xmin, ymin, xmax, ymax) where values are normalized between 0 and 1.
    image_width : int
        The width of the image.
    image_height : int
        The height of the image.

    Returns
    -------
    tuple[float, float, float, float]
        The converted bounding box in normalized (x, y, w, h) format.

    """
    x1, y1, x2, y2 = bbox
    return (
        x1 * image_width,
        y1 * image_height,
        (x2 - x1) / image_width,
        (y2 - y1) / image_height,
    )


def xyxy_to_nxyxy(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Convert a bounding box from (x, y, x, y) to normalized (x, y, x, y).

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
        The converted bounding box in normalized format.

    """
    x1, y1, x2, y2 = bbox
    return (
        x1 / image_width,
        y1 / image_height,
        x2 / image_width,
        y2 / image_height,
    )


def xywh_to_nxyxy(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Convert a bounding box from (x, y, w, h) to normalized (x, y, x, y).

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
        The converted bounding box in normalized format.

    """
    x, y, w, h = bbox
    return (
        x / image_width,
        y / image_height,
        (x + w) / image_width,
        (y + h) / image_height,
    )


def nxywh_to_nxyxy(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """
    Convert a bounding box from normalized (x, y, w, h) format to normalized (x, y, x, y).

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        The bounding box to convert in normalized (x, y, w, h) format.
    image_width : int
        The width of the image.
    image_height : int
        The height of the image.

    Returns
    -------
    tuple[float, float, float, float]
        The converted bounding box in normalized format.

    """
    x, y, w, h = bbox
    return (
        x,
        y,
        x + w,
        y + h,
    )


def yolo_to_xyxy(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """
    Convert a YOLO bounding box format to (xmin, ymin, xmax, ymax).

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        The bounding box in YOLO format (x_center, y_center, width, height).
    image_width : int
        The width of the image.
    image_height : int
        The height of the image.

    Returns
    -------
    tuple[int, int, int, int]
        The converted bounding box in (xmin, ymin, xmax, ymax) format.

    """
    cx, cy, w, h = bbox
    return (
        int((cx - w / 2) * image_width),
        int((cy - h / 2) * image_height),
        int((cx + w / 2) * image_width),
        int((cy + h / 2) * image_height),
    )


def yolo_to_nxyxy(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """
    Convert a YOLO bounding box format to normalized (xmin, ymin, xmax, ymax).

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        The bounding box in YOLO format (x_center, y_center, width, height).

    Returns
    -------
    tuple[float, float, float, float]
        The converted bounding box in normalized (xmin, ymin, xmax, ymax) format.

    """
    cx, cy, w, h = bbox
    return (
        (cx - w / 2),
        (cy - h / 2),
        (cx + w / 2),
        (cy + h / 2),
    )


def yolo_to_xywh(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """
    Convert a YOLO bounding box format to (x, y, w, h).

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        The bounding box in YOLO format (x_center, y_center, width, height).
    image_width : int
        The width of the image.
    image_height : int
        The height of the image.

    Returns
    -------
    tuple[int, int, int, int]
        The converted bounding box in (x, y, w, h) format.

    """
    cx, cy, w, h = bbox
    return (
        int((cx - w / 2) * image_width),
        int((cy - h / 2) * image_height),
        int(w * image_width),
        int(h * image_height),
    )


def yolo_to_nxywh(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """
    Convert a YOLO bounding box format to normalized (x, y, w, h).

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        The bounding box in YOLO format (x_center, y_center, width, height).

    Returns
    -------
    tuple[float, float, float, float]
        The converted bounding box in normalized (x, y, w, h) format.

    """
    cx, cy, w, h = bbox
    return (
        cx - w / 2,
        cy - h / 2,
        w,
        h,
    )


def xyxy_to_yolo(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Convert a bounding box from (xmin, ymin, xmax, ymax) to YOLO format (x_center, y_center, width, height).

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        The bounding box to convert.
        Format is (xmin, ymin, xmax, ymax).
    image_width : int
        The width of the image.
    image_height : int
        The height of the image.

    Returns
    -------
    tuple[float, float, float, float]
        The converted bounding box in YOLO format (x_center, y_center, width, height).

    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return (
        (x1 + w / 2) / image_width,
        (y1 + h / 2) / image_height,
        w / image_width,
        h / image_height,
    )


def nxyxy_to_yolo(
    bbox: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    """
    Convert a bounding box from normalized (xmin, ymin, xmax, ymax) to YOLO format (x_center, y_center, width, height).

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        The bounding box to convert.
        Format is normalized (xmin, ymin, xmax, ymax).

    Returns
    -------
    tuple[float, float, float, float]
        The converted bounding box in YOLO format (x_center, y_center, width, height).

    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return (
        (x1 + w / 2),
        (y1 + h / 2),
        w,
        h,
    )


def xywh_to_yolo(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Convert a bounding box from (x, y, w, h) to YOLO format (x_center, y_center, width, height).

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        The bounding box to convert.
        Format is (x, y, w, h).
    image_width : int
        The width of the image.
    image_height : int
        The height of the image.

    Returns
    -------
    tuple[float, float, float, float]
        The converted bounding box in YOLO format (x_center, y_center, width, height).

    """
    x, y, w, h = bbox
    return (
        (x + w / 2) / image_width,
        (y + h / 2) / image_height,
        w / image_width,
        h / image_height,
    )


def nxywh_to_yolo(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """
    Convert a bounding box from normalized (x, y, w, h) to YOLO format (x_center, y_center, width, height).

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        The bounding box to convert in normalized (x, y, w, h) format.

    Returns
    -------
    tuple[float, float, float, float]
        The converted bounding box in YOLO format (x_center, y_center, width, height).

    """
    x, y, w, h = bbox
    return (
        x + w / 2,
        y + h / 2,
        w,
        h,
    )
