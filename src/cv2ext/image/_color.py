# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import math

import cv2
import numpy as np


def dominant_color(
    image: np.ndarray,
    bbox: tuple[int, int, int, int] | None = None,
    num_colors: int = 4,
    black_threshold: int = 50,
    *,
    ignore_black: bool | None = None,
) -> tuple[int, int, int]:
    """
    Get the dominant color from an image.

    If there is no bounding box provided, then the color
    is computed across the entire image given. If a bbox is given
    then only the data within the bounding box is considered.
    The underlying computation is completed by using KMeans
    clustering on the pixel values.

    Parameters
    ----------
    image : np.ndarray
        The image to compute the color on
    bbox : tuple[int, int, int, int], optional
        The optional bounding box to sample the image with.
        The bounding box should be in form:
        x1, y1, x2, y2 (top-left, bottom-right) format
    num_colors : int
        The number of clusters to use for kmeans.
    black_threshold : int
        The cutoff for when a pixel can be considered black.
        This cutoff is assessed by compared the B, G, and R
        channel values individually.
    ignore_black : bool, optional
        Whether or not to ignore black pixels when returning
        the dominant color.

    Returns
    -------
    tuple[int, int, int]
        The dominant color in BGR format.

    """
    if bbox:
        x1, y1, x2, y2 = bbox
        image = image[y1:y2, x1:x2]

    pixels = np.float32(image.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, num_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant: list[list[float]] = palette[np.argsort(counts)[::-1]]  # type: ignore[index]

    if ignore_black:
        for i in range(len(dominant)):
            d_pixel: list[float] = dominant[i]
            if not (
                d_pixel[0] < black_threshold
                and d_pixel[1] < black_threshold
                and d_pixel[2] < black_threshold
            ):
                return int(dominant[i][0]), int(dominant[i][1]), int(dominant[i][2])
        return int(dominant[0][0]), int(dominant[0][1]), int(dominant[0][2])
    return int(dominant[0][0]), int(dominant[0][1]), int(dominant[0][2])


def mean_color(
    image: np.ndarray,
    bbox: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int]:
    """
    Get the mean color from an image.

    The mean color is computed by splitting the image into
    the corresponding blue, green, and red channels and
    then performing the mean computation on each channel.

    Parameters
    ----------
    image : np.ndarray
        The image to compute the color on
    bbox : tuple[int, int, int, int], optional
        The optional bounding box to sample the image with.
        The bounding box should be in form:
        x1, y1, x2, y2 (top-left, bottom-right) format

    Returns
    -------
    tuple[int, int, int]
        The mean color in BGR format.

    """
    if bbox:
        x1, y1, x2, y2 = bbox
        image = image[y1:y2, x1:x2]

    blue, green, red = cv2.split(image)
    return (
        int(np.mean(blue)),
        int(np.mean(green)),
        int(np.mean(red)),
    )


def color_euclidean_dist(
    color1: tuple[int, int, int],
    color2: tuple[int, int, int],
) -> float:
    """
    Compute the euclidean distance between two colors.

    The user can provide two colors manually, but this function
    was intended to be used with the outputs of 'mean_color' and
    'dominant_color'. Primarily, this function is used to compare
    color changes over time or difference between the two methods.

    Parameters
    ----------
    color1 : tuple[int, int, int]
        The first color.
    color2 : tuple[int, int, int]
        The second color.

    Returns
    -------
    float
        The euclidean distance between the two colors.

    """
    return math.sqrt(
        (color1[0] - color2[0]) ** 2
        + (color1[1] - color2[1]) ** 2
        + (color1[2] - color2[2]) ** 2,
    )
