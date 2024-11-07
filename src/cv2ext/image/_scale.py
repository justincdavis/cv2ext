# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def rescale(image: np.ndarray, value_range: tuple[float, float]) -> np.ndarray:
    """
    Rescale an image to be within the values given.

    Parameters
    ----------
    image : np.ndarray
        The image to scale. Should be a np.ndarray of type np.uint8.
    value_range : tuple[float, float]
        The values to convert the image to

    Returns
    -------
    np.ndarray
        The rescaled image

    Raises
    ------
    ValueError
        If value_range[0] is not less than value_range[1]

    """
    low, high = value_range
    if low >= high:
        err_msg = "Low value of range cannot be higher than or equal to high value."
        raise ValueError(err_msg)

    img_float: np.ndarray = image.astype(float)

    if low == 0.0:
        scaling_factor = high / 255.0
        scaled_img = img_float * scaling_factor
    else:
        normalized_img = img_float / 255.0
        scaled_img = normalized_img * (high - low) + low

    return scaled_img
