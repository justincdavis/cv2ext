# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def divide(
    image: np.ndarray,
    num_rows: int,
    num_cols: int,
    padding: int | tuple[int, int] | None = None,
    overlap_ratio: float | None = None,
    *,
    copy: bool | None = None,
) -> list[list[np.ndarray]]:
    """
    Split an image into multiple sub-images.

    Parameters
    ----------
    image : np.ndarray
        The image to split
    num_rows : int
        The number of rows to divide the image into
    num_cols : int
        The number of columns to divide the image into
    padding : int, tuple[int, int], optional
        The padding to add to each region.
        Adds this value to each side (edges do not get padding).
        Can be a single value integer or a tuple representing
        the padding for (x, y)
    overlap_ratio : float, optional
        An alternative to padding. Represents the overlap as a ratio
        of the overall size of the patch. This can be provided as an
        alternative to padding
    copy : bool, optional
        If set to True, then will return copies of the subregions
        instead of memory views. Can have a performance impact.

    Returns
    -------
    list[list[np.ndarray]]
        The computed subregions in a row major format.
        The overall list represents rows and each sublist iterates
        over the patches column wise.

    Raises
    ------
    ValueError
        If both padding and overlap_ratio are given.

    """
    height, width = image.shape[:2]

    if padding and overlap_ratio:
        err_msg = "padding and overlap_ratio provided, only one can be used."
        raise ValueError(err_msg)

    row_height = height // num_rows
    col_width = width // num_cols

    if padding:
        if isinstance(padding, int):
            vertical_overlap = padding
            horizontal_overlap = padding
        else:
            vertical_overlap = padding[1]
            horizontal_overlap = padding[0]

    elif overlap_ratio:
        vertical_overlap = int(row_height * overlap_ratio)
        horizontal_overlap = int(col_width * overlap_ratio)

    else:
        vertical_overlap = 0
        horizontal_overlap = 0

    subimages = []

    for i in range(num_rows):
        row_subimages = []
        for j in range(num_cols):
            y_start = max(0, i * row_height - vertical_overlap)
            y_end = min(height, (i + 1) * row_height + vertical_overlap)
            x_start = max(0, j * col_width - horizontal_overlap)
            x_end = min(width, (j + 1) * col_width + horizontal_overlap)

            subimage = image[y_start:y_end, x_start:x_end]
            if copy:
                subimage = subimage.copy()
            row_subimages.append(subimage)
        subimages.append(row_subimages)

    return subimages
