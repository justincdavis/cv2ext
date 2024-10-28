# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np


def divide(
    image: np.ndarray,
    num_rows: int,
    num_cols: int,
    padding: int | tuple[int, int] | None = None,
    overlap_ratio: float | tuple[float, float] | None = None,
    *,
    copy: bool | None = None,
) -> tuple[list[list[np.ndarray]], list[list[tuple[int, int]]]]:
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
    overlap_ratio : float, tuple[float, float], optional
        An alternative to padding. Represents the overlap as a ratio
        of the overall size of the patch. This can be provided as an
        alternative to padding.
    copy : bool, optional
        If set to True, then will return copies of the subregions
        instead of memory views. Can have a performance impact.

    Returns
    -------
    tuple[list[list[np.ndarray]], list[list[tuple[int, int]]]]
        The computed subregions in a row major format and offsets in same format.
        The overall list represents rows and each sublist iterates
        over the patches column wise.
        The offsets are the same list dimensions as the subimages.

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
        if isinstance(overlap_ratio, float):
            vertical_overlap = int(row_height * overlap_ratio)
            horizontal_overlap = int(col_width * overlap_ratio)
        else:
            vertical_overlap = int(row_height * overlap_ratio[1])
            horizontal_overlap = int(col_width * overlap_ratio[0])

    else:
        vertical_overlap = 0
        horizontal_overlap = 0

    subimages: list[list[np.ndarray]] = []
    offsets: list[list[tuple[int, int]]] = []

    for i in range(num_rows):
        row_subimages = []
        row_offsets = []
        for j in range(num_cols):
            y_start = max(0, i * row_height - vertical_overlap)
            y_end = min(height, (i + 1) * row_height + vertical_overlap)
            x_start = max(0, j * col_width - horizontal_overlap)
            x_end = min(width, (j + 1) * col_width + horizontal_overlap)

            subimage = image[y_start:y_end, x_start:x_end]
            if copy:
                subimage = subimage.copy()

            row_subimages.append(subimage)
            row_offsets.append((x_start, y_start))

        subimages.append(row_subimages)
        offsets.append(row_offsets)

    return subimages, offsets


def patch(
    image: np.ndarray,
    patch_size: tuple[int, int],
    padding: int | tuple[int, int] | None = None,
    overlap: float | tuple[float, float] | None = None,
) -> tuple[list[np.ndarray], list[tuple[int, int]], tuple[int, int]]:
    """
    Divide an image into patches of a specific size.

    If the image is not evenly divisble by the patch size, it
    will be scaled up to be evenly divisible based on the patch
    size and the provided padding/overlap.

    Parameters
    ----------
    image : np.ndarray
        The image to patch into sections.
    patch_size : tuple[int, int]
        The patch size in (width, height).
    padding : int, tuple[int, int], optional
        The padding to apply between the patches. This
        is the number of pixels (or pixel x/y) to overlap.
    overlap : float, tuple[float, float], optional
        The overlap ratio to apply between the patches.
        For example, overlap=0.25 will have 25% of patch dimension
        be overlapping.

    Returns
    -------
    tuple[list[np.ndarray], list[tuple[int, int]], tuple[int, int]]
        The list of patches, list of offsets, and the image size
        which the patches and offsets are based off (could be different
        than input image size). The size is in format (width, height).

    Raises
    ------
    ValueError
        If the padding in x or y is larger than patch size in width or height.

    """

    def patch_dimension(patch_dim: int, image_dim: int, pad: int) -> tuple[int, int]:
        if image_dim < patch_dim:
            return patch_dim, 1
        effective_stride = patch_dim - pad
        num_patches = math.ceil((image_dim - patch_dim) / effective_stride) + 1
        # want there to be pad values if extra height
        new_dim = effective_stride * (num_patches - 1) + patch_dim
        return new_dim, num_patches

    height, width = image.shape[:2]
    patch_width, patch_height = patch_size

    pad = (0, 0)
    if padding:
        pad = padding if isinstance(padding, tuple) else (padding, padding)
    if overlap:
        if isinstance(overlap, tuple):
            pad_x = int(overlap[0] * patch_width)
            pad_y = int(overlap[1] * patch_height)
        else:
            pad_x = int(overlap * patch_width)
            pad_y = int(overlap * patch_height)
        pad = (pad_x, pad_y)

    if pad[0] >= patch_width:
        err_msg = f"Pad x cannot be greater than or equal to patch width: {pad[0]} !>= {patch_width}"
        raise ValueError(err_msg)
    if pad[1] >= patch_height:
        err_msg = f"Pad y cannot be greater than or equal to patch height: {pad[1]} !>= {patch_height}"
        raise ValueError(err_msg)

    # scale to the closest clustering of patches
    new_height, num_h_patches = patch_dimension(patch_height, height, pad[1])
    new_width, num_w_patches = patch_dimension(patch_width, width, pad[0])

    # resize image and preprocess ahead of time
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    patches: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    for i in range(num_h_patches):
        for j in range(num_w_patches):
            # Calculate start positions
            y_start = i * (patch_height - pad[1])
            x_start = j * (patch_width - pad[0])

            # Calculate end positions
            y_end = y_start + patch_height
            x_end = x_start + patch_width

            # print(i, j, x_start, y_start, x_end, y_end)

            # Extract patch
            patch = resized[y_start:y_end, x_start:x_end, :]

            # Since we resized to be perfectly divisible, no padding needed
            patches.append(patch)
            offsets.append((x_start, y_start))

    return patches, offsets, (new_width, new_height)
