# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator


def create_tiled_image(
    tile: np.ndarray,
    base: np.ndarray | None = None,
    tile_shape: tuple[int, int] | None = None,
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Tile an image across another image, or create a new image of the tile.

    Parameters
    ----------
    tile : np.ndarray
        The image to tile across the base image.
    base : np.ndarray, optional
        The base image to tile the image across.
        If provided, the tiles will be placed across the image
        in a grid pattern. Only placing a tile where the entrie
        tile can fit (i.e. no partial tiles).
        Only one of base, tile_shape, or image_shape can be provided.
    tile_shape : tuple[int, int], optional
        The shape of the tiles to create.
        This should be in the form of (rows, cols).
        Where rows is the number of tiles in the y direction,
        and cols is the number of tiles in the x direction.
        Only one of base, tile_shape, or image_shape can be provided.
    image_shape : tuple[int, int], optional
        The shape of the image to create.
        This should be in the form of (height, width).
        This is used to create a new image based on the tiles.
        Only one of base, tile_shape, or image_shape can be provided.

    Returns
    -------
    np.ndarray
        The tiled image.

    Raises
    ------
    ValueError
        If no base, tile shape, or image shape is provided.
    ValueError
        If more than one of base, tile shape, or image shape is provided.

    """
    overlay_height, overlay_width = tile.shape[:2]

    if base is None and tile_shape is None and image_shape is None:
        err_msg = "Either base, tile shape, or image shape must be provided."
        raise ValueError(err_msg)
    if base is not None and tile_shape is not None and image_shape is not None:
        err_msg = "Only one of base, tile shape, or image shape can be provided."
        raise ValueError(err_msg)

    # compute the shape of the base image and create if needed
    if base is not None:
        base_image = base
        base_height, base_width = base.shape[:2]
    elif tile_shape is not None:
        base_shape = tile_shape[0] * overlay_height, tile_shape[1] * overlay_width, 3
        base_image = np.ones(base_shape, dtype=np.uint8) * 255
        base_height, base_width = base_shape[:2]
    elif image_shape is not None:
        base_image = np.ones((image_shape[0], image_shape[1], 3), dtype=np.uint8) * 255
        base_height, base_width = base_image.shape[:2]
    else:
        err_msg = "Either base, tile shape, or image shape must be provided."
        raise ValueError(err_msg)

    # current indices
    x = 0
    y = 0
    # make a copy of the base image
    canvas: np.ndarray = base_image.copy()

    while y < base_height and y + overlay_height <= base_height:
        while x < base_width and x + overlay_width <= base_width:
            canvas[y : y + overlay_height, x : x + overlay_width] = tile
            x += overlay_width
        x = 0
        y += overlay_height

    return canvas


def image_tiler(
    base: np.ndarray,
    tile: np.ndarray,
) -> Generator[np.ndarray, None, None]:
    """
    Tile an image across another image.

    This function yields images with the tile progressively
    tiling across the base image. Each tile is placed in a grid fashion
    in row then column order. All tiles stay on the image, such that the
    final image produced will be fully filed (without partial fills).

    The first image produced will be the base image.

    Parameters
    ----------
    base : np.ndarray
        The base image to tile the image across.
    tile : np.ndarray
        The image to tile across the base image.

    Yields
    ------
    np.ndarray
        The tiled image.

    """
    overlay_height, overlay_width = tile.shape[:2]
    base_height, base_width = base.shape[:2]

    # current indices
    x = 0
    y = 0
    # make a copy of the base image
    canvas: np.ndarray = base.copy()
    yield canvas

    while y < base_height and y + overlay_height <= base_height:
        while x < base_width and x + overlay_width <= base_width:
            canvas[y : y + overlay_height, x : x + overlay_width] = tile
            yield canvas
            x += overlay_width
        x = 0
        y += overlay_height
    yield canvas
