# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S311
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from cv2ext._jit import register_jit

if TYPE_CHECKING:
    from typing_extensions import Self


class AbstractFramePacker(ABC):
    """
    Pack regions of a frame together based on detection activity.

    Detections are represented as a list of bounding boxes with
    scores and class id labels optional.
    """

    @abstractmethod
    def pack(
        self: Self,
        image: np.ndarray,
        exclude: tuple[int, int, int, int] | list[tuple[int, int, int, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Pack regions of a frame together.

        Parameters
        ----------
        image : np.ndarray
            The image to pack.
        exclude : tuple[int, int, int, int] | list[tuple[int, int, int, int]]
            Regions of the image to exclude from the packing.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The packed image and the transform information.

        """

    @abstractmethod
    def unpack(
        self: Self,
        detections: list[tuple[int, int, int, int]]
        | list[tuple[tuple[int, int, int, int], float, int]],
        transform: np.ndarray,
    ) -> (
        list[tuple[int, int, int, int]]
        | list[tuple[tuple[int, int, int, int], float, int]]
    ):
        """
        Unpack regions of a frame.

        Parameters
        ----------
        detections : list[tuple[int, int, int, int]] | list[tuple[tuple[int, int, int, int], float, int]]
            The regions to unpack.
        transform : np.ndarray
            The transform information generated by the pack method.

        Returns
        -------
        list[tuple[int, int, int, int]] | list[tuple[tuple[int, int, int, int], float, int]]

        """

    @abstractmethod
    def update(
        self: Self,
        detections: list[tuple[int, int, int, int]]
        | list[tuple[tuple[int, int, int, int], float, int]],
    ) -> None:
        """
        Update the packer with new detections.

        Parameters
        ----------
        detections : list[tuple[int, int, int, int]] | list[tuple[tuple[int, int, int, int], float, int]]
            The detections to update the packer with.

        """

    @abstractmethod
    def reset(self: Self, image_shape: tuple[int, int] | None = None) -> None:
        """
        Reset the packer.

        Parameters
        ----------
        image_shape : tuple[int, int] | None
            The shape of the image in form (height, width).

        """


@register_jit()
def _bbox_to_gridcells(
    bbox: tuple[int, int, int, int],
    row_step: int,
    col_step: int,
    n_rows: int,
    n_cols: int,
) -> list[tuple[int, int]]:
    x1, y1, x2, y2 = bbox
    min_row = max(0, math.floor(y1 / row_step))
    max_row = min(n_rows, math.ceil(y2 / row_step))
    min_col = max(0, math.floor(x1 / col_step))
    max_col = min(n_cols, math.ceil(x2 / col_step))

    cells: list[tuple[int, int]] = []
    for i in range(min_row, max_row):
        for j in range(min_col, max_col):
            cells.extend([(i, j)])
    return cells


@register_jit()
def _bboxes_to_gridcells(
    bboxes: list[tuple[int, int, int, int]],
    row_step: int,
    col_step: int,
    n_rows: int,
    n_cols: int,
) -> set[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for bbox in bboxes:
        cells.extend(_bbox_to_gridcells(bbox, row_step, col_step, n_rows, n_cols))
    return set(cells)


@register_jit()
def _unpack_grid_single_bbox(
    bbox: tuple[int, int, int, int],
    transform: np.ndarray,
    gridsize: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox

    # Determine grid coordinates for the packed bounding box
    n_col = int(((x1 + x2) / 2.0) // gridsize)
    n_row = int(((y1 + y2) / 2.0) // gridsize)

    # Retrieve original top-left offset for the grid cell
    o_x, o_y = transform[n_row][n_col]

    # Calculate the absolute coordinates in the original image
    abs_x1 = x1 - (n_col * gridsize) + o_x
    abs_y1 = y1 - (n_row * gridsize) + o_y
    abs_x2 = x2 - (n_col * gridsize) + o_x
    abs_y2 = y2 - (n_row * gridsize) + o_y

    return abs_x1, abs_y1, abs_x2, abs_y2


@register_jit()
def _unpack_grid_bboxes_conf_classid(
    detections: list[tuple[tuple[int, int, int, int], float, int]],
    transform: np.ndarray,
    gridsize: int,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    unpacked_dets: list[tuple[tuple[int, int, int, int], float, int]] = []
    for bbox, conf, classid in detections:
        abs_x1, abs_y1, abs_x2, abs_y2 = _unpack_grid_single_bbox(
            bbox,
            transform,
            gridsize,
        )

        # Append the unpacked detection
        unpacked_dets.append(((abs_x1, abs_y1, abs_x2, abs_y2), conf, classid))
    return unpacked_dets


@register_jit()
def _unpack_grid_bboxes(
    detections: list[tuple[int, int, int, int]],
    transform: np.ndarray,
    gridsize: int,
) -> list[tuple[int, int, int, int]]:
    unpacked_dets: list[tuple[int, int, int, int]] = []
    for bbox in detections:
        abs_x1, abs_y1, abs_x2, abs_y2 = _unpack_grid_single_bbox(
            bbox,
            transform,
            gridsize,
        )

        # Append the unpacked detection
        unpacked_dets.append((abs_x1, abs_y1, abs_x2, abs_y2))
    return unpacked_dets


class AnnealingFramePacker(AbstractFramePacker):
    """
    Pack regions of a frame together based on detection activity.

    Detections are represented as a list of bounding boxes with
    scores and class id labels optional.
    """

    def __init__(
        self: Self,
        image_shape: tuple[int, int],
        gridsize: int = 128,
        alpha: float = 0.01,
        min_prob: float = 0.1,
        detection_buffer: int = 30,
    ) -> None:
        """
        Create a new AnnealingFramePacker.

        Parameters
        ----------
        image_shape : tuple[int, int]
            The shape of the image in form (height, width).
        gridsize : int, optional
            The size of each cell in the overlaid grid.
        alpha : float, optional
            The learning rate for the annealing process.
        min_prob : float, optional
            The minimum probability for a region to be considered active.
        detection_buffer : int, optional
            The number of frames to consider for detection activity.
            Used instead of current frame count once frame count exceeds buffer size.
            Allows more recent detections to have more influence.

        """
        super().__init__()
        self._height, self._width = image_shape
        self._gridsize = gridsize
        self._alpha = alpha
        self._min_prob = min_prob
        self._detection_buffer = detection_buffer

        # assign type hints to variables used in initialize_cells
        self._n_cols: int
        self._n_rows: int
        self._col_step: int
        self._row_step: int
        self._num_dets: np.ndarray
        self._cells: np.ndarray

        self._initialize_cells()

        # tracking variables
        self._counter: int = 0

    def reset(self: Self, image_shape: tuple[int, int] | None = None) -> None:
        """
        Reset the packer.

        Parameters
        ----------
        image_shape : tuple[int, int] | None
            The shape of the image in form (height, width).

        """
        if image_shape:
            self._height, self._width = image_shape
        self._initialize_cells()
        self._counter = 0

    def _initialize_cells(self: Self) -> None:
        """Initialize the grid cells and related parameters."""
        # num rows/cols
        self._n_cols = int(math.ceil(self._width / self._gridsize))
        self._n_rows = int(math.ceil(self._height / self._gridsize))

        # step size between grid cells
        self._col_step = (
            int((self._width - self._gridsize) / (self._n_cols - 1))
            if self._n_cols > 1
            else self._width
        )
        self._row_step = (
            int((self._height - self._gridsize) / (self._n_rows - 1))
            if self._n_rows > 1
            else self._height
        )

        # detection count info
        self._num_dets = np.zeros((self._n_rows, self._n_cols), dtype=int)

        # create the cell setup
        self._cells = np.zeros((self._n_rows * self._n_cols, 6), dtype=int)
        index: int = 0
        for i in range(self._n_rows):
            for j in range(self._n_cols):
                v_off = i * self._row_step
                h_off = j * self._col_step
                bbox = (
                    h_off,
                    v_off,
                    h_off + self._gridsize,
                    v_off + self._gridsize,
                )
                self._cells[index, :4] = bbox
                self._cells[index, 4:] = (i, j)
                index += 1

    def _should_explore(self: Self, detections: int) -> bool:
        time_factor = math.exp(-self._alpha * self._counter)
        detection_factor = min(
            1.0,
            detections / (min(self._counter, self._detection_buffer - 1) + 1),
        )
        explore_probability = max(self._min_prob, time_factor + detection_factor)
        return random.random() < explore_probability

    def pack(
        self,
        image: np.ndarray,
        exclude: tuple[int, int, int, int]
        | list[tuple[int, int, int, int]]
        | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Pack regions of a frame together.

        Parameters
        ----------
        image : np.ndarray
            The image to pack.
        exclude : tuple[int, int, int, int] | list[tuple[int, int, int, int]], optional
            Regions of the image to exclude from the packing.
            By default None.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The packed image and the transform information.

        Raises
        ------
        ValueError
            If the image shape does not match the packer shape.

        """
        height, width = image.shape[:2]
        if height != self._height or width != self._width:
            err_msg = f"Image shape {image.shape} does not match packer shape {self._height, self._width}."
            raise ValueError(err_msg)

        # get the excluded cells
        excluded_cells: set[tuple[int, int]] = set()
        if exclude is not None and len(exclude) > 0:
            if isinstance(exclude, tuple):
                excluded_cells = set(
                    _bbox_to_gridcells(
                        exclude,
                        self._row_step,
                        self._col_step,
                        self._n_rows,
                        self._n_cols,
                    ),
                )
            else:
                excluded_cells = _bboxes_to_gridcells(
                    exclude,
                    self._row_step,
                    self._col_step,
                    self._n_rows,
                    self._n_cols,
                )

        # get all cells which are not in the excluded region
        included_cells = [
            (x1, y1, x2, y2, r, c)
            for (x1, y1, x2, y2, r, c) in self._cells
            if (r, c) not in excluded_cells
        ]

        # need to assess if the cells should be included based on detections and NCC
        filtered_cells: list[tuple[tuple[int, int, int, int], tuple[int, int]]] = []
        for x1, y1, x2, y2, row, col in included_cells:
            bbox = (x1, y1, x2, y2)
            if self._should_explore(
                self._num_dets[row][col],
            ):
                filtered_cells.append((bbox, (row, col)))

        # need to repack the cells into a new image
        num_cells = len(filtered_cells)
        dim1 = max(1, math.ceil(math.sqrt(num_cells)))
        dim2 = max(1, math.ceil(num_cells / dim1))

        # allocate new data for the patches
        new_image: np.ndarray = np.zeros(
            (dim2 * self._gridsize, dim1 * self._gridsize, 3),
            dtype=np.uint8,
        )

        # copy the old data into new packed image
        # generate the transforms
        new_grids: np.ndarray = np.zeros((self._n_rows, self._n_cols, 2), dtype=int)
        for i, (bbox, _) in enumerate(filtered_cells):
            x1, y1, x2, y2 = bbox

            # generate the new row/col
            n_row = math.floor(i / dim1)  # Fixed dimension calculation
            n_col = i % dim1

            # generate the new bounding box
            n_x1 = n_col * self._gridsize
            n_x2 = n_x1 + self._gridsize
            n_y1 = n_row * self._gridsize
            n_y2 = n_y1 + self._gridsize

            # generate the offset, same as old coords for x1, y1
            offset = (x1, y1)

            # perform the data copy
            new_image[n_y1:n_y2, n_x1:n_x2] = image[y1:y2, x1:x2]

            # save the new grid entry
            new_grids[n_row, n_col] = offset

        # update the image
        self._prev_image = image
        self._counter += 1

        return new_image, new_grids

    def unpack(
        self: Self,
        detections: list[tuple[int, int, int, int]]
        | list[tuple[tuple[int, int, int, int], float, int]],
        transform: np.ndarray,
    ) -> (
        list[tuple[int, int, int, int]]
        | list[tuple[tuple[int, int, int, int], float, int]]
    ):
        """
        Unpack regions of a frame.

        Parameters
        ----------
        detections : list[tuple[int, int, int, int]] | list[tuple[tuple[int, int, int, int], float, int]]
            The regions to unpack.
        transform : np.ndarray
            The transform information generated by the pack method.

        Returns
        -------
        list[tuple[int, int, int, int]] | list[tuple[tuple[int, int, int, int], float, int]]

        """
        if len(detections) == 0:
            return []

        # get the type of detections passed
        if len(detections[0]) == 3:
            return _unpack_grid_bboxes_conf_classid(
                detections,  # type: ignore[arg-type]
                transform,
                self._gridsize,
            )

        return _unpack_grid_bboxes(detections, transform, self._gridsize)  # type: ignore[arg-type]

    def update(
        self: Self,
        detections: list[tuple[int, int, int, int]]
        | list[tuple[tuple[int, int, int, int], float, int]],
    ) -> None:
        """
        Update the packer with new detections.

        Parameters
        ----------
        detections : list[tuple[int, int, int, int]] | list[tuple[tuple[int, int, int, int], float, int]]
            The detections to update the packer with.

        """
        # add det counts to all possible grids with detections
        for elements in detections:
            if len(elements) == 3:
                bbox, _, _ = elements
            else:
                bbox = elements

            # get the grid cells that the bbox intersects
            grids = _bbox_to_gridcells(
                bbox,
                self._row_step,
                self._col_step,
                self._n_rows,
                self._n_cols,
            )

            # increment the detection count for each grid cell
            for o_row, o_col in grids:
                if 0 <= o_row < self._n_rows and 0 <= o_col < self._n_cols:
                    self._num_dets[o_row][o_col] += 1

        # decrement all counters by 1
        self._num_dets = np.maximum(0, self._num_dets - 1)
