# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

import numpy as np

from typing_extensions import Self

from ..kernels import crop, csk_target, max_response, csk_train, csk_detection


class CSKTracker:
    def __init__(
            self: Self, 
            image: np.ndarray | None = None,
            bbox: tuple[int, int, int, int] | None = None,
            eta: float = 0.075, 
            sigma: float = 0.2, 
            lmbda: float = 0.01,
        ) -> None:
        # hyperparameters
        self._eta = eta
        self._sigma = sigma
        self._lambda = lmbda

        # state saving
        self._inited = False
        self._prev = None
        self._alpha_f = None

        if image is not None and bbox is not None:
            self.init(image, bbox)

    def init(self: Self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        """
        This method initializes the tracker with the initial bounding box.
        
        Parameters
        ----------
        image : np.ndarray
            The first frame of the video.
        bbox : tuple[int, int, int, int]
            The initial bounding box of the target.
            The bbox is represented as (x1, y1, x2, y2).
        """
        initial_window = crop(image, bbox)  # x
        initial_target = csk_target(initial_window.shape[0] // 2, initial_window.shape[1] // 2)  # y
        initial_response = max_response(initial_target)  # prev
        initial_alpha_f = csk_train(initial_window, initial_target, self._sigma, self._lambda)  # alphaf

        # state saving
        self._prev_bbox = bbox
        self._window = initial_window  # x
        self._target = initial_target  # y
        self._prev = initial_response  # prev
        self._alpha_f = initial_alpha_f  # alphaf
        self._inited = True

    def update(self: Self, image: np.ndarray) -> tuple[int, int, int, int]:
        """
        This method updates the tracker with the next frame.

        Parameters
        ----------
        image : np.ndarray
            The next frame of the video.
        
        Returns
        -------
        tuple[int, int, int, int]
            The bounding box of the target.
            The bbox is represented as (x1, y1, x2, y2).

        Raises
        ------
        ValueError
            If the tracker has not been initialized yet.
        """
        if not self._inited:
            raise ValueError("The tracker has not been initialized yet.")

        # print("CSKTracker")

        # process new image
        window = crop(image, self._prev_bbox)  # z
        responses = csk_detection(self._alpha_f, self._window, window, self._sigma)  
        response = max_response(responses)  # curr
        dx = response[1] - self._prev[1]
        dy = response[0] - self._prev[0]
        # print(dx, dy)

        # compute new bbox top-left and bottom-right coordinates
        x1 = self._prev_bbox[0] - dx
        y1 = self._prev_bbox[1] - dy
        x2 = self._prev_bbox[2] - dx
        y2 = self._prev_bbox[3] - dy
        new_bbox = (x1, y1, x2, y2)

        # print(f"{self._prev_bbox} -> {new_bbox}")
        
        # re-train the tracker
        new_window = crop(image, new_bbox)
        temp_interop_window = self._eta * new_window + (1 - self._eta) * self._window
        new_alpha_f = self._eta * csk_train(crop(image, new_bbox), self._target, self._sigma, self._lambda) + (1 - self._eta) * self._alpha_f
        
        # state saving
        self._prev_bbox = new_bbox
        self._prev = response
        self._window = temp_interop_window
        self._alpha_f = new_alpha_f

        return self._prev_bbox
