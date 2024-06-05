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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from cv2ext.bboxes import xywh_to_xyxy, xyxy_to_xywh

if TYPE_CHECKING:
    import cv2
    import numpy as np
    from typing_extensions import Self


class AbstractTracker(ABC):
    @abstractmethod
    def init(self: Self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        pass

    @abstractmethod
    def update(self: Self, image: np.ndarray) -> tuple[bool, tuple[int, int, int, int]]:
        pass


class CVTrackerInterface(AbstractTracker):
    def __init__(self: Self, tracker: AbstractTracker | cv2.TrackerCSRT | cv2.TrackerKCF | cv2.TrackerMIL) -> None:
        self._tracker: AbstractTracker | cv2.TrackerCSRT | cv2.TrackerKCF | cv2.TrackerMIL = tracker

    def _init(self: Self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        img_shape = image.shape[:2]
        self._image_shape = (img_shape[1], img_shape[0])
        self._tracker.init(image, xyxy_to_xywh(bbox))

    def _update(self: Self, image: np.ndarray) -> tuple[bool, tuple[int, int, int, int]]:
        retval, (x, y, w, h) = self._tracker.update(image)
        bbox = (int(x), int(y), int(w), int(h))
        xyxy = xywh_to_xyxy(bbox)
        # xyxy = constrain(xyxy, self._image_shape)
        return retval, xyxy


class AbstractMultiTracker(ABC):
    @abstractmethod
    def init(self: Self, image: np.ndarray, bboxes: list[tuple[int, int, int, int]]) -> None:
        pass

    @abstractmethod
    def update(self: Self, image: np.ndarray) -> list[tuple[bool, tuple[int, int, int, int]]]:
        pass
