# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
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
    def __init__(
        self: Self,
        tracker: cv2.Tracker,
    ) -> None:
        self._tracker: cv2.Tracker = tracker

    def _init(self: Self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        img_shape = image.shape[:2]
        self._image_shape = (img_shape[1], img_shape[0])
        self._tracker.init(image, xyxy_to_xywh(bbox))

    def _update(
        self: Self,
        image: np.ndarray,
    ) -> tuple[bool, tuple[int, int, int, int]]:
        retval, (x, y, w, h) = self._tracker.update(image)
        bbox = (int(x), int(y), int(w), int(h))
        xyxy = xywh_to_xyxy(bbox)
        # xyxy = constrain(xyxy, self._image_shape)
        return retval, xyxy


class AbstractMultiTracker(ABC):
    @abstractmethod
    def init(
        self: Self,
        image: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
    ) -> None:
        pass

    @abstractmethod
    def update(
        self: Self,
        image: np.ndarray,
    ) -> list[tuple[bool, tuple[int, int, int, int]]]:
        pass
