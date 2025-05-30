# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from abc import ABC, abstractmethod


class Tracker(ABC):
    """
    Abstract base class for multi-object tracking algorithms.

    This class provides a common interface for all tracking algorithms.
    """

    @abstractmethod
    def update(
        self,
        detections: list[tuple[tuple[int, int, int, int], float, int]],
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """Update the tracker with new detections."""

    @abstractmethod
    def predict(self) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """Predict the next state of the tracks."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the tracker."""
