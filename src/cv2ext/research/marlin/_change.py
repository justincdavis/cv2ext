# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

if TYPE_CHECKING:
    from typing_extensions import Self


class ChangeDetector:
    """ChangeDetector for Marlin methodology."""

    def __init__(self, path: str | Path) -> None:
        self._forest: RandomForestClassifier = joblib.load(Path(path))
        if not isinstance(self._forest, RandomForestClassifier):
            err_msg = "ChangeDetector model must be RandomForestClassifier"
            raise ValueError(err_msg)

    def __call__(
        self: Self,
        image: np.ndarray,
        detections: list[tuple[tuple[int, int, int, int], float, int]],
    ) -> bool:
        return self.run(image, detections)

    @staticmethod
    def preprocess(
        image: np.ndarray,
        detections: list[tuple[tuple[int, int, int, int], float, int]],
    ) -> np.ndarray:
        frame = image.copy()
        for bbox, _, _ in detections:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)

        resized_colored_image = cv2.resize(frame, (128, 128))
        hist_red: np.ndarray = cv2.calcHist(
            [resized_colored_image],
            [0],
            None,
            [256],
            [0, 256],
        )
        hist_green: np.ndarray = cv2.calcHist(
            [resized_colored_image],
            [1],
            None,
            [256],
            [0, 256],
        )
        hist_blue: np.ndarray = cv2.calcHist(
            [resized_colored_image],
            [2],
            None,
            [256],
            [0, 256],
        )

        feature_vector: np.ndarray = resized_colored_image.reshape(1, -1)
        feature_vector = feature_vector.astype(float)

        vectors: list[np.ndarray] = [
            feature_vector,
            hist_red.flatten(),
            hist_green.flatten(),
            hist_blue.flatten(),
        ]

        return np.concatenate(vectors, axis=None)  # type: ignore[assignment]

    def run(
        self: Self,
        image: np.ndarray,
        detections: list[tuple[tuple[int, int, int, int], float, int]],
    ) -> bool:
        return self._forest.predict([self.preprocess(image, detections)])[0]
