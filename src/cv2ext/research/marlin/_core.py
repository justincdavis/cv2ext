# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import pathlib

import numpy as np

from ._change import ChangeDetector
from ._tracker import MultiBoxTracker


class Marlin:
    def __init__(
        self,
        dnn: callable[[np.ndarray], list[tuple[int, tuple[int, int, int, int], float]]],
        forest: str | pathlib.Path,
        ncc_threshold: float = 0.3,
        nfeatures: int = 500,
        verbose: bool | None = None,
    ) -> None:
        """Use to create a Marlin object."""
        self._tracker = MultiBoxTracker(nfeatures=nfeatures)
        self._dnn = dnn
        self._change_dect = ChangeDetector(forest)
        self._ncc_threshold = ncc_threshold
        self._last_frame = None
        self._use_dnn = True
        self._last_bboxs: list[tuple[int, tuple[int, int, int, int], float]] | None = (
            None
        )
        self._last_ncc: float | None = None
        self._verbose = verbose if verbose is not None else False

    def _get_cv_frame(self) -> np.ndarray:
        frame = self._cv_frame
        self._cv_frame = None
        return frame

    def add_cv_frame(self, frame: np.ndarray) -> None:
        """Use when the preprocessing of the dnn means errors."""
        self._cv_frame = frame

    def __call__(
        self,
        frame: np.ndarray,
    ) -> list[tuple[int, tuple[int, int, int, int], float]]:
        """Run the Marlin algorithm on the (next) frame"""
        orig_frame = frame.copy()
        if len(frame.shape) == 4:
            frame = np.transpose(frame.squeeze(), (1, 2, 0))
        if self._cv_frame is not None:
            frame = self._get_cv_frame()
        cv_frame = frame.copy()
        cd_frame = frame.copy()
        if self._verbose:
            print("Processing new frame:")
            print(f"    Use DNN: {self._use_dnn}")
        if self._use_dnn or self._last_bboxs is None:
            self._use_dnn = False
            self._last_bboxs = self._dnn(orig_frame)
            self._last_frame = cv_frame
            self._tracker.init(self._last_frame, self._last_bboxs)
            if self._verbose:
                print(f"    DNN detected {len(self._last_bboxs)} objects")
                print(f"    Confidence: {self._last_bboxs[0][2]}")
        else:
            prev_bboxs = self._last_bboxs
            self._last_bboxs = self._tracker.run(cv_frame)
            self._last_frame = cv_frame
            if self._verbose:
                print(f"    Tracker detected {len(self._last_bboxs)} objects")
            if len(self._last_bboxs) == 0 or (
                len(self._last_bboxs) == 1 and self._last_bboxs[0][0] == None
            ):
                self._use_dnn = True
                self._last_bboxs = (
                    prev_bboxs  # get the previous ones to reduce the flashing effect
                )
            else:
                self._last_ncc = min(self._last_bboxs, key=lambda x: x[2])[2]
                self._use_dnn = self._last_ncc <= self._ncc_threshold
            if self._verbose:
                print(f"    NCC: {self._last_ncc}")
                print(f"    Use DNN: {self._use_dnn}")

        # RUN CHANGE DECT ALWAYS
        result = self._change_dect(cd_frame, self._last_bboxs)
        if self._verbose:
            print(f"  Change dect: {result[0]}")
        if result[0]:
            self._use_dnn = True

        return self._last_bboxs
