# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2

from cv2ext.tracking.trackers._klt import KLTMultiTracker

from ._change import ChangeDetector

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np
    from typing_extensions import Self

_log = logging.getLogger(__name__)


class Marlin:
    def __init__(
        self: Self,
        detector: Callable[
            [np.ndarray],
            list[tuple[tuple[int, int, int, int], float, int]],
        ],
        forest: str | Path,
        ncc_threshold: float = 0.3,
        num_features: int = 750,
        window_size: tuple[int, int] = (15, 15),
        max_level: int = 2,
        criteria: tuple[int, int, float] = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            10,
            0.03,
        ),
        success_ratio: float = 0.9,
    ) -> None:
        """
        Create a Marlin instance.

        Parameters
        ----------
        detector : Callable[[np.ndarray], list[tuple[tuple[int, int, int, int], float, int]]]
            The call to get detections for an image.
        forest : str, Path
            The path to the saved RandomForestClassifier.
        ncc_threshold : float, optional
            The threshold for which the frame is determined to be changed.
            By default, 0.3
        num_features : int, optional
            The number of features to track.
            By default, this is set to 750.
        window_size : tuple[int, int], optional
            The size of the window used for tracking.
            By default, this is set to (15, 15).
        max_level : int, optional
            The maximum pyramid level for tracking.
            By default, this is set to 2.
        criteria : tuple[int, int, float], optional
            The criteria used for tracking.
            By default, this is set to (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03).
        success_ratio : float, optional
            The ratio of successes from the KLT tracker to number of boxes
            to determine if tracking for the entire frame was a success.
            By default, 0.9

        """
        # warning for using research method
        _log.warning("MARLIN research implementation is not a tested module.")

        self._detector = detector
        self._tracker = KLTMultiTracker(
            num_features=num_features,
            window_size=window_size,
            max_level=max_level,
            criteria=criteria,
        )
        self._change = ChangeDetector(forest)
        self._ncc_threshold = ncc_threshold
        self._success_ratio = success_ratio

        # state storage
        self._bboxes: list[tuple[tuple[int, int, int, int], float, int]] = []
        self._use_detector = True

    def run(
        self,
        frame: np.ndarray,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """
        Run Marlin on the next frame in a sequence.

        Parameters
        ----------
        frame : np.ndarray
            The next frame in a sequence.

        Returns
        -------
        list[tuple[tuple[int, int, int, int], float, int]]
            The detections

        """

        def _run_det() -> list[tuple[tuple[int, int, int, int], float, int]]:
            bboxes = self._detector(frame)
            raw_bboxes = [det[0] for det in bboxes]

            # when we run the detector, setup the tracker
            if len(bboxes) != 0:
                self._tracker.init(frame, raw_bboxes)

            # set use_detector to False since we just used it
            self._use_detector = False

            # update bboxes
            self._bboxes = bboxes

            return bboxes

        def _run_tracker() -> list[tuple[tuple[int, int, int, int], float, int]]:
            new_tracks = self._tracker.update(frame)
            new_raw_bboxes = [track[1] for track in new_tracks]
            # form the bboxes
            new_bboxes = [
                (bbox, conf, cid)
                for bbox, (_, conf, cid) in zip(new_raw_bboxes, self._bboxes)
            ]

            # we have a success if we meet the ratio
            bbox_success = [s[0] for s in new_tracks]
            success = (sum(bbox_success) / len(new_tracks)) >= self._success_ratio
            if not success:
                # if fail mark detector for use next frame
                self._use_detector = True

            # update bboxes
            self._bboxes = bboxes

            return new_bboxes

        # if state is None, then we run the detector and set it
        # if the change detector said use detector, run it
        # otherwise we simply run tracking
        bboxes = _run_det() if self._use_detector else _run_tracker()

        # run the change detector
        result = self._change(frame, bboxes)
        if result:
            self._use_detector = True

        # return bboxes
        return bboxes
