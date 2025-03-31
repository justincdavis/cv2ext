# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ._scheduler import ShiftScheduler

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self

_log = logging.getLogger(__name__)


class Shift:
    """
    The SHIFT methodology as described in the DATE24 paper.

    Paper link: https://arxiv.org/abs/2402.07415
    """

    def __init__(
        self: Self,
        data_dir: Path | str,
        model_data: list[
            tuple[
                str,
                Callable[
                    [np.ndarray],
                    list[tuple[tuple[int, int, int, int], float, int]],
                ],
            ]
        ],
        cost_threshold: float = 1.0,
        accuracy_threshold: float = 0.5,
        momentum: int = 10,
        knobs: dict[str, float] | None = None,
        metric: str = "iou",
    ) -> None:
        """
        Initialize the SHIFT methodology.

        Parameters
        ----------
        data_dir : Path, str
            The directory containing the model statistics.
            This is created through the model characterization process.
        model_data: list[tuple[str, Callable[[np.ndarray], list[tuple[tuple[int, int, int, int], float, int]]]]]
            A list of tuples of model name, function to create the model
        cost_threshold : float, optional
            The cost threshold to use when determining which models are
            potential candidates for improving or maintaining the accuracy.
            The default is 0.5.
            The higher the value the more models are included for candidacy.
            The lower the value the less models are included for candidacy.
        accuracy_threshold : float, optional
            The accuracy to target when choosing which models to schedule.
            If the predicted accuracy is below this threshold, then the model
            will not be scheduled.
        momentum : int, optional
            The number of previous accuracy estimates to use when determining
            the current accuracy of a model.
            The default is 10.
            The smaller the momentum the more reactive the algorithm is to
            changes in the accuracy.
            The larger the momentum the less reactive the algorithm is to
            changes in the accuracy.
        knobs : dict[str, float], optional
            The knobs to use when running the heuristic.
            The default is None.
            The knobs dict (if provided) should contain, accuracy, latency,
            and energy as keys. All values should be floats.
        metric : str, optional
            The metric to utilize for accuracy calculations.
            The options are: ['iou', 'recall']

        Raises
        ------
        ValueError
            If the metric is not valid

        """
        metric_options = ["iou", "recall"]
        if metric not in metric_options:
            err_msg = f"Metric, {metric} not in {metric_options}"
            raise ValueError(err_msg)
        self._metric = metric

        # create the scheduler
        self._scheduler = ShiftScheduler(
            data_dir=data_dir,
            cost_threshold=cost_threshold,
            accuracy_threshold=accuracy_threshold,
            momentum=momentum,
            knobs=knobs,
        )

        # store models as a dict[str, Callable[[np.ndarray], list[tuple[tuple[int, int, int, int], float, int]]]]
        self._models: dict[
            str,
            Callable[[np.ndarray], list[tuple[tuple[int, int, int, int], float, int]]],
        ] = dict(model_data)

        # general tracking info
        most_accurate_model: str | None = None
        highest_accuracy: float = 0.0
        # load data from model characterization
        for root, dirs, _ in os.walk(str(data_dir)):
            for directory in dirs:
                dirpath = Path(root) / directory
                with Path.open(dirpath / f"{directory}.json") as f:
                    data = json.load(f)
                    try:
                        metric_mean = float(data[self._metric]["mean"])
                    except KeyError as e:
                        err_msg = f"Could not find entry for {self._metric} in data for {dirpath.stem}. "
                        err_msg += "Consider performing characterization again."
                        raise KeyError(err_msg) from e
                    if most_accurate_model is None or metric_mean > highest_accuracy:
                        most_accurate_model = directory
                        highest_accuracy = metric_mean
        if most_accurate_model is None:
            err_msg = "No models were found in the stats_dir."
            raise ValueError(err_msg)
        self._last_model: str = most_accurate_model
        self._main_model = most_accurate_model

        _log.debug(f"SHIFT: Most accurate model, {most_accurate_model}")

    @property
    def scheduler(self: Self) -> ShiftScheduler:
        """
        Get the scheduler instance.

        Returns
        -------
        ShiftScheduler
            The scheduler instance.

        """
        return self._scheduler

    def run(
        self: Self,
        image: np.ndarray,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """
        Call the SHIFT methodology and perform the actual scheduling.

        Parameters
        ----------
        image : np.ndarray
            The most recent image.

        Returns
        -------
        list[tuple[tuple[int, int, int, int], float, int]]
            The detections returned by the detector.

        """
        dets = self._models[self._last_model](image)
        bboxes = []
        scores = []
        for bbox, score, _ in dets:
            bboxes.append(bbox)
            scores.append(score)

        new_model = self._scheduler.run(self._last_model, image, bboxes, scores)

        self._last_model = new_model

        return dets

    def __call__(
        self: Self,
        image: np.ndarray,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """
        Call the SHIFT methodology and perform the actual scheduling.

        Parameters
        ----------
        image : np.ndarray
            The most recent image.

        Returns
        -------
        list[tuple[tuple[int, int, int, int], float, int]]
            The detections returned by the detector.

        """
        return self.run(image)

    def reset(self: Self) -> None:
        """Reset the state of the scheduler."""
        self._last_model = self._main_model
        self._scheduler.reset()
