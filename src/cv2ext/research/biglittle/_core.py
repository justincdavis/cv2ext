# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ._train import find_optimal_theta

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Self

_log = logging.getLogger(__name__)


class BigLittle:
    """
    Implementation of the BigLittle methodology.

    Paper: https://ieeexplore.ieee.org/document/7331375
    """

    def __init__(
        self: Self,
        detector1: Callable[
            [np.ndarray],
            list[tuple[tuple[int, int, int, int], float, int]],
        ],
        detector2: Callable[
            [np.ndarray],
            list[tuple[tuple[int, int, int, int], float, int]],
        ],
        theta: float = 0.5,
    ) -> None:
        """
        Create the BigLittle instance.

        Parameters
        ----------
        detector1 : Callable[[np.ndarray], list[tuple[tuple[int, int, int, int], float, int]]]
            The first detector, the smaller one
        detector2 : Callable[[np.ndarray], list[tuple[tuple[int, int, int, int], float, int]]]
            The second detector, the larger one
        theta : float, optional
            The starting confidence threshold cutoff

        """
        # warning for using research method
        _log.warning("BigLittle research implementation is not a tested module.")

        # store the detection functions
        self._det1 = detector1
        self._det2 = detector2
        self._theta = theta

    def train(
        self: Self,
        dets1: list[float] | list[tuple[tuple[int, int, int, int], float, int]],
        dets2: list[float] | list[tuple[tuple[int, int, int, int], float, int]],
        lambda_param: float = 0.1,
        loss_scale: float = 1.0,
    ) -> None:
        """
        Train the BigLittle method to generate the optimal confidence threshold.

        Parameters
        ----------
        dets1 :  list[float] | list[tuple[tuple[int, int, int, int], float, int]]
            The detections of the first detection model.
            These detections corresponed to the model with less capability.
            Must be same length as dets2
        dets2 :  list[float] | list[tuple[tuple[int, int, int, int], float, int]]
            The detections of the second detection model.
            These detections correspond to the model with more capability.
            Must be same length as dets1
        lambda_param : float, optional
            The lambda value to tune the gain/loss.
            By default, 0.1
        loss_scale : float, optional
            The scaling factor to use on the loss values.
            Formula is: loss / loss_scale
            By default, 1.0 or no scaling.

        """
        self._theta = find_optimal_theta(dets1, dets2, lambda_param, loss_scale)

    def run(
        self: Self,
        image: np.ndarray,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """
        Run the BigLittle detection method.

        Parameters
        ----------
        image : np.ndarray
            The image to run the detection method on.

        Returns
        -------
        list[tuple[tuple[int, int, int, int], float, int]]
            The detections generated by either the small or large detector

        """
        outputs1 = self._det1(image)
        conf1 = (
            float(np.mean([conf for _, conf, _ in outputs1]))
            if len(outputs1) > 0
            else 0.0
        )

        if conf1 >= self._theta:
            return outputs1

        return self._det2(image)
