# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import numpy as np


def compute_gain_loss(
    dets1: list[float],
    dets2: list[float],
    theta: float,
    lambda_param: float = 0.1,
    loss_scale: float = 1.0,
) -> float:
    """
    Compute the gain/loss for detection model switching.

    Adapted from the methodology described in the BitLittle paper.

    Parameters
    ----------
    dets1 :  list[float]
        The detections of the first detection model.
        Must be same length as dets2
    dets2 :  list[float]
        The detections of the second detection model.
        Must be same length as dets1
    theta : float
        The current theta value being tested.
    lambda_param : float, optional
        The lambda value to tune the gain/loss.
        By default, 0.1
    loss_scale : float, optional
        The scaling factor to use on the loss values.
        Formula is: loss / loss_scale
        By default, 1.0 or no scaling.

    Returns
    -------
    float
        The gain/loss between two sets of detections scores for a given theta.

    """
    num_dets: int = len(dets1)
    gain: float = 0.0
    loss: float = 0.0

    # compute gain and loss
    for i, score_1 in enumerate(dets1):
        if score_1 >= theta:
            gain += 1.0
            score_2 = dets2[i]
            loss += score_2 - score_1

    # compute combined value
    gain /= num_dets
    loss /= loss_scale
    return gain - lambda_param * loss


def find_optimal_theta(
    dets1: list[float] | list[tuple[tuple[int, int, int, int], float, int]],
    dets2: list[float] | list[tuple[tuple[int, int, int, int], float, int]],
    lambda_param: float = 0.1,
    loss_scale: float = 1.0,
) -> float:
    """
    Compute the optimal theta (confidence score) to swap models on.

    Assumption is that dets1 correspond to detections from the model with
    lesser capabiltities.
    Adapted from the methodology described in the BitLittle paper.

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

    Returns
    -------
    float
        The optimal theta (confidence score) to swap between.

    Raises
    ------
    ValueError
        If the detection data is not the same length

    """
    # ensure same length
    if len(dets1) != len(dets2):
        err_msg = f"Cannot compute gain/loss, size of detections is not equal, {len(dets1)} != {len(dets2)}"
        raise ValueError(err_msg)

    # ensure list[float]
    dets1_cleaned: list[float] = []
    for det1 in dets1:
        if isinstance(det1, tuple):
            _, score_1, _ = det1
        else:
            score_1 = det1
        dets1_cleaned.append(score_1)
    dets2_cleaned: list[float] = []
    for det2 in dets2:
        if isinstance(det2, tuple):
            _, score_2, _ = det2
        else:
            score_2 = det2
        dets2_cleaned.append(score_2)

    # store best theta and value
    best_theta, best_value = 0, -float("inf")
    for theta in np.linspace(0, 1, 100):  # theta values from 0 to 1
        value = compute_gain_loss(
            dets1_cleaned,
            dets2_cleaned,
            theta,
            lambda_param,
            loss_scale,
        )
        if value > best_value:
            best_theta, best_value = theta, value
    return best_theta
