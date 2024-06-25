# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from enum import Enum

from .cv_trackers import (
    BoostingTracker,
    CSRTTracker,
    KCFTracker,
    MedianFlowTracker,
    MILTracker,
    MOSSETracker,
    TLDTracker,
)
from .trackers import KLTTracker


class TrackerType(Enum):
    """An enumeration of the available tracker types."""

    BOOSTING = BoostingTracker
    CSRT = CSRTTracker
    KCF = KCFTracker
    MEDIAN_FLOW = MedianFlowTracker
    MIL = MILTracker
    MOSSE = MOSSETracker
    TLD = TLDTracker
    KLT = KLTTracker
