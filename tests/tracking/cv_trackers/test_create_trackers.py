# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from cv2ext.tracking.cv_trackers import BoostingTracker, CSRTTracker, KCFTracker, MedianFlowTracker, MILTracker, MOSSETracker, TLDTracker


def test_create_boosting_tracker():
    tracker = BoostingTracker()
    assert tracker is not None


def test_create_csrt_tracker():
    tracker = CSRTTracker()
    assert tracker is not None


def test_create_kcf_tracker():
    tracker = KCFTracker()
    assert tracker is not None


def test_create_median_flow_tracker():
    tracker = MedianFlowTracker()
    assert tracker is not None


def test_create_mil_tracker():
    tracker = MILTracker()
    assert tracker is not None


def test_create_mosse_tracker():
    tracker = MOSSETracker()
    assert tracker is not None


def test_create_tld_tracker():
    tracker = TLDTracker()
    assert tracker is not None
