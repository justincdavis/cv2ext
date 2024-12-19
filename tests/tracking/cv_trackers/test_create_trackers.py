# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib

from cv2ext.tracking.cv_trackers import BoostingTracker, CSRTTracker, KCFTracker, MedianFlowTracker, MILTracker, MOSSETracker, TLDTracker


def test_create_boosting_tracker():
    with contextlib.suppress(ImportError):
        tracker = BoostingTracker()
        assert tracker is not None


def test_create_csrt_tracker():
    with contextlib.suppress(ImportError):
        tracker = CSRTTracker()
        assert tracker is not None


def test_create_kcf_tracker():
    with contextlib.suppress(ImportError):
        tracker = KCFTracker()
        assert tracker is not None


def test_create_median_flow_tracker():
    with contextlib.suppress(ImportError):
        tracker = MedianFlowTracker()
        assert tracker is not None


def test_create_mil_tracker():
    with contextlib.suppress(ImportError):
        tracker = MILTracker()
        assert tracker is not None


def test_create_mosse_tracker():
    with contextlib.suppress(ImportError):
        tracker = MOSSETracker()
        assert tracker is not None


def test_create_tld_tracker():
    with contextlib.suppress(ImportError):
        tracker = TLDTracker()
        assert tracker is not None
