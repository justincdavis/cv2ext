# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from cv2ext.bboxes import score_bbox, score_bboxes
from hypothesis import given
import hypothesis.strategies as st

from ..helpers import wrapper, wrapper_jit


@wrapper
def test_same_box():
    assert score_bbox((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


@wrapper
@given(
    target_bbox=st.tuples(
                st.integers(min_value=0, max_value=9),
                st.integers(min_value=0, max_value=9),
                st.integers(min_value=10),
                st.integers(min_value=10),
            ),
    pred_bbox=st.tuples(
                st.integers(min_value=0, max_value=9),
                st.integers(min_value=0, max_value=9),
                st.integers(min_value=10),
                st.integers(min_value=10),
            ),
    )
def test_random_boxes_bounds(target_bbox, pred_bbox):
    score = score_bbox(target_bbox, pred_bbox)
    assert 0.0 <= score <= 1.0


@wrapper_jit
def test_same_box_jit():
    assert score_bbox((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


@wrapper_jit
@given(
    target_bbox=st.tuples(
                st.integers(min_value=0, max_value=9),
                st.integers(min_value=0, max_value=9),
                st.integers(min_value=10),
                st.integers(min_value=10),
            ),
    pred_bbox=st.tuples(
                st.integers(min_value=0, max_value=9),
                st.integers(min_value=0, max_value=9),
                st.integers(min_value=10),
                st.integers(min_value=10),
            ),
    )
def test_random_boxes_bounds_jit(target_bbox, pred_bbox):
    score = score_bbox(target_bbox, pred_bbox)
    assert 0.0 <= score <= 1.0


@wrapper
@given(
    target_bbox=st.tuples(
                st.integers(min_value=0, max_value=9),
                st.integers(min_value=0, max_value=9),
                st.integers(min_value=10),
                st.integers(min_value=10),
            ),
    pred_bboxs=st.lists(
        st.tuples(
                st.integers(min_value=0, max_value=9),
                st.integers(min_value=0, max_value=9),
                st.integers(min_value=10),
                st.integers(min_value=10),
            ),
        min_size=1,
    ),
    )
def test_score_bboxes_random_bounds(target_bbox, pred_bboxs):
    scores = score_bboxes(target_bbox, pred_bboxs)
    for score in scores:
        assert 0.0 <= score <= 1.0


@wrapper_jit
@given(
    target_bbox=st.tuples(
                st.integers(min_value=0, max_value=9),
                st.integers(min_value=0, max_value=9),
                st.integers(min_value=10),
                st.integers(min_value=10),
            ),
    pred_bboxs=st.lists(
        st.tuples(
                st.integers(min_value=0, max_value=9),
                st.integers(min_value=0, max_value=9),
                st.integers(min_value=10),
                st.integers(min_value=10),
            ),
        min_size=1,
    ),
    )
def test_score_bboxes_random_bounds_jit(target_bbox, pred_bboxs):
    scores = score_bboxes(target_bbox, pred_bboxs)
    for score in scores:
        assert 0.0 <= score <= 1.0
