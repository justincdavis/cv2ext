# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import cv2ext
import hypothesis.strategies as st
from hypothesis import given

from ..helpers import wrapper, wrapper_jit


@wrapper
def test_one_box():
    boxes = [((0, 0, 10, 10), 0.5, 1)]
    sboxes = cv2ext.bboxes.nms(boxes, 0.5)
    assert boxes == sboxes


@wrapper
def test_overlapping_boxes_1():
    boxes = [
        ((0, 0, 10, 10), 0.9, 1),
        ((1, 1, 9, 9), 0.8, 1),
    ]
    sboxes = cv2ext.bboxes.nms(boxes, 0.5)
    assert len(sboxes) == 1
    assert sboxes[0] == boxes[0]


@wrapper
def test_overlapping_boxes_2():
    boxes = [
        ((0, 0, 10, 10), 0.9, 1),
        ((1, 1, 9, 9), 0.8, 1),
        ((2, 2, 8, 8), 0.7, 1),
    ]
    sboxes = cv2ext.bboxes.nms(boxes, 0.5)
    assert len(sboxes) == 2
    assert sboxes[0] == boxes[0]

    mboxes = cv2ext.bboxes.nms(boxes, 0.25)
    assert len(mboxes) == 1
    assert mboxes[0] == boxes[0]


@wrapper
@given(
    bboxes=st.lists(
        st.tuples(
            st.tuples(
                st.integers(min_value=0),
                st.integers(min_value=0),
                st.integers(min_value=1),
                st.integers(min_value=1),
            ),
            st.floats(0.0, 1.0),
            st.integers(min_value=0, max_value=100),
        ),
        min_size=1,
    ),
)
def test_size_constraints(bboxes):
    starting = len(bboxes)
    sboxes = cv2ext.bboxes.nms(bboxes, 0.5)
    assert len(sboxes) >= 1
    assert len(sboxes) <= starting


@wrapper_jit
def test_one_box_jit():
    boxes = [((0, 0, 10, 10), 0.5, 1)]
    sboxes = cv2ext.bboxes.nms(boxes, 0.5)
    assert boxes == sboxes


@wrapper_jit
def test_overlapping_boxes_1_jit():
    boxes = [
        ((0, 0, 10, 10), 0.9, 1),
        ((1, 1, 9, 9), 0.8, 1),
    ]
    sboxes = cv2ext.bboxes.nms(boxes, 0.5)
    assert len(sboxes) == 1
    assert sboxes[0] == boxes[0]


@wrapper_jit
def test_overlapping_boxes_2_jit():
    boxes = [
        ((0, 0, 10, 10), 0.9, 1),
        ((1, 1, 9, 9), 0.8, 1),
        ((2, 2, 8, 8), 0.7, 1),
    ]
    sboxes = cv2ext.bboxes.nms(boxes, 0.5)
    assert len(sboxes) == 2
    assert sboxes[0] == boxes[0]

    mboxes = cv2ext.bboxes.nms(boxes, 0.25)
    assert len(mboxes) == 1
    assert mboxes[0] == boxes[0]


@wrapper_jit
@given(
    bboxes=st.lists(
        st.tuples(
            st.tuples(
                st.integers(min_value=0),
                st.integers(min_value=0),
                st.integers(min_value=1),
                st.integers(min_value=1),
            ),
            st.floats(0.0, 1.0),
            st.integers(min_value=0, max_value=100),
        ),
        min_size=1,
    ),
)
def test_size_constraints_jit(bboxes):
    starting = len(bboxes)
    sboxes = cv2ext.bboxes.nms(bboxes, 0.5)
    assert len(sboxes) >= 1
    assert len(sboxes) <= starting
