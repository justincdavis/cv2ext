# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import cv2ext
import hypothesis.strategies as st
from hypothesis import given

from ..helpers import wrapper, wrapper_jit


@wrapper
@given(
    data=st.lists(
        st.tuples(
            st.lists(
                st.tuples(
                    st.tuples(
                        st.integers(min_value=0),
                        st.integers(min_value=0),
                        st.integers(min_value=1),
                        st.integers(min_value=1),
                    ),
                    st.integers(min_value=0, max_value=99),
                    st.floats(min_value=0.0, max_value=1.0),
                ),
                min_size=0,
            ),
            st.lists(
                st.tuples(
                    st.tuples(
                        st.integers(min_value=0),
                        st.integers(min_value=0),
                        st.integers(min_value=1),
                        st.integers(min_value=1),
                    ),
                    st.integers(min_value=0, max_value=99),
                ),
                min_size=0,
            ),
        ),
        min_size=1,
    ),
)
def test_bounds(data):
    bboxes, gt = [], []
    for bbox, label in data:
        bboxes.append(bbox)
        gt.append(label)
    ap = cv2ext.bboxes.mean_ap(bboxes, gt, num_classes=100)
    assert 0 <= ap <= 1


@wrapper_jit
@given(
    data=st.lists(
        st.tuples(
            st.lists(
                st.tuples(
                    st.tuples(
                        st.integers(min_value=0),
                        st.integers(min_value=0),
                        st.integers(min_value=1),
                        st.integers(min_value=1),
                    ),
                    st.integers(min_value=0, max_value=9),
                    st.floats(min_value=0.0, max_value=1.0),
                ),
                min_size=0,
            ),
            st.lists(
                st.tuples(
                    st.tuples(
                        st.integers(min_value=0),
                        st.integers(min_value=0),
                        st.integers(min_value=1),
                        st.integers(min_value=1),
                    ),
                    st.integers(min_value=0, max_value=9),
                ),
                min_size=0,
            ),
        ),
        min_size=1,
    ),
)
def test_bounds_jit(data):
    bboxes, gt = [], []
    for bbox, label in data:
        bboxes.append(bbox)
        gt.append(label)
    ap = cv2ext.bboxes.mean_ap(bboxes, gt, num_classes=10)
    assert 0 <= ap <= 1
