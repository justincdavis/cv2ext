# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from cv2ext.bboxes import euclidean, manhattan

from hypothesis import given
import hypothesis.strategies as st


def test_euclidean_same():
    bbox1 = (0, 0, 10, 10)
    bbox2 = (0, 0, 10, 10)
    assert euclidean(bbox1, bbox2) == 0.0


def test_manhattan_same():
    bbox1 = (0, 0, 10, 10)
    bbox2 = (0, 0, 10, 10)
    assert manhattan(bbox1, bbox2) == 0.0


@given(
    bbox1=st.tuples(
        st.integers(min_value=0, max_value=9),
        st.integers(min_value=0, max_value=9),
        st.integers(min_value=10),
        st.integers(min_value=10),
    ),
    bbox2=st.tuples(
        st.integers(min_value=0, max_value=9),
        st.integers(min_value=0, max_value=9),
        st.integers(min_value=10),
        st.integers(min_value=10),
    ),
)
def test_euclidean_always_postive(bbox1, bbox2):
    assert euclidean(bbox1, bbox2) >= 0.0


@given(
    bbox1=st.tuples(
        st.integers(min_value=0, max_value=9),
        st.integers(min_value=0, max_value=9),
        st.integers(min_value=10),
        st.integers(min_value=10),
    ),
    bbox2=st.tuples(
        st.integers(min_value=0, max_value=9),
        st.integers(min_value=0, max_value=9),
        st.integers(min_value=10),
        st.integers(min_value=10),
    ),
)
def test_manhattan_always_postive(bbox1, bbox2):
    assert manhattan(bbox1, bbox2) >= 0.0


def test_manhattan_simple_only_x():
    bbox1 = (0, 0, 2, 2)  # center is 1, 1
    bbox2 = (2, 0, 4, 2)  # center is 3, 1
    assert manhattan(bbox1, bbox2) == 2.0


def test_manhattan_simple_only_y():
    bbox1 = (0, 0, 2, 2)  # center is 1, 1
    bbox2 = (0, 2, 2, 4)  # center is 1, 3
    assert manhattan(bbox1, bbox2) == 2.0


def test_manhattan_simple():
    bbox1 = (0, 0, 2, 2)  # center is 1, 1
    bbox2 = (2, 2, 4, 4)  # center is 3, 3
    assert manhattan(bbox1, bbox2) == 4.0


def test_euclidean_simple_only_x():
    bbox1 = (0, 0, 2, 2)
    bbox2 = (2, 0, 4, 2)
    assert euclidean(bbox1, bbox2) == 2.0


def test_euclidean_simple_only_y():
    bbox1 = (0, 0, 2, 2)
    bbox2 = (0, 2, 2, 4)
    assert euclidean(bbox1, bbox2) == 2.0


def test_euclidean_simple():
    bbox1 = (0, 0, 2, 2)
    bbox2 = (2, 2, 4, 4)
    assert euclidean(bbox1, bbox2) == 2.8284271247461903


@given(
    bbox1=st.tuples(
        st.integers(min_value=0, max_value=9),
        st.integers(min_value=0, max_value=9),
        st.integers(min_value=10),
        st.integers(min_value=10),
    ),
    bbox2=st.tuples(
        st.integers(min_value=0, max_value=9),
        st.integers(min_value=0, max_value=9),
        st.integers(min_value=10),
        st.integers(min_value=10),
    ),
)
def test_euclidean_always_less_than_equal_manhattan(bbox1, bbox2):
    assert euclidean(bbox1, bbox2) <= manhattan(bbox1, bbox2)
