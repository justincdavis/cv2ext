# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

import cv2ext
import hypothesis.strategies as st
from hypothesis import given

from ..helpers import wrapper, wrapper_jit


@wrapper
def test_one_box():
    boxes = [((0, 0, 10, 10), 1, 0.5)]
    sboxes = cv2ext.bboxes.nms(boxes, 0.5)
    assert boxes == sboxes


@wrapper
def test_overlapping_boxes_1():
    boxes = [
        ((0, 0, 10, 10), 1, 0.9),
        ((1, 1, 9, 9), 1, 0.8),
    ]
    sboxes = cv2ext.bboxes.nms(boxes, 0.5)
    assert len(sboxes) == 1
    assert sboxes[0] == boxes[0]


@wrapper
def test_overlapping_boxes_2():
    boxes = [
        ((0, 0, 10, 10), 1, 0.9),
        ((1, 1, 9, 9), 1, 0.8),
        ((2, 2, 8, 8), 1, 0.7),
    ]
    sboxes = cv2ext.bboxes.nms(boxes, 0.5)
    assert len(sboxes) == 1
    assert sboxes[0] == boxes[0]


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
            st.integers(min_value=0, max_value=100),
            st.floats(0.0, 1.0),
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
    boxes = [((0, 0, 10, 10), 1, 0.5)]
    sboxes = cv2ext.bboxes.nms(boxes, 0.5)
    assert boxes == sboxes


@wrapper_jit
def test_overlapping_boxes_1_jit():
    boxes = [
        ((0, 0, 10, 10), 1, 0.9),
        ((1, 1, 9, 9), 1, 0.8),
    ]
    sboxes = cv2ext.bboxes.nms(boxes, 0.5)
    assert len(sboxes) == 1
    assert sboxes[0] == boxes[0]


@wrapper_jit
def test_overlapping_boxes_2_jit():
    boxes = [
        ((0, 0, 10, 10), 1, 0.9),
        ((1, 1, 9, 9), 1, 0.8),
        ((2, 2, 8, 8), 1, 0.7),
    ]
    sboxes = cv2ext.bboxes.nms(boxes, 0.5)
    assert len(sboxes) == 1
    assert sboxes[0] == boxes[0]


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
            st.integers(min_value=0, max_value=100),
            st.floats(0.0, 1.0),
        ),
        min_size=1,
    ),
)
def test_size_constraints_jit(bboxes):
    starting = len(bboxes)
    sboxes = cv2ext.bboxes.nms(bboxes, 0.5)
    assert len(sboxes) >= 1
    assert len(sboxes) <= starting
