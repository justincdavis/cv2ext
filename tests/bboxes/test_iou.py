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
from hypothesis import given
import hypothesis.strategies as st

from ..helpers import wrapper, wrapper_jit


@wrapper
def test_no_overlap():
    a = (0, 0, 10, 10)
    b = (10, 10, 20, 20)
    iou = cv2ext.bboxes.iou(a, b)
    assert iou == 0.0


@wrapper
def test_complete_overlap():
    a = (0, 0, 4, 4)
    b = (0, 0, 4, 4)
    iou = cv2ext.bboxes.iou(a, b)
    assert iou == 1.0


@wrapper
def test_partial_overlap():
    a = (0, 0, 10, 10)
    b = (5, 5, 10, 10)
    iou = cv2ext.bboxes.iou(a, b)
    assert iou == 0.25

@given(
    bbox1=st.tuples(st.integers(), st.integers(), st.integers(), st.integers()),
    bbox2=st.tuples(st.integers(), st.integers(), st.integers(), st.integers()),
)
@wrapper
def test_bounds(bbox1, bbox2):
    iou = cv2ext.bboxes.iou(bbox1, bbox2)
    assert 0 <= iou <= 1

@wrapper_jit
def test_no_overlap_jit():
    a = (0, 0, 10, 10)
    b = (10, 10, 20, 20)
    iou = cv2ext.bboxes.iou(a, b)
    assert iou == 0.0


@wrapper_jit
def test_complete_overlap_jit():
    a = (0, 0, 4, 4)
    b = (0, 0, 4, 4)
    iou = cv2ext.bboxes.iou(a, b)
    assert iou == 1.0


@wrapper_jit
def test_partial_overlap_jit():
    a = (0, 0, 10, 10)
    b = (5, 5, 10, 10)
    iou = cv2ext.bboxes.iou(a, b)
    assert iou == 0.25


@given(
    bbox1=st.tuples(st.integers(), st.integers(), st.integers(), st.integers()),
    bbox2=st.tuples(st.integers(), st.integers(), st.integers(), st.integers()),
)
@wrapper_jit
def test_bounds_jit(bbox1, bbox2):
    iou = cv2ext.bboxes.iou(bbox1, bbox2)
    assert 0 <= iou <= 1
