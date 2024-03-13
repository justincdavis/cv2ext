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

import contextlib

import cv2ext
import pybboxes
import hypothesis.strategies as st
from hypothesis import given

from ..helpers import wrapper, wrapper_jit


@wrapper
def test_no_overlap():
    a = (0, 0, 10, 10)
    b = (10, 10, 20, 20)
    iou = cv2ext.bboxes.iou(a, b)
    pyb_iou = pybboxes.BoundingBox(*a).iou(pybboxes.BoundingBox(*b))
    assert iou == 0.0
    assert iou == pyb_iou


@wrapper
def test_complete_overlap():
    a = (0, 0, 4, 4)
    b = (0, 0, 4, 4)
    iou = cv2ext.bboxes.iou(a, b)
    pyb_iou = pybboxes.BoundingBox(*a).iou(pybboxes.BoundingBox(*b))
    assert iou == 1.0
    assert iou == pyb_iou


@wrapper
def test_partial_overlap():
    a = (0, 0, 10, 10)
    b = (5, 5, 10, 10)
    iou = cv2ext.bboxes.iou(a, b)
    pyb_iou = pybboxes.BoundingBox(*a).iou(pybboxes.BoundingBox(*b))
    assert iou == 0.25
    assert iou == pyb_iou


@wrapper
@given(
    bbox1=st.tuples(st.integers(), st.integers(), st.integers(), st.integers()),
    bbox2=st.tuples(st.integers(), st.integers(), st.integers(), st.integers()),
)
def test_bounds(bbox1, bbox2):
    iou = cv2ext.bboxes.iou(bbox1, bbox2)
    assert 0 <= iou <= 1


@wrapper
@given(
    data1=st.tuples(
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=1, max_value=1000),
        st.integers(min_value=1, max_value=1000),
    ),
    data2=st.tuples(
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=1, max_value=1000),
        st.integers(min_value=1, max_value=1000),
    ),
)
def test_parity(data1, data2):
    bbox1 = data1[0], data1[1], data1[0] + data1[2], data1[1] + data1[3]
    bbox2 = data2[0], data2[1], data2[0] + data2[2], data2[1] + data2[3]
    iou = cv2ext.bboxes.iou(bbox1, bbox2)
    pyb_iou = pybboxes.BoundingBox(*bbox1).iou(pybboxes.BoundingBox(*bbox2))
    assert iou == pyb_iou


@wrapper_jit
def test_no_overlap_jit():
    a = (0, 0, 10, 10)
    b = (10, 10, 20, 20)
    iou = cv2ext.bboxes.iou(a, b)
    pyb_iou = pybboxes.BoundingBox(*a).iou(pybboxes.BoundingBox(*b))
    assert iou == 0.0
    assert iou == pyb_iou


@wrapper_jit
def test_complete_overlap_jit():
    a = (0, 0, 4, 4)
    b = (0, 0, 4, 4)
    iou = cv2ext.bboxes.iou(a, b)
    pyb_iou = pybboxes.BoundingBox(*a).iou(pybboxes.BoundingBox(*b))
    assert iou == 1.0
    assert iou == pyb_iou


@wrapper_jit
def test_partial_overlap_jit():
    a = (0, 0, 10, 10)
    b = (5, 5, 10, 10)
    iou = cv2ext.bboxes.iou(a, b)
    pyb_iou = pybboxes.BoundingBox(*a).iou(pybboxes.BoundingBox(*b))
    assert iou == 0.25
    assert iou == pyb_iou


@wrapper_jit
@given(
    bbox1=st.tuples(st.integers(), st.integers(), st.integers(), st.integers()),
    bbox2=st.tuples(st.integers(), st.integers(), st.integers(), st.integers()),
)
def test_bounds_jit(bbox1, bbox2):
    iou = cv2ext.bboxes.iou(bbox1, bbox2)
    assert 0 <= iou <= 1


@wrapper_jit
@given(
    data1=st.tuples(
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=1, max_value=1000),
        st.integers(min_value=1, max_value=1000),
    ),
    data2=st.tuples(
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=1, max_value=1000),
        st.integers(min_value=1, max_value=1000),
    ),
)
def test_parity_jit(data1, data2):
    bbox1 = data1[0], data1[1], data1[0] + data1[2], data1[1] + data1[3]
    bbox2 = data2[0], data2[1], data2[0] + data2[2], data2[1] + data2[3]
    iou = cv2ext.bboxes.iou(bbox1, bbox2)
    pyb_iou = pybboxes.BoundingBox(*bbox1).iou(pybboxes.BoundingBox(*bbox2))
    assert iou == pyb_iou
