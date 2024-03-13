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
