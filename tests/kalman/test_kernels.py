# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import numpy as np
from cv2ext.kalman import kalman_predict_kernel, kalman_update_kernel, create_system
import hypothesis.strategies as st
from hypothesis import given


@given(msize=st.integers(1, 10))
def test_predict_kernel_no_crash(msize: int):
    x, f, h, p, q, r, i = create_system(msize)

    for _ in range(10):
        x, p = kalman_predict_kernel(x, f, p, q)


@given(
    msize=st.integers(1, 5),
    nsize=st.integers(5, 10),
)
def test_predict_kernel_asymmetric_no_crash(msize: int, nsize: int):
    x, f, h, p, q, r, i = create_system(msize, nsize)

    for _ in range(10):
        x, p = kalman_predict_kernel(x, f, p, q)


@given(msize=st.integers(1, 10))
def test_update_kernel_no_crash(msize: int):
    x, f, h, p, q, r, i = create_system(msize)

    for _ in range(10):
        z = np.zeros((4, 1))
        x, p = kalman_update_kernel(z, x, h, p, r, i)


@given(
    msize=st.integers(1, 5),
    nsize=st.integers(5, 10),
)
def test_update_kernel_asymmetric_no_crash(msize: int, nsize: int):
    x, f, h, p, q, r, i = create_system(msize, nsize)

    for _ in range(10):
        z = np.zeros((4, 1))
        x, p = kalman_update_kernel(z, x, h, p, r, i)
