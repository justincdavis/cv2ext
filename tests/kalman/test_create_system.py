# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from cv2ext.kalman import create_system
import hypothesis.strategies as st
from hypothesis import given


@given(msize=st.integers(1, 10))
def test_basic(msize: int):
    x, f, h, p, q, r, i = create_system(msize)
    assert x.shape == (msize, 1)
    assert f.shape == (msize, msize)
    assert h.shape == (msize, msize)
    assert p.shape == (msize, msize)
    assert q.shape == (msize, msize)
    assert r.shape == (msize, msize)
    assert i.shape == (msize, msize)


@given(
    msize=st.integers(1, 5),
    nsize=st.integers(5, 10),
)
def test_asymmetric(msize: int, nsize: int):
    x, f, h, p, q, r, i = create_system(msize, nsize)
    assert x.shape == (nsize, 1)
    assert f.shape == (nsize, nsize)
    assert h.shape == (msize, nsize)
    assert p.shape == (nsize, nsize)
    assert q.shape == (nsize, nsize)
    assert r.shape == (msize, msize)
    assert i.shape == (nsize, nsize)
