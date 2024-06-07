# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from cv2ext import Fourcc


def test_all_ints():
    values = [e.value for e in Fourcc]
    for v in values:
        assert isinstance(v, int)
        assert v >= 0

def test_all_unique():
    values = [e.value for e in Fourcc]
    num_vals = len(values)
    set_vals = set()
    for v in values:
        set_vals.add(v)
    assert len(set_vals) == num_vals
