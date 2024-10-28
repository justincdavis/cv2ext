# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import numpy as np
from cv2ext.image import patch

from hypothesis import given
import hypothesis.strategies as st


def test_patch_basic_size():
    rng = np.random.default_rng()
    image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)

    images, offsets, new_size = patch(image, (10, 10))
    assert new_size == (100, 100)
    assert len(images) == len(offsets)
    assert len(images) == 100
    for img in images:
        assert img.shape == (10, 10, 3)


def test_patch_basic_padding():
    rng = np.random.default_rng()
    image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)

    images, offsets, new_size = patch(image, (60, 60), padding=50)
    assert new_size == (100, 100)
    assert len(images) == len(offsets)
    assert len(images) == 25
    for img in images:
        assert img.shape == (60, 60, 3)


def test_patch_padding_scale_up():
    rng = np.random.default_rng()
    image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)

    images, offsets, new_size = patch(image, (60, 60), padding=10)
    assert new_size == (110, 110)
    assert len(images) == len(offsets)
    assert len(images) == 4
    for img in images:
        assert img.shape == (60, 60, 3)


def test_patch_overlap_scale_up():
    rng = np.random.default_rng()
    image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)

    images, offsets, new_size = patch(image, (50, 50), overlap=0.2)
    assert new_size == (130, 130)
    assert len(images) == len(offsets)
    assert len(images) == 9
    for img in images:
        assert img.shape == (50, 50, 3)


@given(
    patch_size=st.tuples(
        st.integers(min_value=20, max_value=50),
        st.integers(min_value=20, max_value=50),
    ),
    padding=st.integers(min_value=0, max_value=10),
    dim_x=st.integers(min_value=90, max_value=110),
    dim_y=st.integers(min_value=90, max_value=110),
)
def test_patch_always_correct_size_padding(patch_size: tuple[int, int], padding: int, dim_x: int, dim_y: int):
    rng = np.random.default_rng()
    image = rng.integers(0, 255, (dim_y, dim_x, 3), dtype=np.uint8)

    width, height = patch_size

    images, offsets, new_size = patch(image, (width, height), padding=padding)
    assert new_size[0] >= dim_x
    assert new_size[1] >= dim_y
    assert len(images) == len(offsets)
    for img in images:
        assert img.shape == (height, width, 3)


@given(
    patch_size=st.tuples(
        st.integers(min_value=20, max_value=50),
        st.integers(min_value=20, max_value=50),
    ),
    overlap=st.floats(min_value=0, max_value=0.95),
    dim_x=st.integers(min_value=90, max_value=110),
    dim_y=st.integers(min_value=90, max_value=110),
)
def test_patch_always_correct_size_overlap(patch_size: tuple[int, int], overlap: float, dim_x: int, dim_y: int):
    rng = np.random.default_rng()
    image = rng.integers(0, 255, (dim_y, dim_x, 3), dtype=np.uint8)

    width, height = patch_size

    images, offsets, new_size = patch(image, (width, height), overlap=overlap)
    assert new_size[0] >= dim_x
    assert new_size[1] >= dim_y
    assert len(images) == len(offsets)
    for img in images:
        assert img.shape == (height, width, 3)
