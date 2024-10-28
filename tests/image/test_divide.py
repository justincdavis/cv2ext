# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import numpy as np
from cv2ext.image import divide


def test_divide_size_basic():
    rng = np.random.default_rng()
    image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)

    images, offsets = divide(image, 10, 10)
    assert len(images) == 10
    for row in images:
        assert len(row) == 10
        for img in row:
            assert img.shape == (10, 10, 3)


def test_divide_size_padding():
    rng = np.random.default_rng()
    image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)

    images, offsets = divide(image, 10, 10, padding=2)
    assert len(images) == 10
    for row in images:
        assert len(row) == 10
        for img in row:
            # can be in middle (full padding) an edge (three sides get padding) or corner (two sides get padding)
            assert img.shape == (14, 14, 3) or img.shape == (12, 14, 3) or img.shape == (14, 12, 3) or img.shape == (12, 12, 3)


def test_divide_size_ratio():
    rng = np.random.default_rng()
    image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)

    images, offsets = divide(image, 10, 10, overlap_ratio=0.1)
    assert len(images) == 10
    for row in images:
        assert len(row) == 10
        for img in row:
            # can be in middle (full overlap) an edge (three sides get overlap) or corner (two sides get overlap)
            assert img.shape == (12, 12, 3) or img.shape == (12, 11, 3) or img.shape == (11, 12, 3) or img.shape == (11, 11, 3)


def test_divide_only_cols():
    rng = np.random.default_rng()
    image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)

    images, offsets = divide(image, 1, 10)
    assert len(images) == 1
    assert len(images[0]) == 10
    for img in images[0]:
        assert img.shape == (100, 10, 3)


def test_divide_only_rows():
    rng = np.random.default_rng()
    image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)

    images, offsets = divide(image, 10, 1)
    assert len(images) == 10
    for img in images:
        assert len(img) == 1
        assert img[0].shape == (10, 100, 3)
