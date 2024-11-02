# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import numpy as np
from cv2ext.image import rescale


def test_rescale_all_gray():
    gray_img = np.full((10, 10), 114, dtype=np.uint8)
    rescaled = rescale(gray_img, (0.0, 1.0))
    expected_value = 114 / 255
    np.testing.assert_allclose(rescaled, expected_value, rtol=1e-6)
    

def test_rescale_black_and_white():
    bw_img = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    rescaled = rescale(bw_img, (0.0, 1.0))
    expected = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(rescaled, expected, rtol=1e-6)
    

def test_rescale_neg_one_to_one():
    bw_img = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    rescaled = rescale(bw_img, (-1.0, 1.0))
    expected = np.array([[-1.0, 1.0], [1.0, -1.0]])
    np.testing.assert_allclose(rescaled, expected, rtol=1e-6)
    

def test_rescale_non_zero_min():
    bw_img = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    rescaled = rescale(bw_img, (0.5, 1.0))
    expected = np.array([[0.5, 1.0], [1.0, 0.5]])
    np.testing.assert_allclose(rescaled, expected, rtol=1e-6)
    

def test_all_zeros():
    zero_img = np.zeros((5, 5), dtype=np.uint8)
    rescaled = rescale(zero_img, (0.0, 1.0))
    np.testing.assert_allclose(rescaled, 0.0, rtol=1e-6)
    

def test_all_max():
    max_img = np.full((5, 5), 255, dtype=np.uint8)
    rescaled = rescale(max_img, (0.0, 1.0))
    np.testing.assert_allclose(rescaled, 1.0, rtol=1e-6)
    
    
def test_odd_shapes():
    rect_img = np.full((3, 5), 128, dtype=np.uint8)
    rescaled = rescale(rect_img, (0.0, 1.0))
    expected_value = 128 / 255
    np.testing.assert_allclose(rescaled, expected_value, rtol=1e-6)
