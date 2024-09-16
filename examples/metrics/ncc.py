# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing ncc computation between two images."""

from __future__ import annotations

import numpy as np

import cv2ext

if __name__ == "__main__":
    rng = np.random.Generator(np.random.PCG64())
    image1 = rng.random((100, 100))
    image2 = rng.random((100, 100))

    ncc = cv2ext.metrics.ncc(image1, image2)
    print(ncc)
