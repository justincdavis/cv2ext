# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

import cv2
import cv2ext

from ..helpers import wrapper_jit


@wrapper_jit
def test_match_multiple_jit():
    template = cv2.imread(str(Path("data") / "template.png"))
    image = cv2.imread(str(Path("data") / "pictograms.png"))

    output = cv2ext.template.match_multiple(image, template, threshold=0.99)

    assert output is not None
    assert len(output) == 1
    output = output[0]
    assert len(output) == 4


@wrapper_jit
def test_match_multiple_threshold_jit():
    template = cv2.imread(str(Path("data") / "template.png"))
    image = cv2.imread(str(Path("data") / "pictograms.png"))

    num_matches = []
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
        output = cv2ext.template.match_multiple(image, template, threshold=thresh)
        assert output is not None
        assert isinstance(output, list)
        assert len(output) > 0
        num_matches.append(len(output))

    # check numbers of matches is equal or decreasing as we increase the threshold
    for i in range(len(num_matches) - 1):
        assert num_matches[i] >= num_matches[i + 1]


@wrapper_jit
def test_match_multiple_max_thresh_jit():
    template = cv2.imread(str(Path("data") / "template.png"))
    image = cv2.imread(str(Path("data") / "pictograms.png"))

    output = cv2ext.template.match_multiple(image, template, threshold=0.999)

    assert output is not None
    assert len(output) == 1
    assert output[0] == (308, 308, 458, 454)


@wrapper_jit
def test_match_multiple_above_max_thresh_jit():
    template = cv2.imread(str(Path("data") / "template.png"))
    image = cv2.imread(str(Path("data") / "pictograms.png"))

    output = cv2ext.template.match_multiple(image, template, threshold=1.1)

    assert output is not None
    assert len(output) == 0
