# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

import cv2
import cv2ext

from ..helpers import wrapper


@wrapper
def test_match_single():
    template = cv2.imread(str(Path("data") / "template.png"))
    image = cv2.imread(str(Path("data") / "pictograms.png"))

    output = cv2ext.template.match_single(image, template)

    assert output is not None
    assert len(output) == 4
    assert output == (308, 308, 458, 454)
