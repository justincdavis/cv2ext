# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import cv2ext


def test_zero_len_match():
    bboxes1 = []
    bboxes2 = []

    try:
        cv2ext.bboxes.match(bboxes1, bboxes2)
    except ValueError:
        return True

    return False

def test_basic_1_match():
    bboxes1 = [(0, 0, 10, 10)]
    bboxes2 = [(0, 0, 9, 9)]

    matches = cv2ext.bboxes.match(bboxes1, bboxes2)
    assert len(matches) == 1


def test_basic_2_match():
    bboxes1 = [(0, 0, 10, 10)]
    bboxes2 = [(0, 0, 9, 9), (0, 0, 10, 10)]

    matches = cv2ext.bboxes.match(bboxes1, bboxes2)
    assert len(matches) == 1

    assert matches[0] == (0, 1)


if __name__ == "__main__":
    test_basic_1_match()
