# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing IoU calculation for bounding boxes."""

from __future__ import annotations

from cv2ext import bboxes

if __name__ == "__main__":
    a = (0, 0, 10, 10)
    b = (5, 5, 10, 10)
    iou = bboxes.iou(a, b)
    print(iou)
