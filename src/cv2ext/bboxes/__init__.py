# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Subpackage containing tools for working with simple bounding boxes.

Functions
---------
constrain
    Constrain a bounding box to be within the bounds of an image.
iou
    Calculate the intersection over union of two bounding boxes.
ious
    Calculate the intersection over union of a set of bounding boxes.
mean_ap
    Calculate the mean average precision of a set of bounding boxes.
nms
    Perform non-maximum suppression on a set of bounding boxes.
xywh_to_xyxy
    Convert bounding boxes from `(x, y, w, h)` to `(x1, y1, x2, y2)`.
xyxy_to_xywh
    Convert bounding boxes from `(x1, y1, x2, y2)` to `(x, y, w, h)`.

"""

from __future__ import annotations

from ._constrain import constrain
from ._convert import xywh_to_xyxy, xyxy_to_xywh
from ._iou import iou, ious
from ._mean_ap import mean_ap
from ._nms import nms

__all__ = ["constrain", "iou", "ious", "mean_ap", "nms", "xywh_to_xyxy", "xyxy_to_xywh"]
