# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Subpackage containing tools for working with simple bounding boxes.

Functions
---------
bounding
    Get a bounding box which encloses all the given bounding boxes.
constrain
    Constrain a bounding box to be within the bounds of an image.
draw_bboxes
    Draw bounding boxes on an image.
euclidean
    Compute the euclidean distance between two bounding boxes.
iou
    Calculate the intersection over union of two bounding boxes.
ious
    Calculate the intersection over union of a set of bounding boxes.
manhattan
    Compute the manhattan distance between two bounding boxes.
mean_ap
    Calculate the mean average precision of a set of bounding boxes.
nms
    Perform non-maximum suppression on a set of bounding boxes.
score_bbox
    Score a bounding box relative to a target bbox.
score_bboxes
    Score a set of bounding boxes relative to a target bbox.
xywh_to_xyxy
    Convert bounding boxes from `(x, y, w, h)` to `(x1, y1, x2, y2)`.
xyxy_to_xywh
    Convert bounding boxes from `(x1, y1, x2, y2)` to `(x, y, w, h)`.
xyxy_to_yolo
    Convert bounding boxes from `(x1, y1, x2, y2)` to `(x, y, w, h)` in YOLO format.
xywh_to_yolo
    Convert bounding boxes from `(x, y, w, h)` to `(x, y, w, h)` in YOLO format.
yolo_to_xyxy
    Convert bounding boxes from `(x, y, w, h)` in YOLO format to `(x1, y1, x2, y2)`.
yolo_to_xywh
    Convert bounding boxes from `(x, y, w, h)` in YOLO format to `(x, y, w, h)`.

"""

from __future__ import annotations

from ._bounding import bounding
from ._constrain import constrain
from ._convert import (
    xywh_to_xyxy,
    xywh_to_yolo,
    xyxy_to_xywh,
    xyxy_to_yolo,
    yolo_to_xywh,
    yolo_to_xyxy,
)
from ._distance import euclidean, manhattan
from ._draw import draw_bboxes
from ._iou import iou, ious
from ._mean_ap import mean_ap
from ._nms import nms
from ._score import score_bbox, score_bboxes

__all__ = [
    "bounding",
    "constrain",
    "draw_bboxes",
    "euclidean",
    "iou",
    "ious",
    "manhattan",
    "mean_ap",
    "nms",
    "score_bbox",
    "score_bboxes",
    "xywh_to_xyxy",
    "xywh_to_yolo",
    "xyxy_to_xywh",
    "xyxy_to_yolo",
    "yolo_to_xywh",
    "yolo_to_xyxy",
]
