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
resize
    Resize a bounding box based on one image size to another.
score_bbox
    Score a bounding box relative to a target bbox.
score_bboxes
    Score a set of bounding boxes relative to a target bbox.
valid
    Check if a bounding box is valid.
xyxy_to_nxyxy
    Convert bounding boxes from `(x1, y1, x2, y2)` to normalized `(x1, y1, x2, y2)`.
xyxy_to_xywh
    Convert bounding boxes from `(x1, y1, x2, y2)` to `(x, y, w, h)`.
xyxy_to_nxywh
    Convert bounding boxes from ;'x1, y1, x2, y2' to normalized '(x, y, w, h)'.
xyxy_to_yolo
    Convert bounding boxes from `(x1, y1, x2, y2)` to `(cx, cy, w, h)` in YOLO format.
xywh_to_xyxy
    Convert bounding boxes from `(x, y, w, h)` to `(x1, y1, x2, y2)`.
xywh_to_nxyxy
    Convert bounding boxes from `(x, y, w, h)` to normalized `(x1, y1, x2, y2)`.
xywh_to_nxywh
    Convert bounding boxes from `(x, y, w, h)` to normalized `(x, y, w, h)`.
xywh_to_yolo
    Convert bounding boxes from `(x, y, w, h)` to `(cx, cy, w, h)` in YOLO format.
nxyxy_to_xyxy
    Convert bounding boxes from normalized `(x1, y1, x2, y2)` to `(x1, y1, x2, y2)`.
nxywh_to_xywh
    Convert bounding boxes from normalized `(x, y, w, h)` to `(x, y, w, h)`.
nxywh_to_nxywh
    Convert bounding boxes from normalized `(x, y, w, h)` to normalized `(x, y, w, h)`.
nxyxy_to_yolo
    Convert bounding boxes from normalized `(x1, y1, x2, y2)` to `(cx, cy, w, h)` in YOLO format.
nxywh_to_xyxy
    Convert bounding boxes from normalized `(x, y, w, h)` to `(x1, y1, x2, y2)`.
nxywh_to_nxyxy
    Convert bounding boxes from normalized `(x, y, w, h)` to normalized `(x1, y1, x2, y2)`.
nxywh_to_xywh
    Convert bounding boxes from normalized `(x, y, w, h)` to `(x, y, w, h)`.
nxywh_to_yolo
    Convert bounding boxes from normalized `(x, y, w, h)` to `(cx, cy, w, h)` in YOLO format.
yolo_to_xyxy
    Convert bounding boxes from YOLO format `(cx, cy, w, h)` to `(x1, y1, x2, y2)`.
yolo_to_nxyxy
    Convert bounding boxes from YOLO format `(cx, cy, w, h)` to normalized `(x1, y1, x2, y2)`.
yolo_to_xywh
    Convert bounding boxes from YOLO format `(cx, cy, w, h)` to `(x, y, w, h)`.
yolo_to_nxywh
    Convert bounding boxes from YOLO format `(cx, cy, w, h)` to normalized `(x, y, w, h)`.

"""

from __future__ import annotations

from ._bounding import bounding
from ._constrain import constrain
from ._convert import (
    nxywh_to_nxyxy,
    nxywh_to_xywh,
    nxywh_to_xyxy,
    nxywh_to_yolo,
    nxyxy_to_nxywh,
    nxyxy_to_xywh,
    nxyxy_to_xyxy,
    nxyxy_to_yolo,
    xywh_to_nxywh,
    xywh_to_nxyxy,
    xywh_to_xyxy,
    xywh_to_yolo,
    xyxy_to_nxywh,
    xyxy_to_nxyxy,
    xyxy_to_xywh,
    xyxy_to_yolo,
    yolo_to_nxywh,
    yolo_to_nxyxy,
    yolo_to_xywh,
    yolo_to_xyxy,
)
from ._distance import euclidean, manhattan
from ._draw import draw_bboxes
from ._iou import iou, ious
from ._mean_ap import mean_ap
from ._nms import nms
from ._resize import resize
from ._score import score_bbox, score_bboxes
from ._valid import valid

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
    "nxywh_to_nxyxy",
    "nxywh_to_xywh",
    "nxywh_to_xyxy",
    "nxywh_to_yolo",
    "nxyxy_to_nxywh",
    "nxyxy_to_xywh",
    "nxyxy_to_xyxy",
    "nxyxy_to_yolo",
    "resize",
    "score_bbox",
    "score_bboxes",
    "valid",
    "xywh_to_nxywh",
    "xywh_to_nxyxy",
    "xywh_to_xyxy",
    "xywh_to_yolo",
    "xyxy_to_nxywh",
    "xyxy_to_nxyxy",
    "xyxy_to_xywh",
    "xyxy_to_yolo",
    "yolo_to_nxywh",
    "yolo_to_nxyxy",
    "yolo_to_xywh",
    "yolo_to_xyxy",
]
