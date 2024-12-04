# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Subpackage containing tools for working with simple bounding boxes.

Functions
---------
:func:`bounding`
    Get a bounding box which encloses all the given bounding boxes.
:func:`constrain`
    Constrain a bounding box to be within the bounds of an image.
:func:`draw_bboxes`
    Draw bounding boxes on an image.
:func:`euclidean`
    Compute the euclidean distance between two bounding boxes.
:func:`filter_bboxes_by_region`
    Filter a sequence of bounding boxes by a region to contain them in.
:func:`iou`
    Calculate the intersection over union of two bounding boxes.
:func:`ious`
    Calculate the intersection over union of a set of bounding boxes.
:func:`manhattan`
    Compute the manhattan distance between two bounding boxes.
:func:`mean_ap`
    Calculate the mean average precision of a set of bounding boxes.
:func:`nms`
    Perform non-maximum suppression on a set of bounding boxes.
:func:`resize`
    Resize a bounding box based on one image size to another.
:func:`scale`
    Scale a bounding box from one image size to another.
:func:`scale_many`
    Scale a set of bounding boxes from one image size to another.
:func:`score_bbox`
    Score a bounding box relative to a target bbox.
:func:`score_bboxes`
    Score a set of bounding boxes relative to a target bbox.
:func:`valid`
    Check if a bounding box is valid.
:func:`within`
    Check if a bounding box is within the bounds of an image.
:func:`xyxy_to_nxyxy`
    Convert bounding boxes from `(x1, y1, x2, y2)` to normalized `(x1, y1, x2, y2)`.
:func:`xyxy_to_xywh`
    Convert bounding boxes from `(x1, y1, x2, y2)` to `(x, y, w, h)`.
:func:`xyxy_to_nxywh`
    Convert bounding boxes from ;'x1, y1, x2, y2' to normalized '(x, y, w, h)'.
:func:`xyxy_to_yolo`
    Convert bounding boxes from `(x1, y1, x2, y2)` to `(cx, cy, w, h)` in YOLO format.
:func:`xywh_to_xyxy`
    Convert bounding boxes from `(x, y, w, h)` to `(x1, y1, x2, y2)`.
:func:`xywh_to_nxyxy`
    Convert bounding boxes from `(x, y, w, h)` to normalized `(x1, y1, x2, y2)`.
:func:`xywh_to_nxywh`
    Convert bounding boxes from `(x, y, w, h)` to normalized `(x, y, w, h)`.
:func:`xywh_to_yolo`
    Convert bounding boxes from `(x, y, w, h)` to `(cx, cy, w, h)` in YOLO format.
:func:`nxyxy_to_xyxy`
    Convert bounding boxes from normalized `(x1, y1, x2, y2)` to `(x1, y1, x2, y2)`.
:func:`nxywh_to_xywh`
    Convert bounding boxes from normalized `(x, y, w, h)` to `(x, y, w, h)`.
:func:`nxywh_to_nxywh`
    Convert bounding boxes from normalized `(x, y, w, h)` to normalized `(x, y, w, h)`.
:func:`nxyxy_to_yolo`
    Convert bounding boxes from normalized `(x1, y1, x2, y2)` to `(cx, cy, w, h)` in YOLO format.
:func:`nxywh_to_xyxy`
    Convert bounding boxes from normalized `(x, y, w, h)` to `(x1, y1, x2, y2)`.
:func:`nxywh_to_nxyxy`
    Convert bounding boxes from normalized `(x, y, w, h)` to normalized `(x1, y1, x2, y2)`.
:func:`nxywh_to_xywh`
    Convert bounding boxes from normalized `(x, y, w, h)` to `(x, y, w, h)`.
:func:`nxywh_to_yolo`
    Convert bounding boxes from normalized `(x, y, w, h)` to `(cx, cy, w, h)` in YOLO format.
:func:`yolo_to_xyxy`
    Convert bounding boxes from YOLO format `(cx, cy, w, h)` to `(x1, y1, x2, y2)`.
:func:`yolo_to_nxyxy`
    Convert bounding boxes from YOLO format `(cx, cy, w, h)` to normalized `(x1, y1, x2, y2)`.
:func:`yolo_to_xywh`
    Convert bounding boxes from YOLO format `(cx, cy, w, h)` to `(x, y, w, h)`.
:func:`yolo_to_nxywh`
    Convert bounding boxes from YOLO format `(cx, cy, w, h)` to normalized `(x, y, w, h)`.

"""

from __future__ import annotations

from ._algorithms import filter_bboxes_by_region
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
from ._scale import scale, scale_many
from ._score import score_bbox, score_bboxes
from ._valid import valid, within

__all__ = [
    "bounding",
    "constrain",
    "draw_bboxes",
    "euclidean",
    "filter_bboxes_by_region",
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
    "scale",
    "scale_many",
    "score_bbox",
    "score_bboxes",
    "valid",
    "within",
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
