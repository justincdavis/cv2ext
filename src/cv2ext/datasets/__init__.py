# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule containing tools for reading and interacting with datasets.

Submodules
----------
:mod:`coco`
    Tools for interacting with COCO datasets.
:mod:`kitti`
    Tools for interacting with KITTI datasets.
:mod:`mot`
    Tools for interacting with MOT17(det)/MOT20(det) datasets.

"""

from __future__ import annotations

from . import coco, kitti, mot

__all__ = [
    "coco",
    "kitti",
    "mot",
]
