# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
An implementation of the SHIFT methodology.

Paper: https://arxiv.org/pdf/2402.07415

Functions
---------
:func:`build_graph`
    Construct a conf graph using postprocessing techniques from the paper.
:func:`characterize`
    Characterize a set of models using the methodology from the paper.
:func:`create_graph`
    Create a minimal representation of the conf graph.

Classes
-------
:class:`Shift`
    Class implementing the overall methodology of the paper.
:class:`ShiftScheduler`
    An implemementation of the underlying scheduling methodology.

"""

from __future__ import annotations

from ._characterize import characterize
from ._confgraph import build_graph, create_graph
from ._core import Shift
from ._scheduler import ShiftScheduler

__all__ = [
    "Shift",
    "ShiftScheduler",
    "build_graph",
    "characterize",
    "create_graph",
]
