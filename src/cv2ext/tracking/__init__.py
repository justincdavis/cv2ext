# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
Submodule containing tools for tracking objects in videos.

Submodules
----------
kernels
    Submodule containing the kernels used by the CSK tracker.

Classes
-------
CSKTracker
    A class for tracking objects in videos using the CSK tracker.
MultiTracker
    A class for tracking multiple objects in videos using a single tracker.
TrackerInterface
    An interface for tracking objects in videos.

"""
from __future__ import annotations

# block 1 of imports, required for CSKTracker
from ._interface import TrackerInterface
from . import kernels

# block 2 of imports, require use of kernels and Interface
from ._csk import CSKTracker
from ._multi_tracker import MultiTracker

__all__ = ["CSKTracker", "MultiTracker", "TrackerInterface", "kernels"]
