.. _examples_bboxes/iou:

Example: bboxes/iou.py
======================

.. code-block:: python

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
	"""Example showcasing IoU calculation for bounding boxes."""
	from __future__ import annotations
	
	from cv2ext import bboxes
	
	if __name__ == "__main__":
	    a = (0, 0, 10, 10)
	    b = (5, 5, 10, 10)
	    iou = bboxes.iou(a, b)
	    print(iou)

