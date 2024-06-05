.. _examples_metrics/ncc:

Example: metrics/ncc.py
=======================

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
	"""Example showcasing ncc computation between two images."""
	from __future__ import annotations
	
	import cv2ext
	import numpy as np
	
	if __name__ == "__main__":
	    rng = np.random.Generator(np.random.PCG64())
	    image1 = rng.random((100, 100))
	    image2 = rng.random((100, 100))
	
	    ncc = cv2ext.metrics.ncc(image1, image2)
	    print(ncc)

