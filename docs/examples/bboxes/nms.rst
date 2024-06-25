.. _examples_bboxes/nms:

Example: bboxes/nms.py
======================

.. code-block:: python

	# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
	#
	# MIT License
	"""Example showcasing how to use the nms function."""
	
	from __future__ import annotations
	
	import cv2ext
	
	if __name__ == "__main__":
	    bboxes = [
	        ((0, 0, 10, 10), 0, 0.75),
	        ((1, 1, 9, 9), 0, 0.75),
	        ((2, 2, 8, 8), 0, 0.75),
	    ]
	    new_bboxes = cv2ext.bboxes.nms(bboxes, 0.5)
	    print(new_bboxes)

