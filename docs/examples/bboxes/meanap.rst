.. _examples_bboxes/meanap:

Example: bboxes/meanap.py
=========================

.. code-block:: python

	# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
	#
	# MIT License
	"""Example showcasing how to use the mean_ap function."""
	
	from __future__ import annotations
	
	import cv2ext
	
	if __name__ == "__main__":
	    bboxes = [
	        ((0, 0, 10, 10), 0, 0.75),
	        ((1, 1, 9, 9), 0, 0.75),
	        ((2, 2, 8, 8), 0, 0.75),
	    ]
	    gt = [
	        ((0, 0, 10, 10), 0),
	        ((1, 1, 9, 9), 0),
	        ((2, 2, 8, 8), 0),
	    ]
	    mean_ap = cv2ext.bboxes.mean_ap([bboxes], [gt], 1)
	    print(mean_ap)

