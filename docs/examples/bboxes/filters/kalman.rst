.. _examples_bboxes/filters/kalman:

Example: bboxes/filters/kalman.py
=================================

.. code-block:: python

	# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
	#
	# MIT License
	"""Example showcasing kalman filtering on bounding boxes."""
	
	from __future__ import annotations
	
	import time
	from pathlib import Path
	
	import cv2
	from cv2ext.io import Display
	from cv2ext.bboxes.filters import KalmanFilter
	
	
	if __name__ == "__main__":
	    image = cv2.imread(str(Path("data") / "pictograms.png"))
	
	    init_bbox = (0, 0, 20, 20)
	
	    kf = KalmanFilter(init_bbox)
	
	    with Display("Kalman Filter Example") as display:
	        iters = 350
	        for i in range(iters):
	            bbox = kf(init_bbox)
	            new_image = image.copy()
	            cv2.rectangle(new_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
	            display(new_image)
	
	            add_x = int(5.0 - 5.0 * i / iters)
	            add_y = int(5.0 * i / iters)
	            init_bbox = (init_bbox[0] + add_x, init_bbox[1] + add_y, init_bbox[2] + add_x, init_bbox[3] + add_y)
	
	            time.sleep(0.01)

