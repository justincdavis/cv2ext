.. _examples_template/match_single:

Example: template/match_single.py
=================================

.. code-block:: python

	# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
	#
	# MIT License
	"""Example showcasing how to use the IterableVideo class."""
	
	from __future__ import annotations
	
	from pathlib import Path
	
	import cv2
	import cv2ext
	
	if __name__ == "__main__":
	    template = cv2.imread(str(Path("data") / "template.png"))
	    image = cv2.imread(str(Path("data") / "pictograms.png"))
	    output = cv2ext.template.match_single(image, template)
	    print(output)

