.. _examples_image/draw:

Example: image/draw.py
======================

.. code-block:: python

	# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
	#
	# MIT License
	"""Example showcasing how to use image.draw."""
	
	from pathlib import Path
	
	import cv2
	
	from cv2ext.image import draw
	from cv2ext.image.color import Color
	
	
	def main() -> None:
	    """Image tiling example."""
	    data_path = Path(__file__).parent.parent.parent / "data"
	    image = cv2.imread(str(data_path / "horse.jpg"))
	
	    print("Drawing rectangle")
	    canvas = draw.rectangle(image, (50, 50, 300, 200), color=Color.AQUA, thickness=5, copy=True)
	    cv2.imshow("Draw", canvas)
	    cv2.waitKey(0)
	
	    print("Drawing rectangle with opacity")
	    canvas = draw.rectangle(image, (50, 50, 300, 200), color=Color.AQUA, thickness=5, opacity=0.5, copy=True)
	    cv2.imshow("Draw", canvas)
	    cv2.waitKey(0)
	
	    print("Drawing circle")
	    canvas = draw.circle(image, (100, 100), 50, color=Color.AQUA, thickness=5, copy=True)
	    cv2.imshow("Draw", canvas)
	    cv2.waitKey(0)
	
	    print("Drawing circle with opacity")
	    canvas = draw.circle(image, (100, 100), 50, color=Color.AQUA, thickness=5, opacity=0.5, copy=True)
	    cv2.imshow("Draw", canvas)
	    cv2.waitKey(0)
	
	
	if __name__ == "__main__":
	    main()

