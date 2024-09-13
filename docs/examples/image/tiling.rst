.. _examples_image/tiling:

Example: image/tiling.py
========================

.. code-block:: python

	# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
	#
	# MIT License
	"""Example showcasing how to tile an image."""
	
	import time
	from pathlib import Path
	
	import cv2
	import cv2ext
	
	
	def main() -> None:
	    """Image tiling example."""
	    base = cv2.imread(str(Path(__file__).parent / "data" / "blank.png"))
	    tile = cv2.imread(str(Path(__file__).parent / "data" / "person.png"))
	
	    tiled_image = cv2ext.image.create_tiled_image(tile, base)
	    
	    with cv2ext.Display("Tiled image") as display:
	        display.update(tiled_image)
	        time.sleep(1)
	        
	        for partial_tiled in cv2ext.image.image_tiler(base, tile):
	            display.update(partial_tiled)
	            time.sleep(0.1)
	
	
	if __name__ == "__main__":
	    main()

