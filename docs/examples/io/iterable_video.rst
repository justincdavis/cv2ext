.. _examples_io/iterable_video:

Example: io/iterable_video.py
=============================

.. code-block:: python

	# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
	#
	# MIT License
	"""Example showcasing how to use the IterableVideo class."""
	
	from __future__ import annotations
	
	from cv2ext import IterableVideo, set_log_level
	
	if __name__ == "__main__":
	    set_log_level("DEBUG")
	    # create an IterableVideo object
	    video = IterableVideo("video.mp4", use_thread=False)
	
	    # iterate over the video
	    for frame_id, frame in video:
	        print(f"Frame {frame_id}: {frame.shape}")
	
	    # create it again this time using the thread backend
	    video = IterableVideo("video.mp4", use_thread=True)
	
	    # iterate over the video
	    for frame_id, frame in video:
	        print(f"Frame {frame_id}: {frame.shape}")

