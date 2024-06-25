.. _examples_io/video_writer:

Example: io/video_writer.py
===========================

.. code-block:: python

	# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
	#
	# MIT License
	"""Example showcasing how to use the IterableVideo class."""
	
	from __future__ import annotations
	
	from cv2ext import IterableVideo, VideoWriter, set_log_level
	
	if __name__ == "__main__":
	    set_log_level("DEBUG")
	    # create an IterableVideo object
	    video = IterableVideo("video.mp4")
	
	    with VideoWriter("output.mp4") as writer:
	        # iterate over the video
	        for frame_id, frame in video:
	            writer.write(frame)
	            print(f"Frame {frame_id}: {frame.shape}")

