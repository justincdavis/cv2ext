.. _examples_io/video_writer_fourcc:

Example: io/video_writer_fourcc.py
==================================

.. code-block:: python

	# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
	#
	# MIT License
	"""Example showcasing how to use the IterableVideo class."""
	
	from __future__ import annotations
	
	from cv2ext import Fourcc, IterableVideo, VideoWriter, set_log_level
	
	if __name__ == "__main__":
	    set_log_level("DEBUG")
	    # create an IterableVideo object
	    video = IterableVideo("video.mp4")
	
	    # can create a VideoWriter with a wide variety of Fourcc codecs
	    print(f"Available codecs: {len(list(Fourcc))}")
	    writer = None
	    for fourcc in [Fourcc.H264, Fourcc.XVID, Fourcc.MP4V, Fourcc.mp4v]:
	        writer = VideoWriter("output.mp4", fourcc=fourcc)
	
	    with writer as writer:
	        # iterate over the video
	        for _, frame in video:
	            writer.write(frame)
	            # print(f"Frame {frame_id}: {frame.shape}")

