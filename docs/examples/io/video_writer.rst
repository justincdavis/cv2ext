.. _examples_io/video_writer:

Example: io/video_writer.py
===========================

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

