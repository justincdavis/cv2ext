.. _examples_tracking/multi_tracker:

Example: tracking/multi_tracker.py
==================================

.. code-block:: python

	# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
	#
	# MIT License
	"""Example showcasing how to use the CSK tracker."""
	
	from __future__ import annotations
	
	import argparse
	import time
	
	import cv2
	import numpy as np
	from cv2ext import Display, IterableVideo, set_log_level
	from cv2ext.tracking import MultiTracker, TrackerType
	
	
	def main(tracker_type: TrackerType) -> None:
	    """CSK Tracker example."""
	    display = Display("tracking")
	    tracker = MultiTracker(tracker_type, use_threads=True)
	    started = False
	    update_times = []
	    for frame_id, frame in IterableVideo("data/testvid.mp4"):
	        if frame_id < 100:
	            continue
	        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	        if not started:
	            bbox = (149, 66, 69, 49)
	            x, y, w, h = bbox
	            bbox = (x, y, x + w, y + h)
	            tracker.init(gray_frame, [bbox])
	            started = True
	        else:
	            t0 = time.perf_counter()
	            data = tracker.update(gray_frame)
	            t1 = time.perf_counter()
	            update_times.append(t1 - t0)
	            for success, bbox in data:
	                if success:
	                    cv2.rectangle(
	                        gray_frame,
	                        (bbox[0], bbox[1]),
	                        (bbox[2], bbox[3]),
	                        (0, 255, 0),
	                        2,
	                    )
	            display.update(gray_frame)
	            time.sleep(0.01)
	
	    mean_time = round(np.mean(update_times) * 1000, 1)
	    print(f"Average update time: {mean_time} ms.")
	
	
	if __name__ == "__main__":
	    parser = argparse.ArgumentParser()
	    parser.add_argument(
	        "--tracker",
	        type=str,
	        default="kcf",
	        help="The type of tracker to use. Options: boosting, csrt, kcf, medianflow, mil, mosse, tld",
	    )
	    args = parser.parse_args()
	    tracker_type_str = args.tracker.upper()
	    tracker_dict = {
	        "BOOSTING": TrackerType.BOOSTING,
	        "CSRT": TrackerType.CSRT,
	        "KCF": TrackerType.KCF,
	        "MEDIANFLOW": TrackerType.MEDIAN_FLOW,
	        "MIL": TrackerType.MIL,
	        "MOSSE": TrackerType.MOSSE,
	        "TLD": TrackerType.TLD,
	    }
	    try:
	        tracker_type = tracker_dict[tracker_type_str]
	    except KeyError as err:
	        err_msg = f"Invalid tracker type: {tracker_type_str}"
	        raise ValueError(err_msg) from err
	
	    set_log_level("INFO")
	    main(tracker_type)

