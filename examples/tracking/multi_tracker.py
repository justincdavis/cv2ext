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
"""Example showcasing how to use the CSK tracker."""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from cv2ext import Display, IterableVideo, enable_jit, set_log_level
from cv2ext.tracking import CSKTracker, MultiTracker


def main() -> None:
    """CSK Tracker example."""
    display = Display("tracking")
    tracker = MultiTracker(CSKTracker, use_threads=False)
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
            bboxs = tracker.update(gray_frame)
            bbox = bboxs[0]
            t1 = time.perf_counter()
            update_times.append(t1 - t0)
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
    parser.add_argument("--jit", action="store_true", help="Enable JIT.")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel JIT.")
    args = parser.parse_args()

    set_log_level("INFO")

    if args.jit:
        enable_jit(on=True, parallel=args.parallel)

    main()
