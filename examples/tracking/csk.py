from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from cv2ext import Display, IterableVideo, enable_jit, set_log_level
from cv2ext.tracking import CSKTracker


def main():
    display = Display("tracking")
    tracker = CSKTracker(eta=0.09)
    started = False
    update_times = []
    for frame_id, frame in IterableVideo("data/testvid.mp4"):
        if frame_id < 100:
            continue
        if frame_id > 200:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not started:
            bbox = (149, 66, 69, 49)
            x, y, w, h = bbox
            bbox = (x, y, x + w, y + h)
            tracker.init(frame, bbox)
            started = True
        else:
            t0 = time.perf_counter()
            bbox = tracker.update(frame)
            t1 = time.perf_counter()
            update_times.append(t1 - t0)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            display.update(frame)
            time.sleep(0.01)

    mean_time = round(np.mean(update_times) * 1000, 1)
    print(f"Average update time: {mean_time} ms.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jit", action="store_true", help="Enable JIT.")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel JIT.")
    args = parser.parse_args()

    set_log_level("DEBUG")

    if args.jit:
        enable_jit(on=True, parallel=args.parallel)

    main()
