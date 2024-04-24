from __future__ import annotations

import time

import cv2

from cv2ext import Display, IterableVideo
from cv2ext.tracking import CSKTracker


def main():
    display = Display("tracking")
    tracker = CSKTracker(eta=0.09)
    started = False
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
            bbox = tracker.update(frame)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            display.update(frame)
            time.sleep(0.01)


if __name__ == "__main__":
    main()
