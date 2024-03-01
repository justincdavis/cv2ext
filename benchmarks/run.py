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
from __future__ import annotations

import argparse
import os
import time
import sys

import cv2
from cv2ext import Display, IterableVideo
from tqdm import tqdm


def naive(video: str, show: bool) -> float:
    cap = cv2.VideoCapture(video)

    if show:
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.startWindowThread()

    t0 = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if show:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    t1 = time.perf_counter()

    cap.release()
    if show:
        cv2.destroyWindow("frame")
        cv2.waitKey(1)
    return t1 - t0

def threadread_naivedisplay(video: str, show: bool) -> float:
    video = IterableVideo(video, buffersize=128, use_thread=True)

    if show:
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.startWindowThread()

    t0 = time.perf_counter()
    for _, frame in video:
        if show:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    t1 = time.perf_counter()

    if show:
        cv2.destroyWindow("frame")
        cv2.waitKey(1)
    return t1 - t0

def naiveread_threaddisplay(video: str, show: bool) -> float:
    cap = cv2.VideoCapture(video)
    display = Display("example", show=show)

    t0 = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if show:
            display(frame)
    t1 = time.perf_counter()

    cap.release()
    return t1 - t0

def threaded(video: str, show: bool) -> float:
    video = IterableVideo(video, buffersize=128, use_thread=True)
    if show:
        display = Display("example", show=show)

    t0 = time.perf_counter()
    for _, frame in video:
        if show:
            display(frame)
    t1 = time.perf_counter()

    return t1 - t0

def main():
    parser = argparse.ArgumentParser(description="Display a video.")
    parser.add_argument("--video", required=True, help="The video to process.")
    parser.add_argument("--show", action="store_true", help="Show the video.")
    parser.add_argument("--iterations", type=int, default=10, help="The number of iterations to run.")
    parser.add_argument("--threaded", action="store_true", help="Use the threaded backend.")
    parser.add_argument("--mix1", action="store_true", help="Use threaded reading and naive display.")
    parser.add_argument("--mix2", action="store_true", help="Use naive reading and threaded display.")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video {args.video} does not exist.")

    times = []
    if not args.threaded:
        for _ in tqdm(range(args.iterations)):
            naivetime = naive(args.video, args.show)
            times.append(naivetime)
    elif args.mix1:
        for _ in tqdm(range(args.iterations)):
            threadtime = threadread_naivedisplay(args.video, args.show)
            times.append(threadtime)
    elif args.mix2:
        for _ in tqdm(range(args.iterations)):
            mixtime = naiveread_threaddisplay(args.video, args.show)
            times.append(mixtime)
    else:
        for _ in tqdm(range(args.iterations)):
            threadtime = threaded(args.video, args.show)
            times.append(threadtime)

    avgtime = sum(times) / len(times)
    print(f"Time: {avgtime:.3f}s")

    return avgtime

if __name__ == "__main__":
    elapsed = main()
    sys.exit(int((elapsed * 100)))
