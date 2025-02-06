# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import argparse
from functools import partial

import cv2ext
from common import run_benchmark


def main():
    cv2ext.set_log_level("DEBUG")
    parser = argparse.ArgumentParser(description="Process nms benchmarks.")
    parser.add_argument(
        "--iterations", type=int, default=1000, help="The number of iterations to run.",
    )
    args = parser.parse_args()

    dets: list[tuple[tuple[int, int, int, int], float, int]] = [
        ((0, 0, 100, 100), 0.9, 0),
        ((10, 10, 110, 110), 0.8, 0),
        ((200, 200, 300, 300), 0.7, 1),
        ((250, 250, 350, 350), 0.4, 1),
        ((400, 400, 500, 500), 0.95, 0),
        ((0, 0, 50, 50), 0.3, 0),
        ((20, 20, 60, 60), 0.6, 0),
        ((150, 150, 180, 180), 0.85, 1),
        ((100, 100, 200, 200), 0.5, 0),
        ((450, 450, 550, 550), 0.98, 0),
        ((500, 500, 600, 600), 0.88, 1),
        ((550, 550, 650, 650), 0.7, 1),
        ((600, 600, 700, 700), 0.9, 0),
        ((350, 350, 450, 450), 0.6, 0),
        ((100, 100, 130, 130), 0.4, 1),
        ((200, 200, 400, 400), 0.95, 0),
        ((120, 120, 220, 220), 0.75, 1),
        ((0, 0, 40, 40), 0.5, 0),
        ((20, 20, 80, 80), 0.2, 1),
        ((50, 50, 150, 150), 0.85, 0),
        ((70, 70, 170, 170), 0.8, 0),
        ((300, 300, 400, 400), 0.7, 1),
        ((320, 320, 420, 420), 0.72, 1),
        ((600, 600, 620, 620), 0.9, 1),
        ((615, 615, 635, 635), 0.9, 1),
        ((700, 700, 800, 800), 0.95, 0),
        ((750, 750, 850, 850), 0.98, 0),
        ((120, 120, 180, 180), 0.65, 1),
        ((130, 130, 190, 190), 0.55, 1),
    ]
    func = partial(
        cv2ext.bboxes.nms,
        dets,
    )

    run_benchmark(func, "nms", args.iterations)

if __name__ == "__main__":
    main()
