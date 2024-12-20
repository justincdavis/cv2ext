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
    parser = argparse.ArgumentParser(description="Process iou benchmarks.")
    parser.add_argument(
        "--iterations", type=int, default=100000, help="The number of iterations to run.",
    )
    args = parser.parse_args()

    func = partial(cv2ext.bboxes.iou, (5, 5, 100, 100), (0, 0, 95, 95))
    func2 = partial(cv2ext.bboxes.ious, [(5, 5, 100, 100)] * 1000, [(0, 0, 95, 95)] * 1000)

    run_benchmark(func, "iou", args.iterations)
    run_benchmark(func2, "ious", args.iterations // 100)


if __name__ == "__main__":
    main()
