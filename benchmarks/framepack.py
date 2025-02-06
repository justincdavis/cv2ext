# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import argparse
from functools import partial

import numpy as np

import cv2ext
from common import run_benchmark


def main():
    cv2ext.set_log_level("DEBUG")
    parser = argparse.ArgumentParser(description="Process frame packing benchmarks.")
    parser.add_argument(
        "--iterations", type=int, default=1000, help="The number of iterations to run.",
    )
    args = parser.parse_args()

    # simple method
    packer = cv2ext.detection.AnnealingFramePacker((1920, 1080), method="simple")
    frame = np.random.random_integers(0, 255, size=(1080, 1920, 3)).astype(np.uint8)
    func = partial(
        packer.pack,
        frame,
    )
    run_benchmark(func, "framepacking_simple", args.iterations)

    # shelf method
    packer = cv2ext.detection.AnnealingFramePacker((1920, 1080), method="shelf")
    frame = np.random.random_integers(0, 255, size=(1080, 1920, 3)).astype(np.uint8)
    func = partial(
        packer.pack,
        frame,
    )
    run_benchmark(func, "framepacking_shelf", args.iterations)


if __name__ == "__main__":
    main()
