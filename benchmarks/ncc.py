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
    parser = argparse.ArgumentParser(description="Process ncc benchmarks.")
    parser.add_argument(
        "--iterations", type=int, default=1000, help="The number of iterations to run.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng()
    func = partial(
        cv2ext.metrics.ncc,
        rng.integers(0, 255, (320, 320, 1), dtype=np.uint8),
        rng.integers(0, 255, (320, 320, 1), dtype=np.uint8),
        resize=False,
    )

    run_benchmark(func, "ncc", args.iterations)


if __name__ == "__main__":
    main()
