# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import argparse
import time
import sys

import numpy as np

from cv2ext import enable_jit
from cv2ext.metrics import ncc


def main():
    parser = argparse.ArgumentParser(description="Run NCC on two random images.")
    parser.add_argument("--jit", action="store_true", help="Enable JIT.")
    parser.add_argument("--imgsize", type=int, default=1024, help="The size of the images to generate.")
    parser.add_argument("--iterations", type=int, default=100, help="The number of iterations to run.")
    args = parser.parse_args()

    rng = np.random.default_rng()
    if args.jit:
        enable_jit()
        img1 = rng.random((args.imgsize, args.imgsize, 3)).astype(np.uint8)
        img2 = rng.random((args.imgsize, args.imgsize, 3)).astype(np.uint8)
        _ = ncc(img1, img2)  # warmup cycle for jit 
    
    timing = []
    for _ in range(args.iterations):
        img1 = rng.random((args.imgsize, args.imgsize, 3)).astype(np.uint8)
        img2 = rng.random((args.imgsize, args.imgsize, 3)).astype(np.uint8)
        t0 = time.perf_counter()
        _ = ncc(img1, img2)
        t1 = time.perf_counter()
        timing.append(t1 - t0)

    avgtime = sum(timing) / len(timing)
    print(f"Time: {avgtime}s")

    return avgtime


if __name__ == "__main__":
    elapsed = main()
    sys.exit(int((elapsed * 10000)))
