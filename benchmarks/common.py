# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import seaborn as sns

from cv2ext import enable_jit

if TYPE_CHECKING:
    from functools import partial


def run_func(func: partial, iterations: int = 1000, *, jit: bool = False) -> float:
    if jit:
        enable_jit()
        func()

    timing = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        timing.append(t1 - t0)

    return (sum(timing) / len(timing)) * 1000.0


def run_benchmark(func: partial, title: str, iters: int) -> None:
    no_jit = run_func(func, iterations=iters)
    with_jit = run_func(func, iterations=iters, jit=True)

    baseplot = sns.barplot(
        x=["No-JIT", "JIT"],
        y=[no_jit, with_jit],
    )
    baseplot.set_title("IOU")
    baseplot.set_ylabel("Time (ms)")
    basefig = baseplot.get_figure()
    basefig.tight_layout()
    basefig.savefig(str(Path("benchmarks") / "plots" / f"{title}.png"))
    plt.close(basefig)
