# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Process ncc benchmarks.")
    parser.add_argument("--imgsize", type=int, default=1024, help="The size of the images to generate.")
    parser.add_argument("--iterations", type=int, default=1000, help="The number of iterations to run.")
    args = parser.parse_args()

    naivecmd = ["python3", str(Path("benchmarks") / "ncc" / "run.py"), "--iterations", str(args.iterations), "--imgsize", str(args.imgsize)]
    jitcmd = naivecmd.copy() + ["--jit"]

    # No visualization commands
    naiveret = subprocess.run(naivecmd)
    jitret = subprocess.run(jitcmd)
    retcodes = [
        naiveret.returncode / 10000,
        jitret.returncode / 10000,
    ]
    retcodes = [
        retcodes[0] / r for r in retcodes
    ]
    baseplot = sns.barplot(
        x=["No-JIT", "JIT"],
        y=retcodes,
    )
    baseplot.set_title("Speedup over Naive")
    baseplot.set_ylabel("Speedup")
    basefig = baseplot.get_figure()
    basefig.tight_layout()
    basefig.savefig(Path("benchmarks") / "ncc" / "speedup.png")
    plt.close(basefig)


if __name__ == "__main__":
    main()
