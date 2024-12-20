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
    parser = argparse.ArgumentParser(description="Display a video.")
    parser.add_argument("--video", required=True, help="The video to process.")
    parser.add_argument(
        "--iterations", type=int, default=10, help="The number of iterations to run."
    )
    args = parser.parse_args()

    if not Path(args.video).exists():
        raise FileNotFoundError(f"Video {args.video} does not exist.")

    naivecmd = [
        "python3",
        str(Path("benchmarks") / "visual" / "run.py"),
        "--video",
        args.video,
        "--iterations",
        str(args.iterations),
    ]
    naiveshowcmd = naivecmd.copy() + ["--show"]
    threadedcmd = naivecmd.copy() + ["--threaded"]
    threadshowcmd = threadedcmd.copy() + ["--show"]
    mix1cmd = naivecmd.copy() + ["--mix1"]
    mix1showcmd = naiveshowcmd.copy() + ["--mix1"]
    mix2cmd = naivecmd.copy() + ["--mix2"]
    mix2showcmd = naiveshowcmd.copy() + ["--mix2"]

    # No visualization commands
    naiveret = subprocess.run(naivecmd)
    threadedret = subprocess.run(threadedcmd)
    mix1ret = subprocess.run(mix1cmd)
    mix2ret = subprocess.run(mix2cmd)
    retcodes = [
        naiveret.returncode / 100,
        mix1ret.returncode / 100,
        mix2ret.returncode / 100,
        threadedret.returncode / 100,
    ]
    retcodes = [retcodes[0] / r for r in retcodes]
    baseplot = sns.barplot(
        x=["Naive", "Thread-reads", "Thread-displays", "Fully-threaded"],
        y=retcodes,
    )
    baseplot.set_title("Speedup over Naive")
    baseplot.set_ylabel("Speedup")
    basefig = baseplot.get_figure()
    basefig.tight_layout()
    basefig.savefig(Path("benchmarks") / "visual" / "baseplot.png")
    plt.close(basefig)

    # Visualization commands
    naiveshowret = subprocess.run(naiveshowcmd)
    threadedshowret = subprocess.run(threadshowcmd)
    mix1showret = subprocess.run(mix1showcmd)
    mix2showret = subprocess.run(mix2showcmd)
    showretcodes = [
        naiveshowret.returncode / 100,
        mix1showret.returncode / 100,
        mix2showret.returncode / 100,
        threadedshowret.returncode / 100,
    ]
    showretcodes = [showretcodes[0] / r for r in showretcodes]
    showplot = sns.barplot(
        x=["Naive", "Thread-reads", "Thread-displays", "Fully-threaded"],
        y=showretcodes,
    )
    showplot.set_title("Speedup over Naive - Visualization")
    showplot.set_ylabel("Speedup")
    showfig = showplot.get_figure()
    showfig.tight_layout()
    showfig.savefig(Path("benchmarks") / "visual" / "showplot.png")
    plt.close(showfig)


if __name__ == "__main__":
    main()
