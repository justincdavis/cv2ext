# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

from cv2ext import IterableVideo
from cv2ext.detection import AnnealingFramePacker, RandomFramePacker


def _test_annealing_packer(packer: AnnealingFramePacker, video: IterableVideo):
    frame_sizes = []
    for _, frame in video:
        packed, _ = packer.pack(frame, exclude=[])
        h, w = packed.shape[:2]
        frame_sizes.append(h * w)

    assert len(frame_sizes) > 0
    assert len(frame_sizes) == len(video)
    assert min(frame_sizes) > 0
    assert max(frame_sizes) > 0
    assert min(frame_sizes) < max(frame_sizes)


def test_annealing_packer_simple():
    video = IterableVideo(Path("data/testvid.mp4"))
    packer = AnnealingFramePacker(
        (video.width, video.height), method="simple",
    )
    _test_annealing_packer(packer, video)


def test_annealing_packer_smart():
    video = IterableVideo(Path("data/testvid.mp4"))
    packer = AnnealingFramePacker(
        (video.width, video.height), method="smart",
    )
    _test_annealing_packer(packer, video)


def _test_random_packer(packer: RandomFramePacker, video: IterableVideo):
    frame_sizes = []
    for _, frame in video:
        packed, _ = packer.pack(frame, exclude=[])
        h, w = packed.shape[:2]
        frame_sizes.append(h * w)

    assert len(frame_sizes) > 0
    assert len(frame_sizes) == len(video)
    assert min(frame_sizes) > 0
    assert max(frame_sizes) > 0


def test_random_packer_simple():
    video = IterableVideo(Path("data/testvid.mp4"))
    packer = RandomFramePacker(
        (video.width, video.height), method="simple",
    )
    _test_random_packer(packer, video)


def test_random_packer_smart():
    video = IterableVideo(Path("data/testvid.mp4"))
    packer = RandomFramePacker(
        (video.width, video.height), method="smart",
    )
    _test_random_packer(packer, video)
