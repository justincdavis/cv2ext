# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

from cv2ext import IterableVideo
from cv2ext.detection import AnnealingFramePacker, RandomFramePacker


def test_annealing_packer():
    video = IterableVideo(Path("data/testvid.mp4"))

    packer = AnnealingFramePacker(
        (video.height, video.width),
    )

    frame_sizes = []
    for _, frame in video:
        packed, _ = packer.pack(frame, exclude=[])
        h, w = packed.shape[:2]
        frame_sizes.append(h * w)

    assert len(frame_sizes) > 0
    assert len(frame_sizes) == len(video)
    assert min(frame_sizes) > 0
    assert max(frame_sizes) > 0
    assert min(frame_sizes) <= max(frame_sizes)


def test_random_packer():
    video = IterableVideo(Path("data/testvid.mp4"))

    packer = RandomFramePacker(
        (video.height, video.width),
    )

    frame_sizes = []
    for _, frame in video:
        packed, _ = packer.pack(frame, exclude=[])
        h, w = packed.shape[:2]
        frame_sizes.append(h * w)

    assert len(frame_sizes) > 0
    assert len(frame_sizes) == len(video)
    assert min(frame_sizes) > 0
    assert max(frame_sizes) > 0
