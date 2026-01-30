# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

from cv2ext import IterableVideo
from cv2ext.detection import AnnealingFramePacker, RandomFramePacker

from ..helpers import wrapper, wrapper_jit


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


def _test_annealing_packer_simple():
    video = IterableVideo(Path("data/testvid.mp4"))
    packer = AnnealingFramePacker(
        (video.width, video.height), method="simple",
    )
    _test_annealing_packer(packer, video)


def _test_annealing_packer_shelf():
    video = IterableVideo(Path("data/testvid.mp4"))
    packer = AnnealingFramePacker(
        (video.width, video.height), method="shelf",
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


def _test_random_packer_simple():
    video = IterableVideo(Path("data/testvid.mp4"))
    packer = RandomFramePacker(
        (video.width, video.height), method="simple",
    )
    _test_random_packer(packer, video)


def _test_random_packer_shelf():
    video = IterableVideo(Path("data/testvid.mp4"))
    packer = RandomFramePacker(
        (video.width, video.height), method="shelf",
    )
    _test_random_packer(packer, video)


@wrapper
def test_annealing_packer_simple():
    _test_annealing_packer_simple()


@wrapper_jit
def test_annealing_packer_simple_jit():
    _test_annealing_packer_simple()


@wrapper
def test_annealing_packer_shelf():
    _test_annealing_packer_shelf()


@wrapper_jit
def test_annealing_packer_shelf_jit():
    _test_annealing_packer_shelf()


@wrapper
def test_random_packer_simple():
    _test_random_packer_simple()


@wrapper_jit
def test_random_packer_simple_jit():
    _test_random_packer_simple()


@wrapper
def test_random_packer_shelf():
    _test_random_packer_shelf()


@wrapper_jit
def test_random_packer_shelf_jit():
    _test_random_packer_shelf()


# Multi-region tests
def _test_multi_region_packer(packer, video, regions: int, method: str):
    """Test multi-region packing functionality."""
    for _, frame in video:
        images, transforms = packer.pack(frame, exclude=[], method=method, regions=regions)

        # Verify we got lists back
        assert isinstance(images, list), f"Expected list, got {type(images)}"
        assert isinstance(transforms, list), f"Expected list, got {type(transforms)}"

        # Verify we got the expected number of regions (or fewer if not enough cells)
        assert len(images) >= 1
        assert len(images) <= regions
        assert len(images) == len(transforms)

        # Verify each region has valid data
        for i, (img, trans) in enumerate(zip(images, transforms)):
            h, w = img.shape[:2]
            assert h > 0 and w > 0, f"Region {i} has invalid dimensions: {h}x{w}"
            assert trans.shape[2] == 2, f"Transform should have 2 channels, got {trans.shape[2]}"

        # Test unpack with region_idx
        for i, trans in enumerate(transforms):
            # Unpack empty detections (should work without error)
            unpacked = packer.unpack([], transforms, region_idx=i)
            assert unpacked == []

        # Test unpack_multi
        empty_dets_per_region = [[] for _ in range(len(transforms))]
        all_unpacked = packer.unpack_multi(empty_dets_per_region, transforms)
        assert all_unpacked == []

        # Only test first frame to keep tests fast
        break


def _test_annealing_packer_multi_region():
    video = IterableVideo(Path("data/testvid.mp4"))
    packer = AnnealingFramePacker((video.width, video.height))

    # Test with 2 regions
    _test_multi_region_packer(packer, video, regions=2, method="simple")
    packer.reset()
    video = IterableVideo(Path("data/testvid.mp4"))
    _test_multi_region_packer(packer, video, regions=2, method="shelf")

    # Test with 4 regions
    packer.reset()
    video = IterableVideo(Path("data/testvid.mp4"))
    _test_multi_region_packer(packer, video, regions=4, method="simple")
    packer.reset()
    video = IterableVideo(Path("data/testvid.mp4"))
    _test_multi_region_packer(packer, video, regions=4, method="shelf")


def _test_random_packer_multi_region():
    video = IterableVideo(Path("data/testvid.mp4"))
    packer = RandomFramePacker((video.width, video.height), threshold=0.5)

    # Test with 2 regions
    _test_multi_region_packer(packer, video, regions=2, method="simple")
    packer.reset()
    video = IterableVideo(Path("data/testvid.mp4"))
    _test_multi_region_packer(packer, video, regions=2, method="shelf")


def _test_multi_region_backward_compat():
    """Test that regions=1 returns single arrays (backward compatible)."""
    video = IterableVideo(Path("data/testvid.mp4"))
    packer = AnnealingFramePacker((video.width, video.height))

    for _, frame in video:
        packed, transform = packer.pack(frame, exclude=[], regions=1)

        # Should be single arrays, not lists
        assert not isinstance(packed, list), "regions=1 should return single array"
        assert not isinstance(transform, list), "regions=1 should return single array"
        assert packed.ndim == 3, "packed image should be 3D"
        assert transform.ndim == 3, "transform should be 3D"

        # Unpack should work without region_idx
        unpacked = packer.unpack([], transform)
        assert unpacked == []

        break


def _test_multi_region_empty_detections():
    """Test multi-region packing with empty detections - common real-world scenario."""
    video = IterableVideo(Path("data/testvid.mp4"))
    packer = AnnealingFramePacker((video.width, video.height))

    for _, frame in video:
        # Pack into multiple regions
        images, transforms = packer.pack(frame, exclude=[], regions=3)
        num_regions = len(images)

        # Scenario 1: Empty detections for each region individually
        for i in range(num_regions):
            unpacked = packer.unpack([], transforms, region_idx=i)
            assert unpacked == [], f"Empty detections should return empty list for region {i}"

        # Scenario 2: Empty detections via unpack_multi
        empty_dets_per_region = [[] for _ in range(num_regions)]
        all_unpacked = packer.unpack_multi(empty_dets_per_region, transforms)
        assert all_unpacked == [], "Empty detections should return empty list from unpack_multi"

        # Scenario 3: Mix of empty and non-empty detections per region
        # Some regions may have detections, others may not
        mixed_dets_per_region = [[] for _ in range(num_regions)]
        # Only add a detection to the first region if we have one
        if num_regions > 0:
            # Create a fake detection within the first region's packed image bounds
            h, w = images[0].shape[:2]
            if h > 10 and w > 10:
                mixed_dets_per_region[0] = [(5, 5, 10, 10)]

        result = packer.unpack_multi(mixed_dets_per_region, transforms)
        # Result should have same number of detections as non-empty regions contributed
        expected_count = sum(len(d) for d in mixed_dets_per_region)
        assert len(result) == expected_count, f"Expected {expected_count} detections, got {len(result)}"

        # Scenario 4: All regions have empty detections (simulating no objects detected)
        all_empty = [[] for _ in range(num_regions)]
        result = packer.unpack_multi(all_empty, transforms)
        assert result == [], "All empty detections should yield empty result"

        break


def _test_multi_region_edge_cases():
    """Test edge cases for multi-region packing."""
    video = IterableVideo(Path("data/testvid.mp4"))
    packer = RandomFramePacker((video.width, video.height), threshold=0.01)

    for _, frame in video:
        # Test regions=0 should raise ValueError
        try:
            packer.pack(frame, regions=0)
            raise AssertionError("Expected ValueError for regions=0")
        except ValueError as e:
            assert "regions must be >= 1" in str(e)

        # Test very large regions value (should be clamped to cell count)
        images, transforms = packer.pack(frame, regions=1000)
        # Should get at most as many regions as there are cells
        assert len(images) <= 1000
        assert len(images) >= 1

        # Test unpack with list transform but no region_idx and non-empty detections
        # (empty detections returns early without validation, which is acceptable)
        try:
            fake_dets = [(0, 0, 10, 10)]
            packer.unpack(fake_dets, transforms, region_idx=None)
            raise AssertionError("Expected ValueError for missing region_idx")
        except ValueError as e:
            assert "region_idx is required" in str(e)

        # Test unpack_multi with mismatched lengths
        try:
            packer.unpack_multi([[]], transforms)  # 1 det list vs N transforms
            if len(transforms) > 1:
                raise AssertionError("Expected ValueError for length mismatch")
        except ValueError as e:
            assert "Length mismatch" in str(e)

        break


@wrapper
def test_annealing_packer_multi_region():
    _test_annealing_packer_multi_region()


@wrapper_jit
def test_annealing_packer_multi_region_jit():
    _test_annealing_packer_multi_region()


@wrapper
def test_random_packer_multi_region():
    _test_random_packer_multi_region()


@wrapper_jit
def test_random_packer_multi_region_jit():
    _test_random_packer_multi_region()


@wrapper
def test_multi_region_backward_compat():
    _test_multi_region_backward_compat()


@wrapper
def test_multi_region_edge_cases():
    _test_multi_region_edge_cases()


@wrapper
def test_multi_region_empty_detections():
    _test_multi_region_empty_detections()


@wrapper_jit
def test_multi_region_empty_detections_jit():
    _test_multi_region_empty_detections()
