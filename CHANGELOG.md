## 0.0.12 (06-01-2024)

### Added

- tracking Submodule
    - kernels Submodule is available with some generic tracking kernels
    - TrackerInterface, allows trackers to be implemented and follow
        same calling conventions
    - CSKTracker class as a simple computer vision tracking algo
    - MultiTracker class which can use many trackers which
        utilize the TrackerInterface

## 0.0.11 (06-01-2024)

### Added

- VideoWriter
    - Class which makes handling OpenCV VideoWriters easier
        to use. Add context manager and automatic frame
        sizing.

### Improvements

- Improved test layout

## 0.0.10 (04-01-2024)

### Improvements

- Fixed missing documentation

## 0.0.9 (03-25-2024)

### Added

- timeline
    - Add cli for generating timelines of videos

## 0.0.8 (03-13-2024)

### Added

- bboxes
    - Submodule focused on bounding box computations
    - Has basic routines such as iou, iou for many boxes
       non-max-suppression, and mean average precision computation
    - All routines are capable of being accelerated via the JIT
- template
    - Added JIT capability to match_multiple function

### Fixed

- Display
    - Bug when logger gets deleted before global handler.
        Handler now keeps a reference to package logger.

### Improved

- Testing
    - Added wrappers to aid in testing JIT vs. normal
        code execution. Significantly less code in tests

## 0.0.7 (03-07-2024)

### Added

- JIT
    - Added enable_jit function to package, which 
        enables jit compilation using numba for 
        certain routines/algorithms
- Display
    - Improved display window handling

## 0.0.6 (03-03-2024)

### Added

- Display
    - Added stopped and is_alive properties for
        identifying when display has been stopped.
- template submodule
    - Added match_single and match_multiple
- metrics submodule
    - Added ncc (normalized cross correlation)
- CLI
    - Annotate cli program for annotating a video
        with bounding boxes

## 0.0.5 (03-01-2024)

### Added

- Display
    - Class which allows displays to be threaded for
        faster iteration.
- Benchmarking
    - Added a basic benchmarking setup showcasing 
        the speedup achieved if using our thread based
        video reading and display.

## 0.0.4 (02-23-2024)

### Added

- CLI interface
    - resize_video function is first available

## 0.0.3 (02-21-2024)

### Fixed

- Package correctly gets detected as being fully typed.

## 0.0.2 (02-21-2024)

### Fixed

- Docstring for IterableVideo
