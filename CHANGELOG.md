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
