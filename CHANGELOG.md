## 0.1.0 (12-05-2024)

### Fixed

- image.letterbox
    new_shape was incorrectly using (height, width) format

## 0.0.25 (11-07-2024)

### Added

- bboxes.filter_bboxes_by_region
    Get the bounding boxes from a list of bounding boxes
    which are contained inside the region

### Changed

- bboxes.nms now has conf and classid switched
    in the bounding box entries.

## 0.0.24 (11-02-2024)

### Added

- opacity argument to all functions in image.draw
- image.rescale
    Rescale an image from [0:255] to a different range.
    Preserves scale within range.

### Changed

- Display.stopped now looks at whether the stop
    key has been pressed. When stopped is called
    the value will reset. Enables easier use in
    use control flows. Display.is_alive still exists
    and holds the same function as the old
    stopped and is_alive (is_alive is unchanged, just
    preferred now.)

### Improved

- bboxes.bounding now raises ValueError if
    the sequence is empty.

## 0.0.23 (10-28-2024)

### Added

- image.patch
    Generate patches of a fixed size across and image.

### Changed

- image.divide
    Now returns the subimages and offsets

## 0.0.22 (10-09-2024)

### Fixed

- image.divide
    Docstring error, added test case coverage

## 0.0.21 (10-09-2024)

### Added

- image.divide
    Divides an image into regions via the number
    of rows and columns given.
- bboxes.scale and bboxes.scale_many
    Scale bounding boxes from one image size
    to another.

## 0.0.20 (09-24-2024)

### Fixed

- A Display would be created (but unused) if
    show was set to False in VideoWriter

## 0.0.19 (09-20-2024)

### Added

- image.letterbox function
    - Resize an image using the letterbox method

## 0.0.18 (09-18-2024)

### Added

- image.color submodule
    - Color enum for getting BGR colors and
        the color in other formats
- Added nxyxy and nxywh formats to the bboxes
    submodule.

## Fixed

- yolo format in bboxes submodules was incorrectly
    using nxywh format. Renamed and addedd
    correct yolo format.

## 0.0.17 (09-15-2024)

### Added

- image.draw submodule
    - Provides wrappers around some basic cv2
        drawing functions. Primary focuses on
        providing general auto-filling of
        arguments
- io.VideoWriter now has show flag on initialization.
    Will create an io.Display and display frames
    as they are written.

## 0.0.16 (08-12-2024)

### Added

- Tiling functions to the image submodule
    - Create an image using tiles or iterate over an image
        getting progessively tiled.
    - create_tiled_image
    - image_tiler

### Improved

- Display now has enter and exit statements so it can be
    used with the 'with' statement.

## 0.0.15 (07-31-2024)

### Added

- resize_image and convert_video_color cli

### Fixed

- Bug in video_from_images cli where strings were not
    converted to Paths correctly.

## 0.0.14 (07-18-2024)

### Added

- detection Submodule
    - BlobDetector and detect_blobs
- bboxes Submodule
    - xyxy, xywh, and yolo conversion methods
    - draw_bboxes
    - score_bbox and score_bboxes
        - Simple bbox scoring methods based on euclidean distance
            in pixels and area differences.
            Useful for assigning confidence scores to bboxes without
            scores when there is a target bounding box.
- video Submodule
    - create_timeline and video_from_images

## 0.0.13 (06-15-2024)

### Added

- tracking Submodule
    - Added the KLTTracker implementation

## 0.0.12 (06-01-2024)

### Added

- tracking Submodule
    - cv_trackers, all tracking algorithms available through OpenCV
        are wrapped in the same interface with common errors handled
        better than cv2.errors.
    - Tracker, generic class which can use all tracking algorithms
    - MultiTracker, generic class allowing many objects to be tracked
        at once, using the underlying single object trackers.
        Uses threads to achieve parallelism.
    - TrackerType, enum of all possible tracking algorithms
    - AbstractTracker, AbstractMultiTracker, CVTrackerInterface
        Allow custom implementations, and make wrapping OpenCV
        trackers easier.
- xyxy_to_xywh and xywh_to_xyxy in bboxes submodule
    - Allow changing bounding box format
- constrain in bboxes submodule
    - Restricts coordinates of bounding boxes to an image size

### Changed

- Changed license to MIT

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
