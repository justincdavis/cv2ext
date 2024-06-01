Examples for cv2ext
---------------------

A collection of examples showcasing how to use some of the tools.

1. IterableVideo
    - io.iterable_video.py
        Shows how to use the iterable video with and without threading.
    - io.video_writer.py
        IterableVideo used
    - io.video_writer_fourcc.py
        IterableVideo used
2. Display
    - io.display.py
        Shows how to use the display class to get faster IO with threading.
3. Fourcc
    - io.video_writer_fourcc.py
        Shows how to use Fourcc codecs
4. VideoWriter
    - io.video_writer_fourcc.py
        Shows how to write video with different codecs
    - io.video_writer.py
        Shows how to write a video
5. bboxes
    - bboxes.iou
        Shows how to compute iou between two bounding boxes
    - bboxes.meanap
        Shows how to compute the mean average precision between predicted boxes and ground truths
    - bboxes.nms
        Shows how to run non-max-suppression on a set of bounding boxes
6. io
    - io.display
        Shows how to use displays
    - io.iterable_video
        Shows how to use the IterableVideo class
    - io.video_writer_fourcc.py
        Shows IterableVideo, Fourcc, VideoWriter
    - io.video_writer.py
        Shows IterableVideo, VideoWriter
7. metrics
    - metrics.ncc
        Shows how to compute normalized-cross-correlation of two images
8. template
    - template.match_single
        Shows how to match a single template within an image
    - template.match_multiple
        Shows how to get multiple template matches within an image
