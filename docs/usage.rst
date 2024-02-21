Usage
=====

The cv2ext module simply contains tools. The entire workflow is to import and use them as needed.

- Using the `cv2ext.IterableVideo` object:

   .. code-block:: python
   
        from cv2ext import IterableVideo

        # create an IterableVideo object
        video = IterableVideo("path/to/video.mp4")

        # iterate over the video
        for frame_id, frame in video:
            print(f"Frame {frame_id}: {frame.shape}")
