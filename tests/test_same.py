import numpy as np
from tqdm import tqdm
from cv2tools import IterableVideo


VID_LINK = "https://www.youtube.com/watch?v=-DRSruRMZ8o"

def download_youtube_video(url, output_file):
    from pytube import YouTube
    try:
        # Create a YouTube object with the video URL
        yt = YouTube(url)
        # Get the highest resolution stream available
        stream = yt.streams.get_highest_resolution()
        # Download the video to the specified output file
        stream.download(output_path="", filename=output_file)
    except Exception as e:
        # If an exception occurs during the process, print the exception message
        print("An error occurred:", str(e))


def test_video_read():
    download_youtube_video(VID_LINK, "video.mp4")

    # get video from dump dir
    video = IterableVideo("video.mp4")
    video_thread = IterableVideo("video.mp4", use_thread=True)

    assert len(video) == len(video_thread)
    assert video.fps == video_thread.fps
    assert video.size == video_thread.size
    assert video.length == video_thread.length
    assert video.width == video_thread.width
    assert video.height == video_thread.height

    for (frame_id, frame), (frame_id_thread, frame_thread) in zip(video, video_thread):
        assert frame_id == frame_id_thread
        assert frame.shape == frame_thread.shape
        assert frame.dtype == frame_thread.dtype
        assert frame.size == frame_thread.size

        assert np.all(frame == frame_thread)
