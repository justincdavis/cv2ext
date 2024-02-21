from pytube import YouTube

VID_LINK = "https://www.youtube.com/watch?v=-DRSruRMZ8o"

def download_youtube_video(url, output_file):
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
