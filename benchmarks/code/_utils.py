# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
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
