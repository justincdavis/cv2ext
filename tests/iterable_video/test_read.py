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
import os

from cv2ext import IterableVideo

from ._utils import VID_LINK, download_youtube_video


def test_read():
    if not os.path.exists("video.mp4"):
        download_youtube_video(VID_LINK, "video.mp4")

    video = IterableVideo("video.mp4", use_thread=False)

    got = True
    counter = 0
    while got:
        got, frame = video.read()
        if got:
            counter += 1

    assert counter == len(video)


def test_read_thread():
    if not os.path.exists("video.mp4"):
        download_youtube_video(VID_LINK, "video.mp4")

    video = IterableVideo("video.mp4", use_thread=True)

    got = True
    counter = 0
    while got:
        got, frame = video.read()
        if got:
            counter += 1

    assert counter == len(video)
