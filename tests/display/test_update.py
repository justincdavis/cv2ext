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

from cv2ext import IterableVideo, Display
import numpy as np

from ._utils import VID_LINK, download_youtube_video


def test_update():
    if not os.path.exists("video.mp4"):
        download_youtube_video(VID_LINK, "video.mp4")

    video = IterableVideo("video.mp4")
    display = Display("test", show=False)

    for _, frame in video:
        display.update(frame)

        assert np.all(frame == display.frame)

def test_id():
    if not os.path.exists("video.mp4"):
        download_youtube_video(VID_LINK, "video.mp4")

    video = IterableVideo("video.mp4")
    display = Display("test", show=False)

    for frame_id, frame in video:
        display.update(frame)

        assert np.all(frame == display.frame)
        assert frame_id == display.frameid

def test_call():
    if not os.path.exists("video.mp4"):
        download_youtube_video(VID_LINK, "video.mp4")

    video = IterableVideo("video.mp4")
    display1 = Display("test1", show=False)
    display2 = Display("test2", show=False)

    for _, frame in video:
        display1.update(frame)
        display2(frame)

        assert np.all(frame == display1.frame)
        assert np.all(frame == display2.frame)
        assert np.all(display1.frame == display2.frame)
