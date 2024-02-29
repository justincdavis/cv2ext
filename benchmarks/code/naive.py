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
import time

import cv2

from ._utils import download_youtube_video, VID_LINK


def test_naive() -> float:
    if not os.path.exists("video.mp4"):
        download_youtube_video(VID_LINK, "video.mp4")

    video = cv2.VideoCapture("video.mp4")

    t0 = time.perf_counter()
    got = True
    while got:
        got, frame = video.read()
        if not got:
            break
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)
    t1 = time.perf_counter()

    video.release()
    cv2.destroyAllWindows()

    return t1 - t0


if __name__ == "__main__":
    print(test_naive())
