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
"""Example showcasing how to use the IterableVideo class."""
from __future__ import annotations

from cv2ext import Fourcc, IterableVideo, VideoWriter, set_log_level

if __name__ == "__main__":
    set_log_level("DEBUG")
    # create an IterableVideo object
    video = IterableVideo("video.mp4")

    # can create a VideoWriter with a wide variety of Fourcc codecs
    print(f"Available codecs: {len(list(Fourcc))}")
    writer = None
    for fourcc in [Fourcc.H264, Fourcc.XVID, Fourcc.MP4V, Fourcc.mp4v]:
        writer = VideoWriter("output.mp4", fourcc=fourcc)

    with writer as writer:
        # iterate over the video
        for _, frame in video:
            writer.write(frame)
            # print(f"Frame {frame_id}: {frame.shape}")
