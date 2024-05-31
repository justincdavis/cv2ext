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
from __future__ import annotations

from pathlib import Path

from cv2ext import IterableVideo


def test_sequential():
    # get video from dump dir
    video = IterableVideo(Path("data") / "testvid.mp4", use_thread=False)

    prev_id = -1
    for frame_id, _ in video:
        assert prev_id + 1 == frame_id
        prev_id = frame_id


def test_sequential_thread():
    # get video from dump dir
    video = IterableVideo(Path("data") / "testvid.mp4", use_thread=True)

    prev_id = -1
    for frame_id, _ in video:
        assert prev_id + 1 == frame_id
        prev_id = frame_id
