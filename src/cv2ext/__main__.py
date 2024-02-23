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
"""
Command line interface for cv2ext.

Functions
---------
resize_video
    Use --resize_video to resize a video file.
"""
from __future__ import annotations

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for cv2ext.")
    parser.add_argument(
        "--resize_video",
        action="store_true",
        help="Resize a video file.",
    )
    args = parser.parse_args()

    if args.resize_video:
        from cv2ext._cli import resize_video

        resize_video()
