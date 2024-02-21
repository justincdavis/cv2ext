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
# ruff: noqa: E402
"""
Package containing helpful tools for working with opencv.

Classes
-------
IterableVideo
"""
from __future__ import annotations

# setup the logger before importing anything else
import logging
import os
import sys


# Created from answer by Dennis at:
# https://stackoverflow.com/questions/7621897/python-logging-module-globally
def _setup_logger() -> None:
    # get logging level environment variable
    level = os.getenv("SHIFTDETECTOR_LOG_LEVEL")
    if level is not None:
        level = level.upper()
    level_map: dict[str | None, int] = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        None: logging.WARNING,
    }
    log_level = level_map[level]

    # create logger
    logger = logging.getLogger(__package__)
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)


_setup_logger()
_log = logging.getLogger(__name__)


from ._iterablevideo import IterableVideo

__all__ = ["IterableVideo"]
__version__ = "0.0.1"

_log.info(f"Initialized cv2tools with version {__version__}")
