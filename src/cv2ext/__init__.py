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
Display
    A class for displaying images using a separate thread.
IterableVideo
    A class for iterating over frames in a video, optionally with threading.
"""
from __future__ import annotations

# setup the logger before importing anything else
import logging
import os
import sys


# Created from answer by Dennis at:
# https://stackoverflow.com/questions/7621897/python-logging-module-globally
def _setup_logger(level: str | None = None) -> None:
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
    try:
        log_level = level_map[level]
    except KeyError:
        log_level = logging.WARNING

    # create logger
    logger = logging.getLogger(__package__)
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)


def set_log_level(level: str) -> None:
    """
    Set the log level for the oakutils package.

    Parameters
    ----------
    level : str
        The log level to set. One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".

    Raises
    ------
    ValueError
        If the level is not one of the allowed values.

    """
    if level.upper() not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        err_msg = f"Invalid log level: {level}"
        raise ValueError(err_msg)
    _setup_logger(level)


level = os.getenv("OAKUTILS_LOG_LEVEL")
_setup_logger(level)
_log = logging.getLogger(__name__)
if level is not None and level.upper() not in [
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
]:
    _log.warning(f"Invalid log level: {level}. Using default log level: WARNING")

import contextlib
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from typing_extensions import Self


class _DEL:
    def __init__(self: Self) -> None:
        cv2.startWindowThread()
        self._windows: list[str] = []

    def __del__(self: Self) -> None:
        _log.debug("cv2ext is being deleted.")
        for windowname in self._windows:
            _log.debug(f"Deleting window {windowname}.")
            with contextlib.suppress(cv2.error):
                cv2.destroyWindow(windowname)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def logwindow(self: Self, windowname: str) -> None:
        """
        Queue windowname for deletion.

        Parameters
        ----------
        windowname : str
            The name of the window to delete.

        """
        self._windows.append(windowname)


_DELOBJ = _DEL()


from ._display import Display
from ._iterablevideo import IterableVideo

__all__ = ["_DELOBJ", "Display", "IterableVideo", "set_log_level"]
__version__ = "0.0.4"

_log.info(f"Initialized cv2ext with version {__version__}")
