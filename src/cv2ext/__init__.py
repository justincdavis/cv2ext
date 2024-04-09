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
# ruff: noqa: E402, I001
"""
Package containing helpful tools for working with opencv.

Submodules
----------
bboxes
    Submodule containing tools for working with bounding boxes in images.
cli
    Submodule containing command line interface tools.
template
    Submodule containing tools for working with templates in images.
metrics
    Submodule containing tools for working with image metrics.

Classes
-------
Display
    A class for displaying images using a separate thread.
IterableVideo
    A class for iterating over frames in a video, optionally with threading.

Functions
---------
set_log_level
    Set the log level for the cv2ext package.
enable_jit
    Enable just-in-time compilation using Numba for some functions.
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
    Set the log level for the cv2ext package.

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


level = os.getenv("CV2EXT_LOG_LEVEL")
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

if not cv2.useOptimized():
    cv2.setUseOptimized(True)

if TYPE_CHECKING:
    from typing_extensions import Self


class _DEL:
    def __init__(self: Self, log: logging.Logger) -> None:
        self._log = log
        self._started = False
        self._windows: list[str] = []
        self._is_windows = os.name == "nt"
        self._osname = "Windows" if self._is_windows else "Unix"
        self._log.debug(f"cv2ext is running on {self._osname}.")

    def __del__(self: Self) -> None:
        self._log.debug("cv2ext is being deleted.")
        for windowname in self._windows:
            self._log.debug(f"Deleting window {windowname}.")
            with contextlib.suppress(cv2.error):
                cv2.destroyWindow(windowname)
        self._log.debug("Deleting all windows.")
        cv2.destroyAllWindows()
        if self._is_windows:
            cv2.waitKey(1)

    def logwindow(self: Self, windowname: str) -> None:
        """
        Queue windowname for deletion.

        Parameters
        ----------
        windowname : str
            The name of the window to delete.

        """
        if not self._started:
            self._log.debug("Starting cv2 window thread.")
            cv2.startWindowThread()
            self._started = True
        self._windows.append(windowname)


_DELOBJ = _DEL(_log)


from dataclasses import dataclass


@dataclass
class _FLAGS:
    """
    Class for storing flags for cv2ext.

    Attributes
    ----------
    USEJIT : bool
        Whether or not to use jit.

    """

    USEJIT: bool = False


_FLAGSOBJ = _FLAGS()


def enable_jit(*, on: bool | None = None) -> None:
    """
    Enable just-in-time compilation using Numba for some functions.

    Parameters
    ----------
    on : bool | None
        If True, enable jit. If False, disable jit. If None, enable jit.

    """
    if on is None:
        on = True
    _FLAGSOBJ.USEJIT = on
    _log.info(f"JIT is {'enabled' if on else 'disabled'}.")


from . import bboxes, metrics, template
from ._display import Display
from ._iterablevideo import IterableVideo

__all__ = [
    "_DELOBJ",
    "_FLAGSOBJ",
    "Display",
    "IterableVideo",
    "bboxes",
    "cli",
    "enable_jit",
    "metrics",
    "set_log_level",
    "template",
]
__version__ = "0.0.10"

_log.info(f"Initialized cv2ext with version {__version__}")

from . import cli

_log.info("cv2ext.cli initialized.")

__all__ += ["cli"]
