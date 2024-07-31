# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: E402, I001
"""
Package containing helpful tools for working with opencv.

Submodules
----------
bboxes
    Submodule containing tools for working with bounding boxes in images.
cli
    Submodule containing command line interface tools.
detection
    Submodule containing tools for performing simple types of detection.
image
    Submodule containing tools for working with images.
io
    Submodule containing tools for working with video and image io.
template
    Submodule containing tools for working with templates in images.
tracking
    Submodule containing tools for tracking objects in videos.
metrics
    Submodule containing tools for working with image metrics.
video
    Submodule containing tools for working with videos.

Classes
-------
Display
    A class for displaying images using a separate thread.
Fourcc
    A class for handling the codecs for video writing.
IterableVideo
    A class for iterating over frames in a video, optionally with threading.
VideoWriter
    A class for writing videos.

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
    PARALLEL: bool = False


_FLAGSOBJ = _FLAGS()


def enable_jit(*, on: bool | None = None, parallel: bool | None = None) -> None:
    """
    Enable just-in-time compilation using Numba for some functions.

    Parameters
    ----------
    on : bool | None
        If True, enable jit. If False, disable jit. If None, enable jit.
    parallel : bool | None
        If True, enable parallel jit. If False, disable parallel jit. If None, disable parallel jit.


    """
    if on is None:
        on = True
    if parallel is None:
        parallel = False
    _FLAGSOBJ.USEJIT = on
    _FLAGSOBJ.PARALLEL = parallel
    _log.info(f"JIT is {'enabled' if on else 'disabled'}; parallel: {parallel}.")


from . import bboxes, detection, image, io, metrics, template, tracking, video
from .io import Display, Fourcc, IterableVideo, VideoWriter

__all__ = [
    "_DELOBJ",
    "_FLAGSOBJ",
    "Display",
    "Fourcc",
    "IterableVideo",
    "VideoWriter",
    "bboxes",
    "cli",
    "detection",
    "enable_jit",
    "image",
    "io",
    "metrics",
    "set_log_level",
    "template",
    "tracking",
    "video",
]
__version__ = "0.0.14"

_log.info(f"Initialized cv2ext with version {__version__}")

from . import cli

_log.info("cv2ext.cli initialized.")

__all__ += ["cli"]
