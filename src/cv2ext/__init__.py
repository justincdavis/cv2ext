# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: E402, I001
"""
Package containing helpful tools for working with opencv.

Submodules
----------
:mod:`bboxes`
    Submodule containing tools for working with bounding boxes in images.
:mod:`cli`
    Submodule containing command line interface tools.
:mod:`detection`
    Submodule containing tools for performing simple types of detection.
:mod:`image`
    Submodule containing tools for working with images.
:mod:`io`
    Submodule containing tools for working with video and image io.
:mod:`metrics`
    Submodule containing tools for working with image metrics.
:mod:`research`
    Submodule containing implementations of research papers and methods.
:mod:`template`
    Submodule containing tools for working with templates in images.
:mod:`tracking`
    Submodule containing tools for tracking objects in videos.
:mod:`video`
    Submodule containing tools for working with videos.

Classes
-------
:class:`Display`
    A class for displaying images using a separate thread.
:class:`Fourcc`
    A class for handling the codecs for video writing.
:class:`IterableVideo`
    A class for iterating over frames in a video, optionally with threading.
:class:`VideoWriter`
    A class for writing videos.

Functions
---------
:func:`set_log_level`
    Set the log level for the cv2ext package.
:func:`enable_jit`
    Enable just-in-time compilation using Numba for some functions.
:func:`disable_jit`
    Disable just-in-time compilation using Numba for some functions.
:func:`register_jit`
    Register a function to be just-in-time compiled.

"""

from __future__ import annotations

# setup the logger before importing anything else
import logging
import os
import sys

# import the flags object
from ._flags import FLAGS


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

    if not FLAGS.SETUP_LOG_HANDLER:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(log_level)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

        FLAGS.SETUP_LOG_HANDLER = True


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


_WINDOW_MANAGER = _DEL(_log)


from . import bboxes, detection, image, io, metrics, research, template, tracking, video
from .io import Display, Fourcc, IterableVideo, VideoWriter
from ._jit import JIT, enable_jit, disable_jit, register_jit

__all__ = [
    "FLAGS",
    "JIT",
    "_WINDOW_MANAGER",
    "Display",
    "Fourcc",
    "IterableVideo",
    "VideoWriter",
    "bboxes",
    "cli",
    "detection",
    "disable_jit",
    "enable_jit",
    "image",
    "io",
    "metrics",
    "register_jit",
    "research",
    "set_log_level",
    "template",
    "tracking",
    "video",
]
__version__ = "0.1.1"

_log.info(f"Initialized cv2ext with version {__version__}")

from . import cli

_log.info("cv2ext.cli initialized.")

__all__ += ["cli"]

# # automatically enable the JIT if numba is present
# if FLAGS.FOUND_NUMBA:
#     enable_jit()
