# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule containing wrapper/utility functions for drawing on images.

Functions
---------
rectangle
    Wrapper around cv2.rectangle with autofilled args and flexible types.
text
    Wrapper around cv2.putText with autofilled args.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2

from .color import Color

if TYPE_CHECKING:
    import numpy as np

_log = logging.getLogger(__name__)


def rectangle(
    image: np.ndarray,
    p1: tuple[int, int] | tuple[int, int, int, int],
    p2: tuple[int, int] | None = None,
    color: Color | tuple[int, int, int] = Color.RED,
    thickness: int = 2,
    linetype: int = cv2.LINE_8,
    *,
    copy: bool | None = None,
) -> np.ndarray:
    """
    cv2.rectangle with autofilled args and flexible types.

    While the image is returned from this function, the rectangle is drawn
    in memory, so the image is modified in place. Thus, the return value
    can be ignored unless the copy parameter is set to True.

    Parameters
    ----------
    image : np.ndarray
        The image to draw the rectangle on.
    p1 : tuple[int, int] | tuple[int, int, int, int]
        The top-left point of the rectangle or the full rectangle.
        If the full rectangle is given, then p2 should be None, but
        if provided it will be ignored.
        The rectangle should be in x1, y1, x2, y2 format.
    p2 : tuple[int, int], optional
        The bottom-right point of the rectangle.
        If this is provided, then p1 should be the top-left point.
    color : Color | tuple[int, int, int], optional
        The color to draw the rectangle.
        In BGR format and the default is Color.RED or (0, 0, 255).
    thickness : int, optional
        The thickness of the rectangle lines.
        A negative value such as cv2.FILLED will fill the rectangle.
        Default is 2.
    linetype : int, optional
        The type of line to draw.
        Default is cv2.LINE_8.
        The options are cv2.FILLED, cv2.LINE_4, cv2.LINE_8, cv2.LINE_AA.
    copy : bool, optional
        Whether or not to draw on a copy of the image and return that.
        Default is False.

    Returns
    -------
    np.ndarray
        The image with the rectangle drawn.

    Raises
    ------
    ValueError
        If p1 is a single point and p2 is not provided.

    """
    canvas = image
    if copy:
        canvas = image.copy()

    if isinstance(color, Color):
        color = color.value

    if len(p1) == 4:
        if p2 is not None:
            _log.warning(
                "p2 is provided, but p1 is a full rectangle. p2 will be ignored.",
            )
        point1 = p1[:2]
        point2 = p1[2:]
        cv2.rectangle(canvas, point1, point2, color, thickness, linetype)
    elif len(p1) == 2:
        if p2 is None:
            err_msg = "If p1 is a single point, then p2 must be provided."
            raise ValueError(err_msg)
        cv2.rectangle(canvas, p1, p2, color, thickness, linetype)

    return canvas


def circle(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: Color | tuple[int, int, int] = Color.RED,
    thickness: int = 2,
    linetype: int = cv2.LINE_8,
    *,
    copy: bool | None = None,
) -> np.ndarray:
    """
    cv2.circle with autofilled args.

    While the image is returned from this function, the circle is drawn
    in memory, so the image is modified in place. Thus, the return value
    can be ignored unless the copy parameter is set to True.

    Parameters
    ----------
    image : np.ndarray
        The image to draw the circle on.
    center : tuple[int, int]
        The center of the circle.
    radius : int
        The radius of the circle.
    color : Color | tuple[int, int, int], optional
        The color to draw the circle.
        In BGR format and the default is Color.RED or (0, 0, 255).
    thickness : int, optional
        The thickness of the circle lines.
        A negative value such as cv2.FILLED will fill the circle.
        Default is 2.
    linetype : int, optional
        The type of line to draw.
        Default is cv2.LINE_8.
        The options are cv2.FILLED, cv2.LINE_4, cv2.LINE_8, cv2.LINE_AA.
    copy : bool, optional
        Whether or not to draw on a copy of the image and return that.
        Default is False.

    Returns
    -------
    np.ndarray
        The image with the circle drawn.

    """
    canvas = image
    if copy:
        canvas = image.copy()

    if isinstance(color, Color):
        color = color.value

    cv2.circle(canvas, center, radius, color, thickness, linetype)

    return canvas


def text(
    image: np.ndarray,
    text: str,
    p: tuple[int, int],
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1.0,
    color: Color | tuple[int, int, int] = Color.RED,
    thickness: int = 2,
    linetype: int = cv2.LINE_8,
    *,
    copy: bool | None = None,
    bottom_left_origin: bool | None = None,
) -> np.ndarray:
    """
    cv2.putText with autofilled args.

    While the image is returned from this function, the text is drawn
    in memory, so the image is modified in place. Thus, the return value
    can be ignored unless the copy parameter is set to True.

    Parameters
    ----------
    image : np.ndarray
        The image to draw the text on.
    text : str
        The text to draw on the image.
    p : tuple[int, int]
        The origin of the text.
        By default this is the top-left corner of the text.
        When bottom_left_origin is set to True, this is the
        bottom-left corner of the text.
    font : int, optional
        The font to use for the text.
        Default is cv2.FONT_HERSHEY_SIMPLEX.
        Options are: cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN,
        cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX.
        cv2.FONT_ITALIC can be added to any of these options.
    font_scale : float, optional
        The size of the font.
        Default is 1.0.
    color : Color, tuple[int, int, int], optional
        The color of the text.
        Default is Color.RED or (0, 0, 255).
    thickness : int, optional
        The thickness of the text.
        Default is 2.
    linetype : int, optional
        The type of line to draw.
        Default is cv2.LINE_8.
        The options are cv2.FILLED, cv2.LINE_4, cv2.LINE_8, cv2.LINE_AA.
    copy : bool, optional
        Whether or not to draw on a copy of the image and return that.
        Default is False.
    bottom_left_origin : bool, optional
        Whether or not the origin is the bottom-left corner of the text.
        Default is False.

    Returns
    -------
    np.ndarray
        The image with the text drawn.

    """
    canvas = image
    if copy:
        canvas = image.copy()
    if bottom_left_origin is None:
        bottom_left_origin = False

    if isinstance(color, Color):
        color = color.value

    cv2.putText(
        canvas,
        text,
        p,
        font,
        font_scale,
        color,
        thickness,
        linetype,
        bottom_left_origin,
    )

    return canvas
