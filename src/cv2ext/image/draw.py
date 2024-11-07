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
from functools import partial
from typing import TYPE_CHECKING

import cv2
import numpy as np

from cv2ext.bboxes._constrain import constrain

from .color import Color

if TYPE_CHECKING:
    from collections.abc import Callable

_log = logging.getLogger(__name__)


def _opacity(
    image: np.ndarray,
    call: Callable[[np.ndarray], np.ndarray],
    opacity: float,
    bounds: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    # type hints
    shapes: np.ndarray
    mask: np.ndarray

    # if no bounds provided default case, no optimization
    if not bounds:
        # draw shapes
        shapes = np.zeros_like(image, np.uint8)
        shapes = call(shapes)

        # generate mask and blend
        mask = shapes.astype(bool)
        image[mask] = cv2.addWeighted(image, 1 - opacity, shapes, opacity, 0)[mask]

        return image

    # if bounds have been passes, can optimize copies and image ops
    # get roi from bounds
    x1, y1, x2, y2 = bounds
    roi = image[y1:y2, x1:x2]

    # draw shapes
    shapes = np.zeros_like(image, np.uint8)
    shapes = call(shapes)
    shapes = shapes[y1:y2, x1:x2]

    # generate mask and blend
    mask = shapes.astype(bool)
    image[y1:y2, x1:x2][mask] = cv2.addWeighted(
        roi,
        1 - opacity,
        shapes,
        opacity,
        0,
    )[mask]

    return image


def rectangle(
    image: np.ndarray,
    p1: tuple[int, int] | tuple[int, int, int, int],
    p2: tuple[int, int] | None = None,
    color: Color | tuple[int, int, int] = Color.RED,
    thickness: int = 2,
    linetype: int = cv2.LINE_8,
    opacity: float | None = None,
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
    opacity : float, optional
        The opacity to use for drawing. By default None or 100% opacity.
        Opacity should be in the range [0.0, 1.0]
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
        If a valid combination from p1/p2 cannot be achieved.

    """
    canvas = image
    if copy:
        canvas = image.copy()

    if isinstance(color, Color):
        color = color.value

    call: Callable[[np.ndarray], np.ndarray]
    bounds: tuple[int, int, int, int]
    if len(p1) == 4:
        if p2 is not None:
            _log.warning(
                "p2 is provided, but p1 is a full rectangle. p2 will be ignored.",
            )
        point1 = p1[:2]
        point2 = p1[2:]
        call = partial(
            cv2.rectangle,
            pt1=point1,
            pt2=point2,
            color=color,
            thickness=thickness,
            lineType=linetype,
        )
        bounds = p1
    elif len(p1) == 2:
        if p2 is None:
            err_msg = "If p1 is a single point, then p2 must be provided."
            raise ValueError(err_msg)
        call = partial(
            cv2.rectangle,
            pt1=p1,
            pt2=p2,
            color=color,
            thickness=thickness,
            lineType=linetype,
        )
        bounds = (*p1, *p2)
    else:
        err_msg = "Could not get a valid combination of points from p1/p2."
        raise ValueError(err_msg)

    if opacity is not None:
        # add thickness to bounds and constrain
        bounds = (
            bounds[0] - thickness,
            bounds[1] - thickness,
            bounds[2] + thickness,
            bounds[3] + thickness,
        )
        h, w = canvas.shape[:2]
        bounds = constrain(bounds, (w, h))
        return _opacity(canvas, call, opacity)
    return call(canvas)  # type: ignore[return-value]


def circle(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: Color | tuple[int, int, int] = Color.RED,
    thickness: int = 2,
    linetype: int = cv2.LINE_8,
    opacity: float | None = None,
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
    opacity : float, optional
        The opacity to use for drawing. By default None or 100% opacity.
        Opacity should be in the range [0.0, 1.0]
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

    call: Callable[[np.ndarray], np.ndarray] = partial(
        cv2.circle,
        center=center,
        radius=radius,
        color=color,
        thickness=thickness,
        lineType=linetype,
    )

    if opacity is not None:
        # add thickness to bounds and constrain
        x, y = center
        bounds = (
            x - thickness - radius,
            y - thickness - radius,
            x + thickness + radius,
            y + thickness + radius,
        )
        h, w = canvas.shape[:2]
        bounds = constrain(bounds, (w, h))
        return _opacity(canvas, call, opacity)
    return call(canvas)  # type: ignore[return-value]


def text(
    image: np.ndarray,
    text: str,
    p: tuple[int, int],
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1.0,
    color: Color | tuple[int, int, int] = Color.RED,
    thickness: int = 2,
    linetype: int = cv2.LINE_8,
    opacity: float | None = None,
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
    opacity : float, optional
        The opacity to use for drawing. By default None or 100% opacity.
        Opacity should be in the range [0.0, 1.0]
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

    call: Callable[[np.ndarray], np.ndarray] = partial(
        cv2.putText,
        text=text,
        org=p,
        fontFace=font,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
        lineType=linetype,
        bottomLeftOrigin=bottom_left_origin,
    )

    return _opacity(canvas, call, opacity) if opacity is not None else call(canvas)
