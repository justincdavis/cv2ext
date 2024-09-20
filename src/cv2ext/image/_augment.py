# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import cv2
import numpy as np


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    stride: int = 32,
    color: tuple[int, int, int] = (114, 114, 114),
    *,
    auto: bool | None = None,
    scale_fill: bool | None = None,
    scaleup: bool | None = None,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """
    Resize an image using the letterbox method.

    This method resizes an image to a new shape while maintaining
    the aspect ratio of the original image. The new image is padded
    with a specified color to fill the remaining space. The padding
    can be adjusted to be divisible by a specified stride.

    Parameters
    ----------
    image : np.ndarray
        The image to resize
    new_shape : tuple[int, int]
        The new shape to resize the image to
    stride : int
        The stride to use for padding
    color : tuple[int, int, int]
        The color to use for padding
    auto : bool, optional
        Whether or not to automatically adjust the padding
        to be divisible by the stride
    scale_fill : bool, optional
        Whether or not to scale the image to fill the new shape
    scaleup : bool, optional
        Whether or not to scale the image up if the new shape
        is larger than the original image

    Returns
    -------
    tuple[np.ndarray, tuple[float, float], tuple[float, float]]
        The resized image, the ratio of the new shape to the old shape,
        and the padding values used.

    """
    if auto is None:
        auto = False
    if scale_fill is None:
        scale_fill = False
    if scaleup is None:
        scaleup = True

    shape = image.shape[:2]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = (r, r)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw: float = new_shape[1] - new_unpad[0]
    dh: float = new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (
            new_shape[1] / shape[1],
            new_shape[0] / shape[0],
        )

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    return image, ratio, (dw, dh)
