# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self


def detect_blobs(
    image: np.ndarray,
    kernel_size: int = 9,
    sigma: float = 2.0,
    max_area: float = 0.9,
    min_area: float = 0.05,
    *,
    use_blur: bool | None = None,
    is_rgb: bool | None = None,
    filter_size: bool | None = None,
) -> list[tuple[int, int, int, int]]:
    """
    Detect blobs in an image.

    Blob detection is performed using OpenCVs,
    findContours function. Each contour is identified
    and then a bounding box is created for each one.

    Parameters
    ----------
    image : np.ndarray
        The image to detect blobs in.
    kernel_size : int
        The size of the kernel for the Gaussian blur.
    sigma : float
        The standard deviation for the Gaussian blur.
    max_area : float
        The maximum area a bounding box can be relative to the image.
        Default is 0.9, so bounding boxes larger than 90% of the image
        will be removed from final result if filtering enabled.
    min_area : float
        The minimum area a bounding box can be relative to the image.
        Default is 0.01, so bounding boxes smaller than 5% of the image
        will be removed from final result if filtering enabled.
    use_blur : bool, optional
        Whether or not to use a Gaussian blur on the image.
        By default, blur will be used.
    is_rgb : bool, optional
        Whether or not the image is in RGB format.
        Pass True if input image is RGB instead of BGR (OpenCV default).
    filter_size : bool, optional
        Whether or not to filter the final bounding boxes
        by the area of the bounding box.

    Returns
    -------
    list[tuple[int, int, int, int]]
        A list of bounding boxes for each blob.

    """
    # polish parameters
    if use_blur is None:
        use_blur = True
    if filter_size is None:
        filter_size = False

    # image info
    img_height, img_width = image.shape[:2]
    img_area = img_height * img_width

    # color conversion
    conversion = cv2.COLOR_RGB2GRAY if is_rgb else cv2.COLOR_BGR2GRAY
    gray = cv2.cvtColor(image, conversion)
    if use_blur:
        gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # actual contour detection
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # create bounding boxes
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if filter_size:
            area = w * h
            if area / img_area > max_area or area / img_area < min_area:
                continue
        bboxes.append((x, y, x + w, y + h))

    return bboxes


class BlobDetector:
    """Class for detecting blobs in images."""

    def __init__(
        self: Self,
        kernel_size: int = 9,
        sigma: float = 2.0,
        max_area: float = 0.9,
        min_area: float = 0.05,
        *,
        use_blur: bool | None = None,
        is_rgb: bool | None = None,
        filter_size: bool | None = None,
    ) -> None:
        """
        Create a new BlobDetector.

        Parameters
        ----------
        kernel_size : int
            The size of the kernel for the Gaussian blur.
        sigma : float
            The standard deviation for the Gaussian blur.
        max_area : float
            The maximum area a bounding box can be relative to the image.
            Default is 0.9, so bounding boxes larger than 90% of the image
            will be removed from final result if filtering enabled.
        min_area : float
            The minimum area a bounding box can be relative to the image.
            Default is 0.01, so bounding boxes smaller than 5% of the image
            will be removed from final result if filtering enabled.
        use_blur : bool, optional
            Whether or not to use a Gaussian blur on the image.
            By default, blur will be used.
        is_rgb : bool, optional
            Whether or not the image is in RGB format.
            Pass True if input image is RGB instead of BGR (OpenCV default).
        filter_size : bool, optional
            Whether or not to filter the final bounding boxes
            by the area of the bounding box.

        """
        self._kernel_size = kernel_size
        self._sigma = sigma
        self._max_area = max_area
        self._min_area = min_area
        self._use_blur = use_blur
        self._is_rgb = is_rgb
        self._filter_size = filter_size

    def __call__(self: Self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detect blobs in an image.

        Parameters
        ----------
        image : np.ndarray
            The image to detect blobs in.

        Returns
        -------
        list[tuple[int, int, int, int]]
            A list of bounding boxes for each blob.

        """
        return self.detect(image)

    def detect(self: Self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detect blobs in an image.

        Parameters
        ----------
        image : np.ndarray
            The image to detect blobs in.

        Returns
        -------
        list[tuple[int, int, int, int]]
            A list of bounding boxes for each blob.

        """
        return detect_blobs(
            image,
            self._kernel_size,
            self._sigma,
            self._max_area,
            self._min_area,
            use_blur=self._use_blur,
            is_rgb=self._is_rgb,
            filter_size=self._filter_size,
        )
