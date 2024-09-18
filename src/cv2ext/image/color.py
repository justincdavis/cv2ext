# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Submodule containing color utility functions for images.

Classes
-------
Color
    A class for color representation.

"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self


class Color(Enum):
    """Represents a color in BGR format."""

    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    GREEN = (0, 128, 0)
    ORANGE = (0, 185, 255)
    PINK = (147, 20, 255)
    YELLOW = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    WHITE = (255, 255, 255)
    CYAN = (255, 255, 0)
    DARK_GREEN = (0, 204, 0)
    PURPLE = (51, 0, 51)
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128)
    TEAL = (128, 128, 0)
    OLIVE = (0, 128, 128)
    NAVY = (128, 0, 0)
    LIME = (0, 255, 0)
    AQUA = (212, 255, 127)
    SKY_BLUE = (235, 206, 136)
    INDIGO = (130, 0, 75)
    BROWN = (19, 69, 139)
    TURQUOISE = (208, 224, 64)
    CORAL = (80, 127, 255)
    TAN = (140, 180, 210)
    VIOLET = (238, 130, 238)
    SALMON = (114, 128, 250)
    GOLD = (0, 215, 255)
    SILVER = (192, 192, 192)
    BRONZE = (50, 127, 205)
    BEIGE = (179, 222, 245)
    IVORY = (240, 255, 255)
    MINT = (201, 252, 189)

    @property
    def bgr(self: Self) -> tuple[int, int, int]:
        """
        Get Color as BGR format.

        Returns
        -------
        tuple[int, int, int]
            The BGR representation of the color.

        """
        color: tuple[int, int, int] = self.value
        return color

    @property
    def rgb(self: Self) -> tuple[int, int, int]:
        """
        Get Color as RGB format.

        Returns
        -------
        tuple[int, int, int]
            The RGB representation of the color.

        """
        return self.value[2], self.value[1], self.value[0]

    @property
    def hsv(self: Self) -> tuple[float, float, float]:
        """
        Get Color as HSV format.

        Returns
        -------
        tuple[float, float, float]
            The HSV representation of the color.

        """
        b_prime = self.value[0] / 255.0
        g_prime = self.value[1] / 255.0
        r_prime = self.value[2] / 255.0

        cmax = max(r_prime, g_prime, b_prime)
        cmin = min(r_prime, g_prime, b_prime)

        delta = cmax - cmin

        # compute hue
        if delta == 0:
            h = 0
        elif cmax == r_prime:
            h = 60 * (((g_prime - b_prime) / delta) % 6)
        elif cmax == g_prime:
            h = 60 * ((b_prime - r_prime) / delta + 2)
        else:
            h = 60 * ((r_prime - g_prime) / delta + 4)

        # compute saturation
        s = 0 if cmax == 0 else (delta / cmax)

        # compute value
        v = cmax

        return h, s, v

    @property
    def hsl(self: Self) -> tuple[float, float, float]:
        """
        Get Color as HSL format.

        Returns
        -------
        tuple[float, float, float]
            The HSL representation of the color.

        """
        b_prime = self.value[0] / 255.0
        g_prime = self.value[1] / 255.0
        r_prime = self.value[2] / 255.0

        cmax = max(r_prime, g_prime, b_prime)
        cmin = min(r_prime, g_prime, b_prime)

        delta = cmax - cmin

        # compute hue
        if delta == 0:
            h = 0
        elif cmax == r_prime:
            h = 60 * (((g_prime - b_prime) / delta) % 6)
        elif cmax == g_prime:
            h = 60 * ((b_prime - r_prime) / delta + 2)
        else:
            h = 60 * ((r_prime - g_prime) / delta + 4)

        # compute lightness
        lightness = (cmax + cmin) / 2

        # compute saturation
        s = 0 if delta == 0 else delta / (1 - abs(2 * lightness - 1))

        return h, s, lightness

    @property
    def cmyk(self: Self) -> tuple[float, float, float, float]:
        """
        Get Color as CMYK format.

        Returns
        -------
        tuple[int, int, int, int]
            The CMYK representation of the color.

        """
        b_prime = self.value[0] / 255.0
        g_prime = self.value[1] / 255.0
        r_prime = self.value[2] / 255.0

        k = 1 - max(r_prime, g_prime, b_prime)
        c = (1 - r_prime - k) / (1 - k) if k != 1 else 0
        y = (1 - g_prime - k) / (1 - k) if k != 1 else 0
        m = (1 - b_prime - k) / (1 - k) if k != 1 else 0

        return c, y, m, k

    @property
    def hex(self: Self) -> str:
        """
        Get Color as HEX format.

        Returns
        -------
        str
            The HEX representation of the color.

        """
        return "#{:02x}{:02x}{:02x}".format(*self.value[::-1])
