# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import Callable

import cv2ext


def wrapper(func: Callable) -> Callable:
    def inner(*args, **kwargs):
        # importlib.reload(cv2ext)

        # result = func(*args, **kwargs)

        # importlib.reload(cv2ext)

        # return result
        return func(*args, **kwargs)

    return inner


def wrapper_jit(func: Callable) -> Callable:
    def inner(*args, **kwargs):
        # importlib.reload(cv2ext)
        # cv2ext.enable_jit()

        # result = func(*args, **kwargs)

        # importlib.reload(cv2ext)

        # return result
        with cv2ext.JIT:
            return func(*args, **kwargs)

    return inner
