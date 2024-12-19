# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: ARG001
from __future__ import annotations

import logging
from functools import wraps
from typing import TYPE_CHECKING

from typing_extensions import ParamSpec, TypeVar

from ._flags import FLAGS

_P = ParamSpec("_P")
_R = TypeVar("_R")

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from typing_extensions import Self

_log = logging.getLogger(__name__)

try:
    from numba import njit as _jit
except ImportError:

    def _jit(  # type: ignore[misc]
        func: Callable[_P, _R],
        *,
        fastmath: bool,
        parallel: bool,
    ) -> Callable[_P, _R]:
        return func

_JIT_FUNCS: list[Callable] = []


def jit(
    func: Callable[_P, _R],
) -> Callable[_P, _R]:
    """
    Optionally JIT compile a function based on the flags for cv2ext.

    Parameters
    ----------
    func : Callable[_P, _R]
        The function to jit compile

    Returns
    -------
    Callable[_P, _R]
        The JIT compiled or untouched function.

    """
    if FLAGS.JIT:
        _log.debug(f"Marking: {func.__name__} for JIT compilation")
        return _jit(func, fastmath=FLAGS.FASTMATH, parallel=FLAGS.PARALLEL)
    return func


def register_jit(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    Register a function to be re-imported whenever JIT status changes.

    Parameters
    ----------
    func : Callable[_P, _R]
        The function to optionally JIT compile.

    Returns
    -------
    Callable[_P, _R]
        The function passed in

    """
    _JIT_FUNCS.append(func)
    return jit(func)


def enable_jit(*, parallel: bool | None = None, fastmath: bool | None = None) -> None:
    """
    Enable just-in-time compilation using Numba for some functions.

    Parameters
    ----------
    parallel : bool | None
        If True, enable parallel jit. If False, disable parallel jit.
        Default is False.
    fastmath : bool | None
        If True, enable fastmath during jit. If False, disable fastmath.
        Default is True.

    """
    if parallel is None:
        parallel = False
    if fastmath is None:
        fastmath = True
    FLAGS.JIT = True
    FLAGS.PARALLEL = parallel
    FLAGS.FASTMATH = fastmath
    _log.info(f"JIT is enabled; parallel: {parallel}, fastmath: {fastmath}")

    # re-compile if needed
    for func in _JIT_FUNCS:
        globals()[func.__name__] = jit(func)


def disable_jit() -> None:
    """Disable JIT compilation."""
    FLAGS.JIT = False


class _JIT:
    def __init__(self: Self) -> None:
        pass

    def __enter__(self: Self) -> Self:
        enable_jit()
        return self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        disable_jit()


JIT = _JIT()
