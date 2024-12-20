# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: ARG001
from __future__ import annotations

import logging
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
    from numba import jit as _jit
except ImportError:

    def _jit(  # type: ignore[misc]
        func: Callable[_P, _R],
        *,
        nopython: bool,
        fastmath: bool,
        parallel: bool,
        nogil: bool,
        cache: bool,
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
        return _jit(  # type: ignore[no-any-return]
            func,
            nopython=True,
            fastmath=FLAGS.FASTMATH,
            parallel=FLAGS.PARALLEL,
            nogil=FLAGS.NOGIL,
            cache=FLAGS.CACHE,
        )
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


def _reset_funcs() -> None:
    # re-compile if needed
    for func in _JIT_FUNCS:
        globals()[func.__name__] = jit(func)


def enable_jit(
    *,
    parallel: bool | None = None,
    fastmath: bool | None = None,
    nogil: bool | None = None,
    cache: bool | None = None,
) -> None:
    """
    Enable just-in-time compilation using Numba for some functions.

    Parameters
    ----------
    parallel : bool, optional
        If True, enable parallel jit.
        Default is False.
    fastmath : bool, optional
        If True, enable fastmath during jit.
        Default is True.
    nogil : bool, optional
        If True, disable the GIL when running jit compiled functions.
        Default is False.
    cache : bool, optional
        IF True, cache jit compiled functions to disk.
        Default is False.

    """
    if parallel is None:
        parallel = False
    if fastmath is None:
        fastmath = True
    if nogil is None:
        nogil = False
    if cache is None:
        cache = False
    FLAGS.JIT = True
    FLAGS.PARALLEL = parallel
    FLAGS.FASTMATH = fastmath
    FLAGS.NOGIL = nogil
    FLAGS.CACHE = cache
    _log.info(f"JIT is enabled: {FLAGS}")

    _reset_funcs()


def disable_jit() -> None:
    """Disable JIT compilation."""
    FLAGS.JIT = False

    _reset_funcs()


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
