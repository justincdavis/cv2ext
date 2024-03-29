"""
This type stub file was generated by pyright.
"""

import math
import operator
from numba.core import types
from numba.types import float32

registry = ...
lower = ...
booleans = ...
unarys = ...
unarys_fastmath = ...
binarys = ...
binarys_fastmath = ...
@lower(math.isinf, types.Integer)
@lower(math.isnan, types.Integer)
def math_isinf_isnan_int(context, builder, sig, args):
    ...

@lower(operator.truediv, types.float32, types.float32)
def maybe_fast_truediv(context, builder, sig, args):
    ...

@lower(math.isfinite, types.Integer)
def math_isfinite_int(context, builder, sig, args):
    ...

@lower(math.sin, types.float16)
def fp16_sin_impl(context, builder, sig, args):
    ...

@lower(math.cos, types.float16)
def fp16_cos_impl(context, builder, sig, args):
    ...

@lower(math.log, types.float16)
def fp16_log_impl(context, builder, sig, args):
    ...

@lower(math.log10, types.float16)
def fp16_log10_impl(context, builder, sig, args):
    ...

@lower(math.log2, types.float16)
def fp16_log2_impl(context, builder, sig, args):
    ...

@lower(math.exp, types.float16)
def fp16_exp_impl(context, builder, sig, args):
    ...

@lower(math.floor, types.float16)
def fp16_floor_impl(context, builder, sig, args):
    ...

@lower(math.ceil, types.float16)
def fp16_ceil_impl(context, builder, sig, args):
    ...

@lower(math.sqrt, types.float16)
def fp16_sqrt_impl(context, builder, sig, args):
    ...

@lower(math.fabs, types.float16)
def fp16_fabs_impl(context, builder, sig, args):
    ...

@lower(math.trunc, types.float16)
def fp16_trunc_impl(context, builder, sig, args):
    ...

def impl_boolean(key, ty, libfunc): # -> None:
    ...

def get_lower_unary_impl(key, ty, libfunc): # -> Callable[..., Any]:
    ...

def get_unary_impl_for_fn_and_ty(fn, ty): # -> Callable[..., Any]:
    ...

def impl_unary(key, ty, libfunc): # -> None:
    ...

def impl_unary_int(key, ty, libfunc): # -> None:
    ...

def get_lower_binary_impl(key, ty, libfunc): # -> Callable[..., Any]:
    ...

def get_binary_impl_for_fn_and_ty(fn, ty): # -> Callable[..., Any]:
    ...

def impl_binary(key, ty, libfunc): # -> None:
    ...

def impl_binary_int(key, ty, libfunc): # -> None:
    ...

def impl_pow_int(ty, libfunc): # -> None:
    ...

def impl_modf(ty, libfunc): # -> None:
    ...

def impl_frexp(ty, libfunc): # -> None:
    ...

def impl_ldexp(ty, libfunc): # -> None:
    ...

def impl_tanh(ty, libfunc): # -> None:
    ...

def cpow_implement(fty, cty): # -> None:
    ...

