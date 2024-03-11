"""
This type stub file was generated by pyright.
"""

from numba.extending import box, models, register_model, typeof_impl, unbox
from numba.core.imputils import lower_cast, lower_constant
from numba.core.ccallback import CFunc
from numba.core import types
from numba.core.types import FunctionPrototype, FunctionType, UndefinedFunctionType, WrapperAddressProtocol
from numba.core.dispatcher import Dispatcher

"""Provides Numba type, FunctionType, that makes functions as
instances of a first-class function type.
"""
@typeof_impl.register(WrapperAddressProtocol)
@typeof_impl.register(CFunc)
def typeof_function_type(val, c): # -> FunctionType:
    ...

@register_model(FunctionPrototype)
class FunctionProtoModel(models.PrimitiveModel):
    """FunctionProtoModel describes the signatures of first-class functions
    """
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@register_model(FunctionType)
@register_model(UndefinedFunctionType)
class FunctionModel(models.StructModel):
    """FunctionModel holds addresses of function implementations
    """
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@lower_constant(types.Dispatcher)
def lower_constant_dispatcher(context, builder, typ, pyval):
    ...

@lower_constant(FunctionType)
def lower_constant_function_type(context, builder, typ, pyval): # -> Any:
    ...

def lower_get_wrapper_address(context, builder, func, sig, failure_mode=...):
    """Low-level call to _get_wrapper_address(func, sig).

    When calling this function, GIL must be acquired.
    """
    ...

@unbox(FunctionType)
def unbox_function_type(typ, obj, c): # -> NativeValue:
    ...

@box(FunctionType)
def box_function_type(typ, val, c):
    ...

@lower_cast(UndefinedFunctionType, FunctionType)
def lower_cast_function_type_to_function_type(context, builder, fromty, toty, val):
    ...

@lower_cast(types.Dispatcher, FunctionType)
def lower_cast_dispatcher_to_function_type(context, builder, fromty, toty, val): # -> Any:
    ...
