"""
This type stub file was generated by pyright.
"""

from numba.extending import box, lower_builtin, models, register_model, type_callable, unbox
from numba.core import types
from numpy.polynomial.polynomial import Polynomial

@register_model(types.PolynomialType)
class PolynomialModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


@type_callable(Polynomial)
def type_polynomial(context): # -> Callable[..., PolynomialType | None]:
    ...

@lower_builtin(Polynomial, types.Array)
def impl_polynomial1(context, builder, sig, args): # -> Any:
    ...

@lower_builtin(Polynomial, types.Array, types.Array, types.Array)
def impl_polynomial3(context, builder, sig, args): # -> Any:
    ...

@unbox(types.PolynomialType)
def unbox_polynomial(typ, obj, c): # -> NativeValue:
    """
    Convert a Polynomial object to a native polynomial structure.
    """
    ...

@box(types.PolynomialType)
def box_polynomial(typ, val, c):
    """
    Convert a native polynomial structure to a Polynomial object.
    """
    ...
