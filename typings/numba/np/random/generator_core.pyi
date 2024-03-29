"""
This type stub file was generated by pyright.
"""

from numba.core import types
from numba.core.extending import models, register_jitable, register_model

"""
Core Implementations for Generator/BitGenerator Models.
"""
@register_model(types.NumPyRandomBitGeneratorType)
class NumPyRngBitGeneratorModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


_bit_gen_type = ...
@register_model(types.NumPyRandomGeneratorType)
class NumPyRandomGeneratorTypeModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None:
        ...
    


def next_double(bitgen):
    ...

def next_uint32(bitgen):
    ...

def next_uint64(bitgen):
    ...

@register_jitable
def next_float(bitgen): # -> Signature:
    ...

