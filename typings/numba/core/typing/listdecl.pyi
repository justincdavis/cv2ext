"""
This type stub file was generated by pyright.
"""

import operator
from numba.core import types
from .templates import AbstractTemplate, AttributeTemplate, bound_function

registry = ...
infer = ...
infer_global = ...
infer_getattr = ...
@infer_global(list)
class ListBuiltin(AbstractTemplate):
    def generic(self, args, kws): # -> Signature | None:
        ...
    


@infer_getattr
class ListAttribute(AttributeTemplate):
    key = types.List
    @bound_function("list.append")
    def resolve_append(self, list, args, kws): # -> Signature | None:
        ...
    
    @bound_function("list.clear")
    def resolve_clear(self, list, args, kws): # -> Signature:
        ...
    
    @bound_function("list.extend")
    def resolve_extend(self, list, args, kws): # -> Signature | None:
        ...
    
    @bound_function("list.insert")
    def resolve_insert(self, list, args, kws): # -> Signature | None:
        ...
    
    @bound_function("list.pop")
    def resolve_pop(self, list, args, kws): # -> Signature | None:
        ...
    


@infer_global(operator.add)
class AddList(AbstractTemplate):
    def generic(self, args, kws): # -> Signature | None:
        ...
    


@infer_global(operator.iadd)
class InplaceAddList(AbstractTemplate):
    def generic(self, args, kws): # -> Signature | None:
        ...
    


@infer_global(operator.mul)
class MulList(AbstractTemplate):
    def generic(self, args, kws): # -> Signature | None:
        ...
    


@infer_global(operator.imul)
class InplaceMulList(MulList):
    ...


class ListCompare(AbstractTemplate):
    def generic(self, args, kws): # -> Signature | None:
        ...
    


@infer_global(operator.eq)
class ListEq(ListCompare):
    ...


