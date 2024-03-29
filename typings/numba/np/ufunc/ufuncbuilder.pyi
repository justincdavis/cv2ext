"""
This type stub file was generated by pyright.
"""

from numba.core import serialize
from numba.core.descriptors import TargetDescriptor
from numba.core.options import TargetOptions
from numba.core.compiler_lock import global_compiler_lock

_options_mixin = ...
class UFuncTargetOptions(_options_mixin, TargetOptions):
    def finalize(self, flags, options): # -> None:
        ...
    


class UFuncTarget(TargetDescriptor):
    options = UFuncTargetOptions
    def __init__(self) -> None:
        ...
    
    @property
    def typing_context(self): # -> threadsafe_cached_property:
        ...
    
    @property
    def target_context(self): # -> threadsafe_cached_property:
        ...
    


ufunc_target = ...
class UFuncDispatcher(serialize.ReduceMixin):
    """
    An object handling compilation of various signatures for a ufunc.
    """
    targetdescr = ...
    def __init__(self, py_func, locals=..., targetoptions=...) -> None:
        ...
    
    def enable_caching(self): # -> None:
        ...
    
    def compile(self, sig, locals=..., **targetoptions):
        ...
    


_identities = ...
def parse_identity(identity):
    """
    Parse an identity value and return the corresponding low-level value
    for Numpy.
    """
    ...

class _BaseUFuncBuilder:
    def add(self, sig=...):
        ...
    
    def disable_compile(self): # -> None:
        """
        Disable the compilation of new signatures at call time.
        """
        ...
    


class UFuncBuilder(_BaseUFuncBuilder):
    def __init__(self, py_func, identity=..., cache=..., targetoptions=...) -> None:
        ...
    
    def build_ufunc(self):
        ...
    
    def build(self, cres, signature): # -> tuple[list[int | Any], Any, Any]:
        '''Slated for deprecation, use
        ufuncbuilder._build_element_wise_ufunc_wrapper().
        '''
        ...
    


class GUFuncBuilder(_BaseUFuncBuilder):
    def __init__(self, py_func, signature, identity=..., cache=..., targetoptions=..., writable_args=...) -> None:
        ...
    
    @global_compiler_lock
    def build_ufunc(self):
        ...
    
    def build(self, cres): # -> tuple[list[Any], Any, Any]:
        """
        Returns (dtype numbers, function ptr, EnvironmentObject)
        """
        ...
    


