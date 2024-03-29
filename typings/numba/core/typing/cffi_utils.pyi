"""
This type stub file was generated by pyright.
"""

from numba.core.typing import templates

"""
Support for CFFI. Allows checking whether objects are CFFI functions and
obtaining the pointer and numba signature.
"""
ffi = ...
SUPPORTED = ...
_ool_func_types = ...
_ool_func_ptr = ...
_ffi_instances = ...
def is_ffi_instance(obj): # -> bool:
    ...

def is_cffi_func(obj): # -> Any | bool:
    """Check whether the obj is a CFFI function"""
    ...

def get_pointer(cffi_func): # -> int:
    """
    Get a pointer to the underlying function for a CFFI function as an
    integer.
    """
    ...

_cached_type_map = ...
def map_type(cffi_type, use_record_dtype=...): # -> Signature | RawPointer | CPointer | NestedArray | Record | UnicodeCharSeq | CharSeq | NPTimedelta | NPDatetime:
    """
    Map CFFI type to numba type.

    Parameters
    ----------
    cffi_type:
        The CFFI type to be converted.
    use_record_dtype: bool (default: False)
        When True, struct types are mapped to a NumPy Record dtype.

    """
    ...

def map_struct_to_record_dtype(cffi_type): # -> Record | UnicodeCharSeq | CharSeq | NPTimedelta | NPDatetime | NestedArray:
    """Convert a cffi type into a NumPy Record dtype
    """
    ...

def make_function_type(cffi_func, use_record_dtype=...): # -> ExternalFunctionPointer:
    """
    Return a Numba type for the given CFFI function pointer.
    """
    ...

registry = ...
@registry.register
class FFI_from_buffer(templates.AbstractTemplate):
    key = ...
    def generic(self, args, kws): # -> Signature | None:
        ...
    


@registry.register_attr
class FFIAttribute(templates.AttributeTemplate):
    key = ...
    def resolve_from_buffer(self, ffi): # -> BoundFunction:
        ...
    


def register_module(mod): # -> None:
    """
    Add typing for all functions in an out-of-line CFFI module to the typemap
    """
    ...

def register_type(cffi_type, numba_type): # -> None:
    """
    Add typing for a given CFFI type to the typemap
    """
    ...

