"""
This type stub file was generated by pyright.
"""

from numba.core.compiler_lock import global_compiler_lock
from numba.core.typing.typeof import typeof_impl

_nrt_mstats = ...
class _Runtime:
    def __init__(self) -> None:
        ...
    
    @global_compiler_lock
    def initialize(self, ctx): # -> None:
        """Initializes the NRT

        Must be called before any actual call to the NRT API.
        Safe to be called multiple times.
        """
        ...
    
    @staticmethod
    def shutdown(): # -> None:
        """
        Shutdown the NRT
        Safe to be called without calling Runtime.initialize first
        """
        ...
    
    @property
    def library(self):
        """
        Return the Library object containing the various NRT functions.
        """
        ...
    
    def meminfo_new(self, data, pyobj):
        """
        Returns a MemInfo object that tracks memory at `data` owned by `pyobj`.
        MemInfo will acquire a reference on `pyobj`.
        The release of MemInfo will release a reference on `pyobj`.
        """
        ...
    
    def meminfo_alloc(self, size, safe=...):
        """
        Allocate a new memory of `size` bytes and returns a MemInfo object
        that tracks the allocation.  When there is no more reference to the
        MemInfo object, the underlying memory will be deallocated.

        If `safe` flag is True, the memory is allocated using the `safe` scheme.
        This is used for debugging and testing purposes.
        See `NRT_MemInfo_alloc_safe()` in "nrt.h" for details.
        """
        ...
    
    def get_allocation_stats(self): # -> nrt_mstats:
        """
        Returns a namedtuple of (alloc, free, mi_alloc, mi_free) for count of
        each memory operations.
        """
        ...
    


MemInfo = ...
@typeof_impl.register(MemInfo)
def typeof_meminfo(val, c): # -> MemInfoPointer:
    ...

rtsys = ...
