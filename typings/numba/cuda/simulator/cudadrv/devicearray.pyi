"""
This type stub file was generated by pyright.
"""

from contextlib import contextmanager

'''
The Device Array API is not implemented in the simulator. This module provides
stubs to allow tests to import correctly.
'''
DeviceRecord = ...
from_record_like = ...
errmsg_contiguous_buffer = ...
class FakeShape(tuple):
    '''
    The FakeShape class is used to provide a shape which does not allow negative
    indexing, similar to the shape in CUDA Python. (Numpy shape arrays allow
    negative indexing)
    '''
    def __getitem__(self, k):
        ...
    


class FakeWithinKernelCUDAArray:
    '''
    Created to emulate the behavior of arrays within kernels, where either
    array.item or array['item'] is valid (that is, give all structured
    arrays `numpy.recarray`-like semantics). This behaviour does not follow
    the semantics of Python and NumPy with non-jitted code, and will be
    deprecated and removed.
    '''
    def __init__(self, item) -> None:
        ...
    
    def __getattr__(self, attrname): # -> FakeWithinKernelCUDAArray:
        ...
    
    def __setattr__(self, nm, val): # -> None:
        ...
    
    def __getitem__(self, idx): # -> FakeWithinKernelCUDAArray:
        ...
    
    def __setitem__(self, idx, val): # -> None:
        ...
    
    def __len__(self): # -> int:
        ...
    
    def __array_ufunc__(self, ufunc, method, *args, **kwargs): # -> Any:
        ...
    


class FakeCUDAArray:
    '''
    Implements the interface of a DeviceArray/DeviceRecord, but mostly just
    wraps a NumPy array.
    '''
    __cuda_ndarray__ = ...
    def __init__(self, ary, stream=...) -> None:
        ...
    
    @property
    def alloc_size(self):
        ...
    
    @property
    def nbytes(self):
        ...
    
    def __getattr__(self, attrname): # -> Any:
        ...
    
    def bind(self, stream=...): # -> FakeCUDAArray:
        ...
    
    @property
    def T(self): # -> FakeCUDAArray:
        ...
    
    def transpose(self, axes=...): # -> FakeCUDAArray:
        ...
    
    def __getitem__(self, idx): # -> FakeCUDAArray:
        ...
    
    def __setitem__(self, idx, val):
        ...
    
    def copy_to_host(self, ary=..., stream=...): # -> Any:
        ...
    
    def copy_to_device(self, ary, stream=...): # -> None:
        '''
        Copy from the provided array into this array.

        This may be less forgiving than the CUDA Python implementation, which
        will copy data up to the length of the smallest of the two arrays,
        whereas this expects the size of the arrays to be equal.
        '''
        ...
    
    @property
    def shape(self): # -> FakeShape:
        ...
    
    def ravel(self, *args, **kwargs): # -> FakeCUDAArray:
        ...
    
    def reshape(self, *args, **kwargs): # -> FakeCUDAArray:
        ...
    
    def view(self, *args, **kwargs): # -> FakeCUDAArray:
        ...
    
    def is_c_contiguous(self):
        ...
    
    def is_f_contiguous(self):
        ...
    
    def __str__(self) -> str:
        ...
    
    def __repr__(self): # -> str:
        ...
    
    def __len__(self): # -> int:
        ...
    
    def __eq__(self, other) -> bool:
        ...
    
    def __ne__(self, other) -> bool:
        ...
    
    def __lt__(self, other) -> bool:
        ...
    
    def __le__(self, other) -> bool:
        ...
    
    def __gt__(self, other) -> bool:
        ...
    
    def __ge__(self, other) -> bool:
        ...
    
    def __add__(self, other): # -> FakeCUDAArray:
        ...
    
    def __sub__(self, other): # -> FakeCUDAArray:
        ...
    
    def __mul__(self, other): # -> FakeCUDAArray:
        ...
    
    def __floordiv__(self, other): # -> FakeCUDAArray:
        ...
    
    def __truediv__(self, other): # -> FakeCUDAArray:
        ...
    
    def __mod__(self, other): # -> FakeCUDAArray:
        ...
    
    def __pow__(self, other): # -> FakeCUDAArray:
        ...
    
    def split(self, section, stream=...): # -> list[FakeCUDAArray]:
        ...
    


def array_core(ary):
    """
    Extract the repeated core of a broadcast array.

    Broadcast arrays are by definition non-contiguous due to repeated
    dimensions, i.e., dimensions with stride 0. In order to ascertain memory
    contiguity and copy the underlying data from such arrays, we must create
    a view without the repeated dimensions.

    """
    ...

def is_contiguous(ary): # -> bool:
    """
    Returns True iff `ary` is C-style contiguous while ignoring
    broadcasted and 1-sized dimensions.
    As opposed to array_core(), it does not call require_context(),
    which can be quite expensive.
    """
    ...

def sentry_contiguous(ary): # -> None:
    ...

def check_array_compatibility(ary1, ary2): # -> None:
    ...

def to_device(ary, stream=..., copy=..., to=...): # -> FakeCUDAArray | None:
    ...

@contextmanager
def pinned(arg): # -> Generator[None, Any, None]:
    ...

def mapped_array(*args, **kwargs): # -> FakeCUDAArray:
    ...

def pinned_array(shape, dtype=..., strides=..., order=...): # -> ndarray[Any, Any]:
    ...

def managed_array(shape, dtype=..., strides=..., order=...): # -> ndarray[Any, Any]:
    ...

def device_array(*args, **kwargs): # -> FakeCUDAArray:
    ...

def device_array_like(ary, stream=...): # -> FakeCUDAArray:
    ...

def pinned_array_like(ary): # -> ndarray[Any, Any]:
    ...

def auto_device(ary, stream=..., copy=...): # -> tuple[FakeCUDAArray, Literal[False]] | tuple[FakeCUDAArray | None, Literal[True]]:
    ...

def is_cuda_ndarray(obj): # -> Any | bool:
    "Check if an object is a CUDA ndarray"
    ...

def verify_cuda_ndarray_interface(obj): # -> None:
    "Verify the CUDA ndarray interface for an obj"
    ...

def require_cuda_ndarray(obj): # -> None:
    "Raises ValueError is is_cuda_ndarray(obj) evaluates False"
    ...

