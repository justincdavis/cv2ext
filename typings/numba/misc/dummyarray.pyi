"""
This type stub file was generated by pyright.
"""

Extent = ...
attempt_nocopy_reshape = ...
class Dim:
    """A single dimension of the array

    Attributes
    ----------
    start:
        start offset
    stop:
        stop offset
    size:
        number of items
    stride:
        item stride
    """
    __slots__ = ...
    def __init__(self, start, stop, size, stride, single) -> None:
        ...
    
    def __getitem__(self, item): # -> Dim:
        ...
    
    def get_offset(self, idx):
        ...
    
    def __repr__(self): # -> LiteralString:
        ...
    
    def normalize(self, base): # -> Dim:
        ...
    
    def copy(self, start=..., stop=..., size=..., stride=..., single=...): # -> Dim:
        ...
    
    def is_contiguous(self, itemsize):
        ...
    


def compute_index(indices, dims): # -> int:
    ...

class Element:
    is_array = ...
    def __init__(self, extent) -> None:
        ...
    
    def iter_contiguous_extent(self): # -> Generator[Any, Any, None]:
        ...
    


class Array:
    """A dummy numpy array-like object.  Consider it an array without the
    actual data, but offset from the base data pointer.

    Attributes
    ----------
    dims: tuple of Dim
        describing each dimension of the array

    ndim: int
        number of dimension

    shape: tuple of int
        size of each dimension

    strides: tuple of int
        stride of each dimension

    itemsize: int
        itemsize

    extent: (start, end)
        start and end offset containing the memory region
    """
    is_array = ...
    @classmethod
    def from_desc(cls, offset, shape, strides, itemsize): # -> Self:
        ...
    
    def __init__(self, dims, itemsize) -> None:
        ...
    
    def __repr__(self): # -> str:
        ...
    
    def __getitem__(self, item): # -> Array | Element:
        ...
    
    @property
    def is_c_contig(self): # -> bool:
        ...
    
    @property
    def is_f_contig(self): # -> bool:
        ...
    
    def iter_contiguous_extent(self): # -> Generator[Extent | tuple[Any, Any] | tuple[int, int], Any, None]:
        """ Generates extents
        """
        ...
    
    def reshape(self, *newdims, **kws): # -> tuple[Self, None] | tuple[Self, list[Extent | tuple[Any, Any] | tuple[int, int]]]:
        ...
    
    def squeeze(self, axis=...): # -> tuple[Self, list[Extent | tuple[Any, Any] | tuple[int, int]]]:
        ...
    
    def ravel(self, order=...): # -> tuple[Self, list[Extent | tuple[Any, Any] | tuple[int, int]]]:
        ...
    


def iter_strides_f_contig(arr, shape=...): # -> Generator[Any, Any, None]:
    """yields the f-contiguous strides
    """
    ...

def iter_strides_c_contig(arr, shape=...): # -> Generator[Any, Any, None]:
    """yields the c-contiguous strides
    """
    ...

def is_element_indexing(item, ndim): # -> bool:
    ...

