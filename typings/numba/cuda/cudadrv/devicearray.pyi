"""
This type stub file was generated by pyright.
"""

import numpy as np
from numba import _devicearray
from numba.cuda.cudadrv import devices

"""
A CUDA ND Array is recognized by checking the __cuda_memory__ attribute
on the object.  If it exists and evaluate to True, it must define shape,
strides, dtype and size attributes similar to a NumPy ndarray.
"""
lru_cache = ...
def is_cuda_ndarray(obj): # -> Any | bool:
    "Check if an object is a CUDA ndarray"
    ...

def verify_cuda_ndarray_interface(obj): # -> None:
    "Verify the CUDA ndarray interface for an obj"
    ...

def require_cuda_ndarray(obj): # -> None:
    "Raises ValueError is is_cuda_ndarray(obj) evaluates False"
    ...

class DeviceNDArrayBase(_devicearray.DeviceArray):
    """A on GPU NDArray representation
    """
    __cuda_memory__ = ...
    __cuda_ndarray__ = ...
    def __init__(self, shape, strides, dtype, stream=..., gpu_data=...) -> None:
        """
        Args
        ----

        shape
            array shape.
        strides
            array strides.
        dtype
            data type as np.dtype coercible object.
        stream
            cuda stream.
        gpu_data
            user provided device memory for the ndarray data buffer
        """
        ...
    
    @property
    def __cuda_array_interface__(self): # -> dict[str, Any]:
        ...
    
    def bind(self, stream=...): # -> Self:
        """Bind a CUDA stream to this object so that all subsequent operation
        on this array defaults to the given stream.
        """
        ...
    
    @property
    def T(self): # -> Self:
        ...
    
    def transpose(self, axes=...): # -> Self:
        ...
    
    @property
    def device_ctypes_pointer(self): # -> c_void_p | Any:
        """Returns the ctypes pointer to the GPU data buffer
        """
        ...
    
    @devices.require_context
    def copy_to_device(self, ary, stream=...): # -> None:
        """Copy `ary` to `self`.

        If `ary` is a CUDA memory, perform a device-to-device transfer.
        Otherwise, perform a a host-to-device transfer.
        """
        ...
    
    @devices.require_context
    def copy_to_host(self, ary=..., stream=...): # -> ndarray[Any, Any] | NDArray[Any]:
        """Copy ``self`` to ``ary`` or create a new Numpy ndarray
        if ``ary`` is ``None``.

        If a CUDA ``stream`` is given, then the transfer will be made
        asynchronously as part as the given stream.  Otherwise, the transfer is
        synchronous: the function returns after the copy is finished.

        Always returns the host array.

        Example::

            import numpy as np
            from numba import cuda

            arr = np.arange(1000)
            d_arr = cuda.to_device(arr)

            my_kernel[100, 100](d_arr)

            result_array = d_arr.copy_to_host()
        """
        ...
    
    def split(self, section, stream=...): # -> Generator[DeviceNDArray, Any, None]:
        """Split the array into equal partition of the `section` size.
        If the array cannot be equally divided, the last section will be
        smaller.
        """
        ...
    
    def as_cuda_arg(self): # -> Any | MemoryPointer:
        """Returns a device memory object that is used as the argument.
        """
        ...
    
    def get_ipc_handle(self): # -> IpcArrayHandle:
        """
        Returns a *IpcArrayHandle* object that is safe to serialize and transfer
        to another process to share the local allocation.

        Note: this feature is only available on Linux.
        """
        ...
    
    def squeeze(self, axis=..., stream=...): # -> DeviceNDArray:
        """
        Remove axes of size one from the array shape.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Subset of dimensions to remove. A `ValueError` is raised if an axis
            with size greater than one is selected. If `None`, all axes with
            size one are removed.
        stream : cuda stream or 0, optional
            Default stream for the returned view of the array.

        Returns
        -------
        DeviceNDArray
            Squeezed view into the array.

        """
        ...
    
    def view(self, dtype): # -> DeviceNDArray:
        """Returns a new object by reinterpretting the dtype without making a
        copy of the data.
        """
        ...
    
    @property
    def nbytes(self): # -> int:
        ...
    


class DeviceRecord(DeviceNDArrayBase):
    '''
    An on-GPU record type
    '''
    def __init__(self, dtype, stream=..., gpu_data=...) -> None:
        ...
    
    @property
    def flags(self): # -> dict[str, bool]:
        """
        For `numpy.ndarray` compatibility. Ideally this would return a
        `np.core.multiarray.flagsobj`, but that needs to be constructed
        with an existing `numpy.ndarray` (as the C- and F- contiguous flags
        aren't writeable).
        """
        ...
    
    @devices.require_context
    def __getitem__(self, item): # -> DeviceRecord | Any | DeviceNDArray:
        ...
    
    @devices.require_context
    def getitem(self, item, stream=...): # -> DeviceRecord | Any | DeviceNDArray:
        """Do `__getitem__(item)` with CUDA stream
        """
        ...
    
    @devices.require_context
    def __setitem__(self, key, value): # -> None:
        ...
    
    @devices.require_context
    def setitem(self, key, value, stream=...): # -> None:
        """Do `__setitem__(key, value)` with CUDA stream
        """
        ...
    


class DeviceNDArray(DeviceNDArrayBase):
    '''
    An on-GPU array type
    '''
    def is_f_contiguous(self): # -> bool:
        '''
        Return true if the array is Fortran-contiguous.
        '''
        ...
    
    @property
    def flags(self): # -> dict[str, bool]:
        """
        For `numpy.ndarray` compatibility. Ideally this would return a
        `np.core.multiarray.flagsobj`, but that needs to be constructed
        with an existing `numpy.ndarray` (as the C- and F- contiguous flags
        aren't writeable).
        """
        ...
    
    def is_c_contiguous(self): # -> bool:
        '''
        Return true if the array is C-contiguous.
        '''
        ...
    
    def __array__(self, dtype=...): # -> ndarray[Any, Any] | ndarray[Any, dtype[Any]]:
        """
        :return: an `numpy.ndarray`, so copies to the host.
        """
        ...
    
    def __len__(self): # -> int:
        ...
    
    def reshape(self, *newshape, **kws): # -> Self:
        """
        Reshape the array without changing its contents, similarly to
        :meth:`numpy.ndarray.reshape`. Example::

            d_arr = d_arr.reshape(20, 50, order='F')
        """
        ...
    
    def ravel(self, order=..., stream=...): # -> Self:
        '''
        Flattens a contiguous array without changing its contents, similar to
        :meth:`numpy.ndarray.ravel`. If the array is not contiguous, raises an
        exception.
        '''
        ...
    
    @devices.require_context
    def __getitem__(self, item): # -> DeviceRecord | Any | Self:
        ...
    
    @devices.require_context
    def getitem(self, item, stream=...): # -> DeviceRecord | Any | Self:
        """Do `__getitem__(item)` with CUDA stream
        """
        ...
    
    @devices.require_context
    def __setitem__(self, key, value): # -> None:
        ...
    
    @devices.require_context
    def setitem(self, key, value, stream=...): # -> None:
        """Do `__setitem__(key, value)` with CUDA stream
        """
        ...
    


class IpcArrayHandle:
    """
    An IPC array handle that can be serialized and transfer to another process
    in the same machine for share a GPU allocation.

    On the destination process, use the *.open()* method to creates a new
    *DeviceNDArray* object that shares the allocation from the original process.
    To release the resources, call the *.close()* method.  After that, the
    destination can no longer use the shared array object.  (Note: the
    underlying weakref to the resource is now dead.)

    This object implements the context-manager interface that calls the
    *.open()* and *.close()* method automatically::

        with the_ipc_array_handle as ipc_array:
            # use ipc_array here as a normal gpu array object
            some_code(ipc_array)
        # ipc_array is dead at this point
    """
    def __init__(self, ipc_handle, array_desc) -> None:
        ...
    
    def open(self): # -> DeviceNDArray:
        """
        Returns a new *DeviceNDArray* that shares the allocation from the
        original process.  Must not be used on the original process.
        """
        ...
    
    def close(self): # -> None:
        """
        Closes the IPC handle to the array.
        """
        ...
    
    def __enter__(self): # -> DeviceNDArray:
        ...
    
    def __exit__(self, type, value, traceback): # -> None:
        ...
    


class MappedNDArray(DeviceNDArrayBase, np.ndarray):
    """
    A host array that uses CUDA mapped memory.
    """
    def device_setup(self, gpu_data, stream=...): # -> None:
        ...
    


class ManagedNDArray(DeviceNDArrayBase, np.ndarray):
    """
    A host array that uses CUDA managed memory.
    """
    def device_setup(self, gpu_data, stream=...): # -> None:
        ...
    


def from_array_like(ary, stream=..., gpu_data=...): # -> DeviceNDArray:
    "Create a DeviceNDArray object that is like ary."
    ...

def from_record_like(rec, stream=..., gpu_data=...): # -> DeviceRecord:
    "Create a DeviceRecord object that is like rec."
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

errmsg_contiguous_buffer = ...
def sentry_contiguous(ary): # -> None:
    ...

def auto_device(obj, stream=..., copy=..., user_explicit=...): # -> tuple[Any, Literal[False]] | tuple[DeviceRecord | DeviceNDArray, Literal[True]]:
    """
    Create a DeviceRecord or DeviceArray like obj and optionally copy data from
    host to device. If obj already represents device memory, it is returned and
    no copy is made.
    """
    ...

def check_array_compatibility(ary1, ary2): # -> None:
    ...

