"""
This type stub file was generated by pyright.
"""

from numba import cuda, types
from numba.core.extending import overload_attribute
from numba.cuda.extending import intrinsic

@intrinsic
def grid(typingctx, ndim): # -> tuple[Signature, Callable[..., Any | list[Any] | Constant | None]]:
    '''grid(ndim)

    Return the absolute position of the current thread in the entire grid of
    blocks.  *ndim* should correspond to the number of dimensions declared when
    instantiating the kernel. If *ndim* is 1, a single integer is returned.
    If *ndim* is 2 or 3, a tuple of the given number of integers is returned.

    Computation of the first integer is as follows::

        cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    and is similar for the other two indices, but using the ``y`` and ``z``
    attributes.
    '''
    ...

@intrinsic
def gridsize(typingctx, ndim): # -> tuple[Signature, Callable[..., Any | Constant | None]]:
    '''gridsize(ndim)

    Return the absolute size (or shape) in threads of the entire grid of
    blocks. *ndim* should correspond to the number of dimensions declared when
    instantiating the kernel. If *ndim* is 1, a single integer is returned.
    If *ndim* is 2 or 3, a tuple of the given number of integers is returned.

    Computation of the first integer is as follows::

        cuda.blockDim.x * cuda.gridDim.x

    and is similar for the other two indices, but using the ``y`` and ``z``
    attributes.
    '''
    ...

@overload_attribute(types.Module(cuda), 'warpsize', target='cuda')
def cuda_warpsize(mod): # -> Callable[..., Any]:
    '''
    The size of a warp. All architectures implemented to date have a warp size
    of 32.
    '''
    ...

@intrinsic
def syncthreads(typingctx): # -> tuple[Signature, Callable[..., Any]]:
    '''
    Synchronize all threads in the same thread block.  This function implements
    the same pattern as barriers in traditional multi-threaded programming: this
    function waits until all threads in the block call it, at which point it
    returns control to all its callers.
    '''
    ...

@intrinsic
def syncthreads_count(typingctx, predicate): # -> tuple[Signature, Callable[..., Any]] | None:
    '''
    syncthreads_count(predicate)

    An extension to numba.cuda.syncthreads where the return value is a count
    of the threads where predicate is true.
    '''
    ...

@intrinsic
def syncthreads_and(typingctx, predicate): # -> tuple[Signature, Callable[..., Any]] | None:
    '''
    syncthreads_and(predicate)

    An extension to numba.cuda.syncthreads where 1 is returned if predicate is
    true for all threads or 0 otherwise.
    '''
    ...

@intrinsic
def syncthreads_or(typingctx, predicate): # -> tuple[Signature, Callable[..., Any]] | None:
    '''
    syncthreads_or(predicate)

    An extension to numba.cuda.syncthreads where 1 is returned if predicate is
    true for any thread or 0 otherwise.
    '''
    ...

