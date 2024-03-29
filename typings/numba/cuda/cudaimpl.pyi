"""
This type stub file was generated by pyright.
"""

import operator
from numba.core.imputils import lower_cast
from numba.core import types
from numba import cuda
from numba.cuda import stubs
from numba.cuda.types import CUDADispatcher, dim3

registry = ...
lower = ...
lower_attr = ...
lower_constant = ...
def initialize_dim3(builder, prefix): # -> Constant:
    ...

@lower_attr(types.Module(cuda), 'threadIdx')
def cuda_threadIdx(context, builder, sig, args): # -> Constant:
    ...

@lower_attr(types.Module(cuda), 'blockDim')
def cuda_blockDim(context, builder, sig, args): # -> Constant:
    ...

@lower_attr(types.Module(cuda), 'blockIdx')
def cuda_blockIdx(context, builder, sig, args): # -> Constant:
    ...

@lower_attr(types.Module(cuda), 'gridDim')
def cuda_gridDim(context, builder, sig, args): # -> Constant:
    ...

@lower_attr(types.Module(cuda), 'laneid')
def cuda_laneid(context, builder, sig, args):
    ...

@lower_attr(dim3, 'x')
def dim3_x(context, builder, sig, args):
    ...

@lower_attr(dim3, 'y')
def dim3_y(context, builder, sig, args):
    ...

@lower_attr(dim3, 'z')
def dim3_z(context, builder, sig, args):
    ...

@lower(cuda.const.array_like, types.Array)
def cuda_const_array_like(context, builder, sig, args):
    ...

_unique_smem_id = ...
@lower(cuda.shared.array, types.IntegerLiteral, types.Any)
def cuda_shared_array_integer(context, builder, sig, args):
    ...

@lower(cuda.shared.array, types.Tuple, types.Any)
@lower(cuda.shared.array, types.UniTuple, types.Any)
def cuda_shared_array_tuple(context, builder, sig, args):
    ...

@lower(cuda.local.array, types.IntegerLiteral, types.Any)
def cuda_local_array_integer(context, builder, sig, args):
    ...

@lower(cuda.local.array, types.Tuple, types.Any)
@lower(cuda.local.array, types.UniTuple, types.Any)
def ptx_lmem_alloc_array(context, builder, sig, args):
    ...

@lower(stubs.threadfence_block)
def ptx_threadfence_block(context, builder, sig, args):
    ...

@lower(stubs.threadfence_system)
def ptx_threadfence_system(context, builder, sig, args):
    ...

@lower(stubs.threadfence)
def ptx_threadfence_device(context, builder, sig, args):
    ...

@lower(stubs.syncwarp)
def ptx_syncwarp(context, builder, sig, args):
    ...

@lower(stubs.syncwarp, types.i4)
def ptx_syncwarp_mask(context, builder, sig, args):
    ...

@lower(stubs.shfl_sync_intrinsic, types.i4, types.i4, types.i4, types.i4, types.i4)
@lower(stubs.shfl_sync_intrinsic, types.i4, types.i4, types.i8, types.i4, types.i4)
@lower(stubs.shfl_sync_intrinsic, types.i4, types.i4, types.f4, types.i4, types.i4)
@lower(stubs.shfl_sync_intrinsic, types.i4, types.i4, types.f8, types.i4, types.i4)
def ptx_shfl_sync_i32(context, builder, sig, args): # -> Constant:
    """
    The NVVM intrinsic for shfl only supports i32, but the cuda intrinsic
    function supports both 32 and 64 bit ints and floats, so for feature parity,
    i64, f32, and f64 are implemented. Floats by way of bitcasting the float to
    an int, then shuffling, then bitcasting back. And 64-bit values by packing
    them into 2 32bit values, shuffling thoose, and then packing back together.
    """
    ...

@lower(stubs.vote_sync_intrinsic, types.i4, types.i4, types.boolean)
def ptx_vote_sync(context, builder, sig, args):
    ...

@lower(stubs.match_any_sync, types.i4, types.i4)
@lower(stubs.match_any_sync, types.i4, types.i8)
@lower(stubs.match_any_sync, types.i4, types.f4)
@lower(stubs.match_any_sync, types.i4, types.f8)
def ptx_match_any_sync(context, builder, sig, args):
    ...

@lower(stubs.match_all_sync, types.i4, types.i4)
@lower(stubs.match_all_sync, types.i4, types.i8)
@lower(stubs.match_all_sync, types.i4, types.f4)
@lower(stubs.match_all_sync, types.i4, types.f8)
def ptx_match_all_sync(context, builder, sig, args):
    ...

@lower(stubs.activemask)
def ptx_activemask(context, builder, sig, args):
    ...

@lower(stubs.lanemask_lt)
def ptx_lanemask_lt(context, builder, sig, args):
    ...

@lower(stubs.popc, types.Any)
def ptx_popc(context, builder, sig, args):
    ...

@lower(stubs.fma, types.Any, types.Any, types.Any)
def ptx_fma(context, builder, sig, args):
    ...

def float16_float_ty_constraint(bitwidth):
    ...

@lower_cast(types.float16, types.Float)
def float16_to_float_cast(context, builder, fromty, toty, val):
    ...

@lower_cast(types.Float, types.float16)
def float_to_float16_cast(context, builder, fromty, toty, val):
    ...

def float16_int_constraint(bitwidth): # -> str:
    ...

@lower_cast(types.float16, types.Integer)
def float16_to_integer_cast(context, builder, fromty, toty, val):
    ...

@lower_cast(types.Integer, types.float16)
@lower_cast(types.IntegerLiteral, types.float16)
def integer_to_float16_cast(context, builder, fromty, toty, val):
    ...

def lower_fp16_binary(fn, op): # -> None:
    ...

@lower(stubs.fp16.hneg, types.float16)
def ptx_fp16_hneg(context, builder, sig, args):
    ...

@lower(operator.neg, types.float16)
def operator_hneg(context, builder, sig, args):
    ...

@lower(stubs.fp16.habs, types.float16)
def ptx_fp16_habs(context, builder, sig, args):
    ...

@lower(abs, types.float16)
def operator_habs(context, builder, sig, args):
    ...

@lower(stubs.fp16.hfma, types.float16, types.float16, types.float16)
def ptx_hfma(context, builder, sig, args):
    ...

@lower(operator.truediv, types.float16, types.float16)
@lower(operator.itruediv, types.float16, types.float16)
def fp16_div_impl(context, builder, sig, args):
    ...

_fp16_cmp = ...
def lower_fp16_minmax(fn, fname, op): # -> None:
    ...

cbrt_funcs = ...
@lower(stubs.cbrt, types.float32)
@lower(stubs.cbrt, types.float64)
def ptx_cbrt(context, builder, sig, args):
    ...

@lower(stubs.brev, types.u4)
def ptx_brev_u4(context, builder, sig, args):
    ...

@lower(stubs.brev, types.u8)
def ptx_brev_u8(context, builder, sig, args):
    ...

@lower(stubs.clz, types.Any)
def ptx_clz(context, builder, sig, args):
    ...

@lower(stubs.ffs, types.i4)
@lower(stubs.ffs, types.u4)
def ptx_ffs_32(context, builder, sig, args):
    ...

@lower(stubs.ffs, types.i8)
@lower(stubs.ffs, types.u8)
def ptx_ffs_64(context, builder, sig, args):
    ...

@lower(stubs.selp, types.Any, types.Any, types.Any)
def ptx_selp(context, builder, sig, args):
    ...

@lower(max, types.f4, types.f4)
def ptx_max_f4(context, builder, sig, args):
    ...

@lower(max, types.f8, types.f4)
@lower(max, types.f4, types.f8)
@lower(max, types.f8, types.f8)
def ptx_max_f8(context, builder, sig, args):
    ...

@lower(min, types.f4, types.f4)
def ptx_min_f4(context, builder, sig, args):
    ...

@lower(min, types.f8, types.f4)
@lower(min, types.f4, types.f8)
@lower(min, types.f8, types.f8)
def ptx_min_f8(context, builder, sig, args):
    ...

@lower(round, types.f4)
@lower(round, types.f8)
def ptx_round(context, builder, sig, args):
    ...

@lower(round, types.f4, types.Integer)
@lower(round, types.f8, types.Integer)
def round_to_impl(context, builder, sig, args):
    ...

def gen_deg_rad(const): # -> Callable[..., Any]:
    ...

_deg2rad = ...
_rad2deg = ...
@lower(stubs.atomic.add, types.Array, types.intp, types.Any)
@lower(stubs.atomic.add, types.Array, types.UniTuple, types.Any)
@lower(stubs.atomic.add, types.Array, types.Tuple, types.Any)
@_atomic_dispatcher
def ptx_atomic_add_tuple(context, builder, dtype, ptr, val):
    ...

@lower(stubs.atomic.sub, types.Array, types.intp, types.Any)
@lower(stubs.atomic.sub, types.Array, types.UniTuple, types.Any)
@lower(stubs.atomic.sub, types.Array, types.Tuple, types.Any)
@_atomic_dispatcher
def ptx_atomic_sub(context, builder, dtype, ptr, val):
    ...

@lower(stubs.atomic.inc, types.Array, types.intp, types.Any)
@lower(stubs.atomic.inc, types.Array, types.UniTuple, types.Any)
@lower(stubs.atomic.inc, types.Array, types.Tuple, types.Any)
@_atomic_dispatcher
def ptx_atomic_inc(context, builder, dtype, ptr, val):
    ...

@lower(stubs.atomic.dec, types.Array, types.intp, types.Any)
@lower(stubs.atomic.dec, types.Array, types.UniTuple, types.Any)
@lower(stubs.atomic.dec, types.Array, types.Tuple, types.Any)
@_atomic_dispatcher
def ptx_atomic_dec(context, builder, dtype, ptr, val):
    ...

def ptx_atomic_bitwise(stub, op): # -> None:
    ...

@lower(stubs.atomic.exch, types.Array, types.intp, types.Any)
@lower(stubs.atomic.exch, types.Array, types.UniTuple, types.Any)
@lower(stubs.atomic.exch, types.Array, types.Tuple, types.Any)
@_atomic_dispatcher
def ptx_atomic_exch(context, builder, dtype, ptr, val):
    ...

@lower(stubs.atomic.max, types.Array, types.intp, types.Any)
@lower(stubs.atomic.max, types.Array, types.Tuple, types.Any)
@lower(stubs.atomic.max, types.Array, types.UniTuple, types.Any)
@_atomic_dispatcher
def ptx_atomic_max(context, builder, dtype, ptr, val):
    ...

@lower(stubs.atomic.min, types.Array, types.intp, types.Any)
@lower(stubs.atomic.min, types.Array, types.Tuple, types.Any)
@lower(stubs.atomic.min, types.Array, types.UniTuple, types.Any)
@_atomic_dispatcher
def ptx_atomic_min(context, builder, dtype, ptr, val):
    ...

@lower(stubs.atomic.nanmax, types.Array, types.intp, types.Any)
@lower(stubs.atomic.nanmax, types.Array, types.Tuple, types.Any)
@lower(stubs.atomic.nanmax, types.Array, types.UniTuple, types.Any)
@_atomic_dispatcher
def ptx_atomic_nanmax(context, builder, dtype, ptr, val):
    ...

@lower(stubs.atomic.nanmin, types.Array, types.intp, types.Any)
@lower(stubs.atomic.nanmin, types.Array, types.Tuple, types.Any)
@lower(stubs.atomic.nanmin, types.Array, types.UniTuple, types.Any)
@_atomic_dispatcher
def ptx_atomic_nanmin(context, builder, dtype, ptr, val):
    ...

@lower(stubs.atomic.compare_and_swap, types.Array, types.Any, types.Any)
def ptx_atomic_compare_and_swap(context, builder, sig, args):
    ...

@lower(stubs.atomic.cas, types.Array, types.intp, types.Any, types.Any)
@lower(stubs.atomic.cas, types.Array, types.Tuple, types.Any, types.Any)
@lower(stubs.atomic.cas, types.Array, types.UniTuple, types.Any, types.Any)
def ptx_atomic_cas(context, builder, sig, args):
    ...

@lower(stubs.nanosleep, types.uint32)
def ptx_nanosleep(context, builder, sig, args): # -> None:
    ...

@lower_constant(CUDADispatcher)
def cuda_dispatcher_const(context, builder, ty, pyval):
    ...

