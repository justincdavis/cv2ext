"""
This type stub file was generated by pyright.
"""

from numba.core.extending import overload

"""Codegen for functions used as kernels in NumPy functions

Typically, the kernels of several ufuncs that can't map directly to
Python builtins
"""
_NPY_LOG2E = ...
_NPY_LOG10E = ...
_NPY_LOGE2 = ...
def np_int_sdiv_impl(context, builder, sig, args):
    ...

def np_int_srem_impl(context, builder, sig, args):
    ...

def np_int_sdivrem_impl(context, builder, sig, args):
    ...

def np_int_udiv_impl(context, builder, sig, args):
    ...

def np_int_urem_impl(context, builder, sig, args):
    ...

def np_int_udivrem_impl(context, builder, sig, args):
    ...

np_int_fmod_impl = ...
def np_real_div_impl(context, builder, sig, args):
    ...

def np_real_mod_impl(context, builder, sig, args):
    ...

def np_real_fmod_impl(context, builder, sig, args):
    ...

def np_complex_div_impl(context, builder, sig, args):
    ...

def np_real_logaddexp_impl(context, builder, sig, args):
    ...

def npy_log2_1p(x): # -> None:
    ...

@overload(npy_log2_1p, target='generic')
def ol_npy_log2_1p(x): # -> Callable[..., Any]:
    ...

def np_real_logaddexp2_impl(context, builder, sig, args):
    ...

def np_int_truediv_impl(context, builder, sig, args):
    ...

def np_real_floor_div_impl(context, builder, sig, args):
    ...

def np_real_divmod_impl(context, builder, sig, args):
    ...

def np_complex_floor_div_impl(context, builder, sig, args):
    ...

def np_complex_power_impl(context, builder, sig, args):
    ...

def real_float_power_impl(context, builder, sig, args):
    ...

def np_complex_float_power_impl(context, builder, sig, args):
    ...

def np_gcd_impl(context, builder, sig, args):
    ...

def np_lcm_impl(context, builder, sig, args):
    ...

def np_complex_sign_impl(context, builder, sig, args):
    ...

def np_real_rint_impl(context, builder, sig, args):
    ...

def np_complex_rint_impl(context, builder, sig, args):
    ...

def np_real_exp_impl(context, builder, sig, args):
    ...

def np_complex_exp_impl(context, builder, sig, args): # -> complex:
    ...

def np_real_exp2_impl(context, builder, sig, args):
    ...

def np_complex_exp2_impl(context, builder, sig, args): # -> complex:
    ...

def np_real_log_impl(context, builder, sig, args):
    ...

def np_complex_log_impl(context, builder, sig, args): # -> complex:
    ...

def np_real_log2_impl(context, builder, sig, args):
    ...

def np_complex_log2_impl(context, builder, sig, args):
    ...

def np_real_log10_impl(context, builder, sig, args):
    ...

def np_complex_log10_impl(context, builder, sig, args):
    ...

def np_real_expm1_impl(context, builder, sig, args):
    ...

def np_complex_expm1_impl(context, builder, sig, args):
    ...

def np_real_log1p_impl(context, builder, sig, args):
    ...

def np_complex_log1p_impl(context, builder, sig, args):
    ...

def np_real_sqrt_impl(context, builder, sig, args):
    ...

def np_complex_sqrt_impl(context, builder, sig, args):
    ...

def np_int_square_impl(context, builder, sig, args):
    ...

def np_real_square_impl(context, builder, sig, args):
    ...

def np_complex_square_impl(context, builder, sig, args):
    ...

def np_real_cbrt_impl(context, builder, sig, args):
    ...

def np_int_reciprocal_impl(context, builder, sig, args):
    ...

def np_real_reciprocal_impl(context, builder, sig, args):
    ...

def np_complex_reciprocal_impl(context, builder, sig, args):
    ...

def np_real_sin_impl(context, builder, sig, args):
    ...

def np_complex_sin_impl(context, builder, sig, args):
    ...

def np_real_cos_impl(context, builder, sig, args):
    ...

def np_complex_cos_impl(context, builder, sig, args):
    ...

def np_real_tan_impl(context, builder, sig, args):
    ...

def np_real_asin_impl(context, builder, sig, args):
    ...

def np_real_acos_impl(context, builder, sig, args):
    ...

def np_real_atan_impl(context, builder, sig, args):
    ...

def np_real_atan2_impl(context, builder, sig, args):
    ...

def np_real_hypot_impl(context, builder, sig, args):
    ...

def np_real_sinh_impl(context, builder, sig, args):
    ...

def np_complex_sinh_impl(context, builder, sig, args):
    ...

def np_real_cosh_impl(context, builder, sig, args):
    ...

def np_complex_cosh_impl(context, builder, sig, args):
    ...

def np_real_tanh_impl(context, builder, sig, args):
    ...

def np_complex_tanh_impl(context, builder, sig, args):
    ...

def np_real_asinh_impl(context, builder, sig, args):
    ...

def np_real_acosh_impl(context, builder, sig, args):
    ...

def np_complex_acosh_impl(context, builder, sig, args): # -> complex:
    ...

def np_real_atanh_impl(context, builder, sig, args):
    ...

def np_real_floor_impl(context, builder, sig, args):
    ...

def np_real_ceil_impl(context, builder, sig, args):
    ...

def np_real_trunc_impl(context, builder, sig, args):
    ...

def np_real_fabs_impl(context, builder, sig, args):
    ...

def np_complex_ge_impl(context, builder, sig, args):
    ...

def np_complex_le_impl(context, builder, sig, args):
    ...

def np_complex_gt_impl(context, builder, sig, args):
    ...

def np_complex_lt_impl(context, builder, sig, args):
    ...

def np_complex_eq_impl(context, builder, sig, args):
    ...

def np_complex_ne_impl(context, builder, sig, args):
    ...

def np_logical_and_impl(context, builder, sig, args):
    ...

def np_complex_logical_and_impl(context, builder, sig, args):
    ...

def np_logical_or_impl(context, builder, sig, args):
    ...

def np_complex_logical_or_impl(context, builder, sig, args):
    ...

def np_logical_xor_impl(context, builder, sig, args):
    ...

def np_complex_logical_xor_impl(context, builder, sig, args):
    ...

def np_logical_not_impl(context, builder, sig, args):
    ...

def np_complex_logical_not_impl(context, builder, sig, args):
    ...

def np_int_smax_impl(context, builder, sig, args):
    ...

def np_int_umax_impl(context, builder, sig, args):
    ...

def np_real_maximum_impl(context, builder, sig, args):
    ...

def np_real_fmax_impl(context, builder, sig, args):
    ...

def np_complex_maximum_impl(context, builder, sig, args):
    ...

def np_complex_fmax_impl(context, builder, sig, args):
    ...

def np_int_smin_impl(context, builder, sig, args):
    ...

def np_int_umin_impl(context, builder, sig, args):
    ...

def np_real_minimum_impl(context, builder, sig, args):
    ...

def np_real_fmin_impl(context, builder, sig, args):
    ...

def np_complex_minimum_impl(context, builder, sig, args):
    ...

def np_complex_fmin_impl(context, builder, sig, args):
    ...

def np_int_isnan_impl(context, builder, sig, args): # -> Constant:
    ...

def np_real_isnan_impl(context, builder, sig, args):
    ...

def np_complex_isnan_impl(context, builder, sig, args):
    ...

def np_int_isfinite_impl(context, builder, sig, args): # -> Constant:
    ...

def np_datetime_isfinite_impl(context, builder, sig, args):
    ...

def np_datetime_isnat_impl(context, builder, sig, args):
    ...

def np_real_isfinite_impl(context, builder, sig, args):
    ...

def np_complex_isfinite_impl(context, builder, sig, args):
    ...

def np_int_isinf_impl(context, builder, sig, args): # -> Constant:
    ...

def np_real_isinf_impl(context, builder, sig, args):
    ...

def np_complex_isinf_impl(context, builder, sig, args):
    ...

def np_real_signbit_impl(context, builder, sig, args):
    ...

def np_real_copysign_impl(context, builder, sig, args):
    ...

def np_real_nextafter_impl(context, builder, sig, args):
    ...

def np_real_spacing_impl(context, builder, sig, args):
    ...

def np_real_ldexp_impl(context, builder, sig, args):
    ...

