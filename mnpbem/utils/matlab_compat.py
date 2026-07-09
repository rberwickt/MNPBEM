"""MATLAB-compatible floating-point primitives.

Problem: Python's `np.linspace`, `np.arctan2`, `np.exp`, `np.log`, ... output
differ from MATLAB's by 1 ULP due to different FP accumulation order /
algorithm. `np.cos/sin` itself already matches MATLAB on Linux with numpy
linked against MKL, but 1 ULP drift in theta or linspace propagates through
`p @ rot` and `np.unique` causing mesh topology divergence.

This module provides drop-in replacements that match MATLAB bit-for-bit by
calling into MATLAB's own libmwmathutil.so (fdlibm-derived). Falls back to
numpy equivalents when MATLAB is not installed.

Functions:
    mlinspace(a, b, n):        MATLAB's linspace formula
    matan2(y, x):              MATLAB's atan2 via libmwmathutil
    mcos(x), msin(x):          cos/sin
    mtan(x):                   tan
    mexp(x), mlog(x):          exp, natural log
    mlog10(x), mlog2(x):       log10, log2
    mlog1p(x), mexpm1(x):      log1p, expm1
    msqrt(x):                  sqrt
    msinh(x), mcosh(x), mtanh(x):   hyperbolic functions
    masin(x), macos(x), matan(x):   inverse trig
    mpow(b, e):                power (a^b)
    mhypot(a, b):              hypot(a, b) = sqrt(a^2 + b^2)
    mabs(x):                   abs
    msign(x):                  sign
    mround(x):                 round (half-away-from-zero, MATLAB default)
    mfloor(x), mceil(x), mfix(x):   floor, ceil, fix (truncate toward 0)
    m_exp_c(z):                complex exp via Euler (bit-identical parts)
    m_sqrt_c(z):               complex sqrt via mhypot / msqrt
    m_unique(arr, ...):        MATLAB unique (stable sort, first-occurrence)
"""
import ctypes
import os
from ctypes import c_int, c_double, c_size_t, c_ulong, POINTER
import numpy as np


# --- MATLAB libmwmathutil loader ---
# MATLAB R2025b ships its own transcendentals in libmwmathutil.so
# (fdlibm-derived). These differ from np.<func> / Intel MKL at 1 ULP on certain
# inputs, and 1-ULP drift propagates through p @ rot / unique and breaks mesh
# topology. We call both the scalar forms (muDoubleScalar*) and the vector
# template forms (mu::*<double,...>) via ctypes.
_MATLAB_LIB_PATH = '/usr/local/MATLAB/R2025b/bin/glnxa64/libmwmathutil.so'
_mathutil = None

# Registry populated by _init_matlab_lib(). Maps function name -> (scalar_fn,
# vec_fn, n_args). scalar_fn takes n_args doubles and returns a double; vec_fn
# is the raw ctypes function with signature specific to n_args.
_REG = {}

# Vector template specs: name -> (mangled_symbol, n_args, return_type).
# n_args = 1 means 1-arg (out, in, stride_out, stride_in, count).
# n_args = 2 means 2-arg (out, in0, in1, s_out, s0, s1, count).
# return_type is None (void) or c_ulong (error-count variant).
_VEC_SPECS = {
    'cos':   ('_ZN2mu3CosIdEEvPT_S2_mmm',     1, None),
    'sin':   ('_ZN2mu3SinIdEEvPT_S2_mmm',     1, None),
    'tan':   ('_ZN2mu3TanIdEEvPT_S2_mmm',     1, None),
    'exp':   ('_ZN2mu3ExpIdEEvPT_S2_mmm',     1, None),
    'log':   ('_ZN2mu3LogIdEEmPT_S2_mmm',     1, c_ulong),
    'log10': ('_ZN2mu5Log10IdEEmPT_S2_mmm',   1, c_ulong),
    'log2':  ('_ZN2mu4Log2IdEEmPT_S2_mmm',    1, c_ulong),
    'log1p': ('_ZN2mu5Log1pIdEEmPT_S2_mmm',   1, c_ulong),
    'expm1': ('_ZN2mu5Expm1IdEEvPT_S2_mmm',   1, None),
    'sqrt':  ('_ZN2mu4SqrtIdEEmPT_S2_mmm',    1, c_ulong),
    'sinh':  ('_ZN2mu4SinhIdEEvPT_S2_mmm',    1, None),
    'cosh':  ('_ZN2mu4CoshIdEEvPT_S2_mmm',    1, None),
    'tanh':  ('_ZN2mu4TanhIdEEvPT_S2_mmm',    1, None),
    'asin':  ('_ZN2mu4AsinIdEEmPT_S2_mmm',    1, c_ulong),
    'acos':  ('_ZN2mu4AcosIdEEmPT_S2_mmm',    1, c_ulong),
    'atan':  ('_ZN2mu4AtanIdEEvPT_S2_mmm',    1, None),
    'abs':   ('_ZN2mu3AbsIdEEvPN10mfl_scalar8realTypeIT_E4typeEPS3_mmm', 1, None),
    'sign':  ('_ZN2mu4SignIdEEvPT_S2_mmm',    1, None),
    'round': ('_ZN2mu5RoundIdEEvPT_S2_mmm',   1, None),
    'floor': ('_ZN2mu5FloorIdEEvPT_S2_mmm',   1, None),
    'ceil':  ('_ZN2mu4CeilIdEEvPT_S2_mmm',    1, None),
    'fix':   ('_ZN2mu3FixIdEEvPT_S2_mmm',     1, None),
    'atan2': ('_ZN2mu5Atan2IdddEEvPT_PT0_PT1_mmmm', 2, None),
    'power': ('_ZN2mu5PowerIdddEEvPT_PT0_PT1_mmmm', 2, None),
    'hypot': ('_ZN2mu5HypotIdddEEvPT_PT0_PT1_mmmm', 2, None),
}

# Scalar fast-path symbol names (MATLAB's muDoubleScalar* family).
_SCALAR_SPECS = {
    'cos':   ('muDoubleScalarCos',   1),
    'sin':   ('muDoubleScalarSin',   1),
    'tan':   ('muDoubleScalarTan',   1),
    'exp':   ('muDoubleScalarExp',   1),
    'log':   ('muDoubleScalarLog',   1),
    'log10': ('muDoubleScalarLog10', 1),
    'sqrt':  ('muDoubleScalarSqrt',  1),
    'sinh':  ('muDoubleScalarSinh',  1),
    'cosh':  ('muDoubleScalarCosh',  1),
    'tanh':  ('muDoubleScalarTanh',  1),
    'asin':  ('muDoubleScalarAsin',  1),
    'acos':  ('muDoubleScalarAcos',  1),
    'atan':  ('muDoubleScalarAtan',  1),
    'abs':   ('muDoubleScalarAbs',   1),
    'sign':  ('muDoubleScalarSign',  1),
    'round': ('muDoubleScalarRound', 1),
    'floor': ('muDoubleScalarFloor', 1),
    'ceil':  ('muDoubleScalarCeil',  1),
    'fix':   ('muDoubleScalarFix',   1),
    'atan2': ('muDoubleScalarAtan2', 2),
    'power': ('muDoubleScalarPower', 2),
    'hypot': ('muDoubleScalarHypot', 2),
}


def _init_matlab_lib():
    """Load libmwmathutil.so and bind every function in _VEC_SPECS /
    _SCALAR_SPECS. Returns True on success, False if MATLAB is not installed
    (in which case all wrappers fall back to numpy).
    """
    global _mathutil
    if _mathutil is not None:
        return True
    if not os.path.exists(_MATLAB_LIB_PATH):
        return False
    try:
        _mathutil = ctypes.CDLL(_MATLAB_LIB_PATH)
    except OSError:
        _mathutil = None
        return False

    for name, (symbol, n_args, rtype) in _VEC_SPECS.items():
        try:
            vec = getattr(_mathutil, symbol)
        except AttributeError:
            _mathutil = None
            return False
        if n_args == 1:
            vec.argtypes = [POINTER(c_double), POINTER(c_double),
                            c_size_t, c_size_t, c_size_t]
        else:
            vec.argtypes = [POINTER(c_double), POINTER(c_double),
                            POINTER(c_double),
                            c_size_t, c_size_t, c_size_t, c_size_t]
        vec.restype = rtype
        _REG[name + '_vec'] = vec
        _REG[name + '_nargs'] = n_args

    for name, (symbol, n_args) in _SCALAR_SPECS.items():
        try:
            scalar = getattr(_mathutil, symbol)
        except AttributeError:
            continue
        scalar.argtypes = [c_double] * n_args
        scalar.restype = c_double
        _REG[name + '_scalar'] = scalar

    return True


_MATLAB_ATAN2_AVAILABLE = _init_matlab_lib()


# --- backwards-compatible module-level aliases (used by tests) ---
if _MATLAB_ATAN2_AVAILABLE:
    _matan2_vec    = _REG['atan2_vec']
    _matan2_scalar = _REG['atan2_scalar']
    _mcos_vec      = _REG['cos_vec']
    _mcos_scalar   = _REG['cos_scalar']
    _msin_vec      = _REG['sin_vec']
    _msin_scalar   = _REG['sin_scalar']
else:
    _matan2_vec = None
    _matan2_scalar = None
    _mcos_vec = None
    _mcos_scalar = None
    _msin_vec = None
    _msin_scalar = None


def mlinspace(a, b, n):
    """MATLAB-compatible linspace.

    MATLAB formula: y(i) = a + (i-1) * (b-a) / (n-1), with y(end)=b enforced.
    Differs from np.linspace at up to 1 ULP because numpy uses
    y = start + arange(num) * step where step = (stop-start)/div.
    """
    a = float(a)
    b = float(b)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([b], dtype=np.float64)
    idx = np.arange(n, dtype=np.float64)
    y = a + idx * (b - a) / (n - 1)
    y[-1] = b
    return y


# --- Generic 1-arg / 2-arg dispatchers (shared by every wrapper below) ---

def _call_unary(name, x, np_fn):
    """Scalar-or-array dispatcher for unary MATLAB transcendentals."""
    if not globals().get('_MATLAB_ATAN2_AVAILABLE', False):
        return np_fn(x)
    scalar = _REG.get(name + '_scalar')
    is_py_scalar = np.isscalar(x)
    if is_py_scalar and scalar is not None:
        return scalar(float(x))
    x_raw = np.asarray(x, dtype=np.float64)
    scalar_shape = is_py_scalar or x_raw.ndim == 0
    if scalar_shape and scalar is not None:
        return np.float64(scalar(float(x_raw)))
    # Either no scalar symbol exported (log2/log1p/expm1) or a real array.
    if scalar_shape:
        in_buf = np.array([float(x_raw)], dtype=np.float64)
        out = np.empty(1, dtype=np.float64)
    else:
        x_arr = np.ascontiguousarray(x_raw)
        if x_arr.size == 0:
            return np.empty_like(x_arr)
        in_buf = x_arr
        out = np.empty_like(x_arr)
    vec = _REG[name + '_vec']
    vec(
        out.ctypes.data_as(POINTER(c_double)),
        in_buf.ctypes.data_as(POINTER(c_double)),
        c_size_t(1), c_size_t(1), c_size_t(in_buf.size),
    )
    if scalar_shape:
        if is_py_scalar:
            return float(out[0])
        return np.float64(out[0])
    return out


def _call_binary(name, a, b, np_fn):
    """Scalar-or-array dispatcher for binary MATLAB transcendentals (atan2,
    power, hypot). Accepts broadcastable inputs like np.<fn>."""
    if not globals().get('_MATLAB_ATAN2_AVAILABLE', False):
        return np_fn(a, b)
    scalar = _REG.get(name + '_scalar')
    is_py_scalar = np.isscalar(a) and np.isscalar(b)
    if is_py_scalar and scalar is not None:
        return scalar(float(a), float(b))
    a_raw = np.asarray(a, dtype=np.float64)
    b_raw = np.asarray(b, dtype=np.float64)
    if a_raw.shape != b_raw.shape:
        a_raw, b_raw = np.broadcast_arrays(a_raw, b_raw)
    scalar_shape = (is_py_scalar or (a_raw.ndim == 0 and b_raw.ndim == 0))
    if scalar_shape and scalar is not None:
        return np.float64(scalar(float(a_raw), float(b_raw)))
    if scalar_shape:
        a_buf = np.array([float(a_raw)], dtype=np.float64)
        b_buf = np.array([float(b_raw)], dtype=np.float64)
        out = np.empty(1, dtype=np.float64)
    else:
        a_arr = np.ascontiguousarray(a_raw)
        b_arr = np.ascontiguousarray(b_raw)
        if a_arr.size == 0:
            return np.empty_like(a_arr)
        a_buf = a_arr
        b_buf = b_arr
        out = np.empty_like(a_arr)
    vec = _REG[name + '_vec']
    vec(
        out.ctypes.data_as(POINTER(c_double)),
        a_buf.ctypes.data_as(POINTER(c_double)),
        b_buf.ctypes.data_as(POINTER(c_double)),
        c_size_t(1), c_size_t(1), c_size_t(1), c_size_t(a_buf.size),
    )
    if scalar_shape:
        if is_py_scalar:
            return float(out[0])
        return np.float64(out[0])
    return out


# --- Public wrappers: unary ---

def mcos(x):   return _call_unary('cos',   x, np.cos)
def msin(x):   return _call_unary('sin',   x, np.sin)
def mtan(x):   return _call_unary('tan',   x, np.tan)
def mexp(x):   return _call_unary('exp',   x, np.exp)
def mlog(x):   return _call_unary('log',   x, np.log)
def mlog10(x): return _call_unary('log10', x, np.log10)
def mlog2(x):  return _call_unary('log2',  x, np.log2)
def mlog1p(x): return _call_unary('log1p', x, np.log1p)
def mexpm1(x): return _call_unary('expm1', x, np.expm1)
def msqrt(x):  return _call_unary('sqrt',  x, np.sqrt)
def msinh(x):  return _call_unary('sinh',  x, np.sinh)
def mcosh(x):  return _call_unary('cosh',  x, np.cosh)
def mtanh(x):  return _call_unary('tanh',  x, np.tanh)
def masin(x):  return _call_unary('asin',  x, np.arcsin)
def macos(x):  return _call_unary('acos',  x, np.arccos)
def matan(x):  return _call_unary('atan',  x, np.arctan)
def mabs(x):   return _call_unary('abs',   x, np.abs)
def msign(x):  return _call_unary('sign',  x, np.sign)


def mfloor(x): return _call_unary('floor', x, np.floor)
def mceil(x):  return _call_unary('ceil',  x, np.ceil)
def mfix(x):   return _call_unary('fix',   x, np.trunc)


def mround(x, n=0):
    """MATLAB-compatible round.

    MATLAB's round uses half-away-from-zero tie-breaking, unlike numpy which
    uses banker's rounding. With n given (digits), matches MATLAB's
    `round(x, n)` = round(x * 10^n) / 10^n using MATLAB's round semantics.
    """
    if n == 0:
        if not globals().get('_MATLAB_ATAN2_AVAILABLE', False):
            return np.trunc(np.asarray(x, dtype=np.float64)
                            + np.copysign(0.5, np.asarray(x, dtype=np.float64)))
        return _call_unary('round', x, lambda a: np.trunc(
            np.asarray(a, dtype=np.float64)
            + np.copysign(0.5, np.asarray(a, dtype=np.float64))))
    scale = 10.0 ** n
    x_arr = np.asarray(x, dtype=np.float64)
    scaled = x_arr * scale
    rounded = mround(scaled, 0)
    return rounded / scale


# --- Public wrappers: binary ---

def matan2(y, x):
    """MATLAB-compatible atan2.

    Loads MATLAB's own atan2 from libmwmathutil.so when available; otherwise
    falls back to np.arctan2. Bit-identical with MATLAB `atan2(y, x)` on the
    full domain including signed zeros, inf, and NaN.
    """
    return _call_binary('atan2', y, x, np.arctan2)


def mpow(base, exponent):
    """MATLAB-compatible power (a .^ b)."""
    return _call_binary('power', base, exponent, np.power)


def mhypot(a, b):
    """MATLAB-compatible hypot: sqrt(a^2 + b^2), avoids over/underflow."""
    return _call_binary('hypot', a, b, np.hypot)


# --- Complex wrappers (Euler composition over real MATLAB primitives) ---

def m_exp_c(z):
    """MATLAB-compatible complex exp via Euler formula.

    exp(a + b*j) = exp(a) * (cos(b) + j*sin(b)).  Uses mexp / mcos / msin so
    real and imaginary parts are each bit-identical to MATLAB.  Real input
    falls through to mexp directly.
    """
    z_arr = np.asarray(z)
    if not np.iscomplexobj(z_arr):
        return mexp(z_arr)
    a = z_arr.real
    b = z_arr.imag
    ea = mexp(a)
    return ea * mcos(b) + 1j * ea * msin(b)


def m_sqrt_c(z):
    """MATLAB-compatible complex sqrt via mhypot / msqrt.

    For z = x + y*j, sqrt(z) = sqrt((|z| + x) / 2) + sign(y) * j *
    sqrt((|z| - x) / 2).  Real non-negative input goes through msqrt; real
    negative input returns a pure-imaginary sqrt.  When y == 0 exactly we
    preserve the sign of the imaginary zero via the MATLAB convention
    sign(+0) = +1.
    """
    z_arr = np.asarray(z)
    if not np.iscomplexobj(z_arr):
        x = z_arr
        if np.isscalar(x) or x.ndim == 0:
            xv = float(x)
            if xv >= 0:
                return msqrt(xv)
            return 1j * msqrt(-xv)
        x = np.asarray(x, dtype = np.float64)
        out = np.empty_like(x, dtype = np.complex128)
        pos = x >= 0
        neg = ~pos
        out[pos] = msqrt(x[pos])
        out[neg] = 1j * msqrt(-x[neg])
        return out
    x = z_arr.real
    y = z_arr.imag
    r = mhypot(x, y)
    # (r + x) / 2 and (r - x) / 2 are both >= 0 for any finite z
    re = msqrt((r + x) / 2.0)
    im_mag = msqrt((r - x) / 2.0)
    sgn = np.where(y >= 0, 1.0, -1.0)
    return re + 1j * sgn * im_mag


def m_unique(arr, axis = None, return_index = False, return_inverse = False):
    """MATLAB-compatible unique.

    MATLAB `[C, ia, ic] = unique(A)` uses stable sort and returns the FIRST
    occurrence index for each unique element, whereas `np.unique` returns the
    LAST occurrence when there are duplicates (numpy uses a sort that then
    picks values at sorted positions).  For mesh topology the index choice
    matters because downstream code uses `ia` to reference original rows.

    This implementation:
      - uses numpy stable sort to group equal entries
      - selects the smallest original index within each group (= first
        occurrence) for `return_index`
      - reproduces `return_inverse` using the same grouping

    Supports 1-D input (axis=None) and 2-D row-unique (axis=0), which is all
    MATLAB's unique(A, 'rows') needs.
    """
    a = np.asarray(arr)
    if axis is None:
        flat = a.ravel()
        order = np.argsort(flat, kind = 'stable')
        sorted_vals = flat[order]
        if sorted_vals.size == 0:
            uniq = sorted_vals.copy()
            ia = np.array([], dtype = np.intp)
            ic = np.array([], dtype = np.intp)
        else:
            diff_mask = np.concatenate(
                ([True], sorted_vals[1:] != sorted_vals[:-1]))
            group_ids = np.cumsum(diff_mask) - 1
            n_groups = int(group_ids[-1]) + 1
            uniq = sorted_vals[diff_mask]
            # first-occurrence: within each group pick min of original index
            ia = np.full(n_groups, np.iinfo(np.intp).max, dtype = np.intp)
            for gid, orig_idx in zip(group_ids, order):
                if orig_idx < ia[gid]:
                    ia[gid] = orig_idx
            ic = np.empty(flat.size, dtype = np.intp)
            ic[order] = group_ids
        result = [uniq]
        if return_index:
            result.append(ia)
        if return_inverse:
            result.append(ic)
        return tuple(result) if len(result) > 1 else uniq
    if axis == 0:
        if a.ndim != 2:
            raise ValueError("m_unique(axis=0) expects 2-D input")
        # Lexicographic stable sort over rows
        keys = tuple(a[:, c] for c in range(a.shape[1] - 1, -1, -1))
        order = np.lexsort(keys)
        sorted_rows = a[order]
        if sorted_rows.shape[0] == 0:
            uniq = sorted_rows.copy()
            ia = np.array([], dtype = np.intp)
            ic = np.array([], dtype = np.intp)
        else:
            diff_mask = np.concatenate(
                ([True],
                 np.any(sorted_rows[1:] != sorted_rows[:-1], axis = 1)))
            group_ids = np.cumsum(diff_mask) - 1
            n_groups = int(group_ids[-1]) + 1
            uniq = sorted_rows[diff_mask]
            ia = np.full(n_groups, np.iinfo(np.intp).max, dtype = np.intp)
            for gid, orig_idx in zip(group_ids, order):
                if orig_idx < ia[gid]:
                    ia[gid] = orig_idx
            ic = np.empty(a.shape[0], dtype = np.intp)
            ic[order] = group_ids
        result = [uniq]
        if return_index:
            result.append(ia)
        if return_inverse:
            result.append(ic)
        return tuple(result) if len(result) > 1 else uniq
    raise ValueError(f"m_unique: unsupported axis={axis}")


# --- Legacy-shaped internal for test_matan2.py ---
def _matan2_impl(y_arr, x_arr):
    """Vectorized call used by the legacy atan2 test. y_arr, x_arr must be
    contiguous float64 arrays of the same shape."""
    out = np.empty_like(y_arr)
    n = y_arr.size
    if n == 0:
        return out
    _matan2_vec(
        out.ctypes.data_as(POINTER(c_double)),
        y_arr.ctypes.data_as(POINTER(c_double)),
        x_arr.ctypes.data_as(POINTER(c_double)),
        c_size_t(1), c_size_t(1), c_size_t(1), c_size_t(n),
    )
    return out
