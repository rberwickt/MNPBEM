"""
Numba-accelerated kernels for retarded Green function assembly.

Used by GreenRetRefined.eval for the dense G/F/Gp pre-phase fill.
Refinement overlays are applied by the Python caller after these kernels
produce the dense pre-phase matrices.

Activation:
  - default: enabled when numba is importable
  - disable by setting MNPBEM_NUMBA=0

GPU path:
  - opt-in via MNPBEM_GPU=1 (default OFF)
  - requires cupy import; falls back to numba/numpy when cupy unavailable
  - implemented as cupy element-wise expressions (IEEE 754 strict, no
    fastmath); produces bit-identical results vs the CPU path within
    cupy's ufunc evaluation order.
"""

import os
import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import cupy as _cp  # type: ignore
    CUPY_AVAILABLE = True
except Exception:
    _cp = None  # type: ignore
    CUPY_AVAILABLE = False


_EPS = 2.220446049250313e-16


def numba_enabled():
    """Return True iff numba kernels should be used."""
    if not NUMBA_AVAILABLE:
        return False
    return os.environ.get('MNPBEM_NUMBA', '1') != '0'


def gpu_enabled():
    """Return True iff cupy GPU path should be used."""
    if not CUPY_AVAILABLE:
        return False
    return os.environ.get('MNPBEM_GPU', '0') == '1'


def gpu_native_enabled():
    """Return True iff GPU-native cupy passthrough is enabled.

    Default ON when MNPBEM_GPU=1 (Phase 3 native path verified bit-identical).
    Set MNPBEM_GPU_NATIVE=0 to opt out (escape hatch).
    """
    if not CUPY_AVAILABLE:
        return False
    if os.environ.get('MNPBEM_GPU', '0') != '1':
        return False
    return os.environ.get('MNPBEM_GPU_NATIVE', '1') != '0'


if NUMBA_AVAILABLE:

    @njit(parallel = True, fastmath = False, cache = True)
    def _green_ret_dGF(pos1, pos2, nvec1, area2, same):
        """
        Build distance d, pre-phase G and F matrices.

        G_pre[i, j] = area2[j] / d
        F_pre[i, j] = (n_dot_r) * (1j * k - 1/d) / d^2 * area2[j]   [k applied later]

        For numerical efficiency we return only k-independent factors:
          d[i, j]
          inv_d[i, j]            = 1 / d
          n_dot_r[i, j]          = nvec1[i] · (pos1[i] - pos2[j])
        Caller assembles G_pre, F_pre (cheap) then multiplies by exp(1j*k*d).
        Self-block (same and i == j) entries get d = eps so refinement /
        analytical correction can overwrite them safely.
        """
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        d_out = np.empty((n1, n2))
        inv_d_out = np.empty((n1, n2))
        ndr_out = np.empty((n1, n2))
        for i in prange(n1):
            p0 = pos1[i, 0]
            p1 = pos1[i, 1]
            p2 = pos1[i, 2]
            nx = nvec1[i, 0]
            ny = nvec1[i, 1]
            nz = nvec1[i, 2]
            for j in range(n2):
                rx = p0 - pos2[j, 0]
                ry = p1 - pos2[j, 1]
                rz = p2 - pos2[j, 2]
                d2 = rx * rx + ry * ry + rz * rz
                d = d2 ** 0.5
                if same and i == j:
                    d = _EPS
                    inv_d = 1.0 / _EPS
                    ndotr = 0.0
                else:
                    if d < _EPS:
                        d = _EPS
                    inv_d = 1.0 / d
                    ndotr = nx * rx + ny * ry + nz * rz
                d_out[i, j] = d
                inv_d_out[i, j] = inv_d
                ndr_out[i, j] = ndotr
        return d_out, inv_d_out, ndr_out

    @njit(parallel = True, fastmath = False, cache = True)
    def _green_ret_dGFr(pos1, pos2, nvec1, area2, same):
        """
        Like _green_ret_dGF but additionally returns the relative vector
        components rx, ry, rz needed for cart deriv / Gp.
        """
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        d_out = np.empty((n1, n2))
        inv_d_out = np.empty((n1, n2))
        ndr_out = np.empty((n1, n2))
        rx_out = np.empty((n1, n2))
        ry_out = np.empty((n1, n2))
        rz_out = np.empty((n1, n2))
        for i in prange(n1):
            p0 = pos1[i, 0]
            p1 = pos1[i, 1]
            p2 = pos1[i, 2]
            nx = nvec1[i, 0]
            ny = nvec1[i, 1]
            nz = nvec1[i, 2]
            for j in range(n2):
                rx = p0 - pos2[j, 0]
                ry = p1 - pos2[j, 1]
                rz = p2 - pos2[j, 2]
                d2 = rx * rx + ry * ry + rz * rz
                d = d2 ** 0.5
                if same and i == j:
                    d = _EPS
                    inv_d = 1.0 / _EPS
                    ndotr = 0.0
                else:
                    if d < _EPS:
                        d = _EPS
                    inv_d = 1.0 / d
                    ndotr = nx * rx + ny * ry + nz * rz
                d_out[i, j] = d
                inv_d_out[i, j] = inv_d
                ndr_out[i, j] = ndotr
                rx_out[i, j] = rx
                ry_out[i, j] = ry
                rz_out[i, j] = rz
        return d_out, inv_d_out, ndr_out, rx_out, ry_out, rz_out

    @njit(parallel = True, fastmath = False, cache = True)
    def _green_ret_dGF_slice(pos1, pos2, nvec1, same, col_offset):
        """
        Build distance d, inv_d, n_dot_r for a column slice of pos2.

        Operates on pos2 already sliced to ``pos2[col_start:col_stop]``;
        ``col_offset`` is the global column index of pos2[0] in the
        original ``self.p2.pos``.  Used to detect the diagonal in the
        self-block case: ``row i corresponds to col i - col_offset``,
        so the diagonal entry of the slice is at ``(j + col_offset, j)``.
        """
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        d_out = np.empty((n1, n2))
        inv_d_out = np.empty((n1, n2))
        ndr_out = np.empty((n1, n2))
        for i in prange(n1):
            p0 = pos1[i, 0]
            p1 = pos1[i, 1]
            p2 = pos1[i, 2]
            nx = nvec1[i, 0]
            ny = nvec1[i, 1]
            nz = nvec1[i, 2]
            for j in range(n2):
                rx = p0 - pos2[j, 0]
                ry = p1 - pos2[j, 1]
                rz = p2 - pos2[j, 2]
                d2 = rx * rx + ry * ry + rz * rz
                d = d2 ** 0.5
                if same and i == (j + col_offset):
                    d = _EPS
                    inv_d = 1.0 / _EPS
                    ndotr = 0.0
                else:
                    if d < _EPS:
                        d = _EPS
                    inv_d = 1.0 / d
                    ndotr = nx * rx + ny * ry + nz * rz
                d_out[i, j] = d
                inv_d_out[i, j] = inv_d
                ndr_out[i, j] = ndotr
        return d_out, inv_d_out, ndr_out

    @njit(parallel = True, fastmath = False, cache = True)
    def _green_ret_dGFr_slice(pos1, pos2, nvec1, same, col_offset):
        """
        Like _green_ret_dGF_slice but also returns rx, ry, rz components.
        """
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        d_out = np.empty((n1, n2))
        inv_d_out = np.empty((n1, n2))
        ndr_out = np.empty((n1, n2))
        rx_out = np.empty((n1, n2))
        ry_out = np.empty((n1, n2))
        rz_out = np.empty((n1, n2))
        for i in prange(n1):
            p0 = pos1[i, 0]
            p1 = pos1[i, 1]
            p2 = pos1[i, 2]
            nx = nvec1[i, 0]
            ny = nvec1[i, 1]
            nz = nvec1[i, 2]
            for j in range(n2):
                rx = p0 - pos2[j, 0]
                ry = p1 - pos2[j, 1]
                rz = p2 - pos2[j, 2]
                d2 = rx * rx + ry * ry + rz * rz
                d = d2 ** 0.5
                if same and i == (j + col_offset):
                    d = _EPS
                    inv_d = 1.0 / _EPS
                    ndotr = 0.0
                else:
                    if d < _EPS:
                        d = _EPS
                    inv_d = 1.0 / d
                    ndotr = nx * rx + ny * ry + nz * rz
                d_out[i, j] = d
                inv_d_out[i, j] = inv_d
                ndr_out[i, j] = ndotr
                rx_out[i, j] = rx
                ry_out[i, j] = ry
                rz_out[i, j] = rz
        return d_out, inv_d_out, ndr_out, rx_out, ry_out, rz_out


def green_ret_distances(pos1, pos2, nvec1, area2, same, want_r = False):
    """
    Compute distance and inner products for retarded Green function.

    Returns
    -------
    d, inv_d, n_dot_r              (when want_r is False)
    d, inv_d, n_dot_r, rx, ry, rz  (when want_r is True)

    Falls back to numpy broadcasting when numba is unavailable / disabled.
    """
    pos1 = np.ascontiguousarray(pos1, dtype = np.float64)
    pos2 = np.ascontiguousarray(pos2, dtype = np.float64)
    nvec1 = np.ascontiguousarray(nvec1, dtype = np.float64)
    area2 = np.ascontiguousarray(area2, dtype = np.float64)

    if numba_enabled():
        if want_r:
            return _green_ret_dGFr(pos1, pos2, nvec1, area2, same)
        return _green_ret_dGF(pos1, pos2, nvec1, area2, same)

    return _green_ret_distances_numpy(pos1, pos2, nvec1, area2, same, want_r)


def _green_ret_distances_numpy(pos1, pos2, nvec1, area2, same, want_r):
    """Reference numpy implementation (used when MNPBEM_NUMBA=0)."""
    rx = pos1[:, 0:1] - pos2[:, 0]
    ry = pos1[:, 1:2] - pos2[:, 1]
    rz = pos1[:, 2:3] - pos2[:, 2]
    d = np.sqrt(rx * rx + ry * ry + rz * rz)
    d = np.maximum(d, np.finfo(float).eps)
    inv_d = 1.0 / d
    n_dot_r = (nvec1[:, 0:1] * rx +
               nvec1[:, 1:2] * ry +
               nvec1[:, 2:3] * rz)
    if want_r:
        return d, inv_d, n_dot_r, rx, ry, rz
    return d, inv_d, n_dot_r


def green_ret_distances_slice(pos1, pos2_slice, nvec1, same, col_offset,
                              want_r = False):
    """
    Compute distances for a column slice of pos2.

    Parameters
    ----------
    pos1 : (M, 3)
    pos2_slice : (ncol, 3)  -- already sliced to pos2[col_start:col_stop]
    nvec1 : (M, 3)
    same : bool
        True when the original p1 is p2.  Required so the diagonal
        of the self-block (where row index equals global column index)
        is forced to d=eps.
    col_offset : int
        Global column index of pos2_slice[0], i.e., col_start.
    want_r : bool
        Also return rx, ry, rz components (needed for cart deriv / Gp).

    Returns
    -------
    d, inv_d, n_dot_r              (when want_r is False)
    d, inv_d, n_dot_r, rx, ry, rz  (when want_r is True)
    """
    pos1 = np.ascontiguousarray(pos1, dtype = np.float64)
    pos2_slice = np.ascontiguousarray(pos2_slice, dtype = np.float64)
    nvec1 = np.ascontiguousarray(nvec1, dtype = np.float64)

    if numba_enabled():
        if want_r:
            return _green_ret_dGFr_slice(
                pos1, pos2_slice, nvec1, same, int(col_offset))
        return _green_ret_dGF_slice(
            pos1, pos2_slice, nvec1, same, int(col_offset))

    # Numpy fallback (no numba): broadcast straight; same/col_offset
    # handled by manual post-fix below.
    rx = pos1[:, 0:1] - pos2_slice[:, 0]
    ry = pos1[:, 1:2] - pos2_slice[:, 1]
    rz = pos1[:, 2:3] - pos2_slice[:, 2]
    d = np.sqrt(rx * rx + ry * ry + rz * rz)
    d = np.maximum(d, np.finfo(float).eps)
    n_dot_r = (nvec1[:, 0:1] * rx +
               nvec1[:, 1:2] * ry +
               nvec1[:, 2:3] * rz)
    if same:
        n1 = pos1.shape[0]
        ncol = pos2_slice.shape[0]
        # Diagonal of self-block: row i, col j where i == j + col_offset.
        # The valid j range is intersection of [0, ncol) with [-col_offset, n1-col_offset).
        j_lo = max(0, -int(col_offset))
        j_hi = min(ncol, n1 - int(col_offset))
        for j in range(j_lo, j_hi):
            i = j + int(col_offset)
            d[i, j] = np.finfo(float).eps
            n_dot_r[i, j] = 0.0
    inv_d = 1.0 / d
    if want_r:
        return d, inv_d, n_dot_r, rx, ry, rz
    return d, inv_d, n_dot_r


# ------------------------------------------------------------------ GPU path
#
# The GPU helpers below mirror the numba kernels but execute as cupy
# elementwise expressions on device. They return cupy ndarrays so that
# subsequent matrix builds in GreenRetRefined.eval stay on the device until
# the caller decides to bring the result back to host.

def green_ret_distances_gpu(pos1, pos2, nvec1, area2, same, want_r=False):
    """
    Compute d, inv_d, n_dot_r [, rx, ry, rz] on GPU using cupy.

    Inputs may be numpy or cupy arrays; outputs are cupy arrays (float64).
    The diagonal of self-blocks (i == j when same) is forced to d = eps
    and n_dot_r = 0 so the caller's refinement overlay can replace those
    entries safely (matches the numba and numpy paths bit-for-bit).
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("cupy is not available; cannot use GPU path")

    pos1g = _cp.asarray(pos1, dtype=_cp.float64)
    pos2g = _cp.asarray(pos2, dtype=_cp.float64)
    nvec1g = _cp.asarray(nvec1, dtype=_cp.float64)

    rx = pos1g[:, 0:1] - pos2g[:, 0]
    ry = pos1g[:, 1:2] - pos2g[:, 1]
    rz = pos1g[:, 2:3] - pos2g[:, 2]
    d = _cp.sqrt(rx * rx + ry * ry + rz * rz)
    d = _cp.maximum(d, _EPS)
    n_dot_r = (nvec1g[:, 0:1] * rx +
               nvec1g[:, 1:2] * ry +
               nvec1g[:, 2:3] * rz)
    if same:
        n = min(d.shape[0], d.shape[1])
        idx = _cp.arange(n)
        d[idx, idx] = _EPS
        n_dot_r[idx, idx] = 0.0
    inv_d = 1.0 / d

    if want_r:
        return d, inv_d, n_dot_r, rx, ry, rz
    return d, inv_d, n_dot_r


def green_ret_distances_slice_gpu(pos1, pos2_slice, nvec1, same, col_offset,
                                   want_r = False):
    """
    Column-sliced GPU version of green_ret_distances.

    Parameters
    ----------
    pos1 : (M, 3)
    pos2_slice : (ncol, 3)  -- already sliced to pos2[col_start:col_stop]
    nvec1 : (M, 3)
    same : bool
        True when original p1 is p2.  Diagonal entries of the self-block
        (row i, col j where i == j + col_offset) are forced to d=eps.
    col_offset : int
        Global column index of pos2_slice[0].
    want_r : bool
        Also return rx, ry, rz components.

    Returns
    -------
    Same shape as green_ret_distances_gpu but with (M, ncol) instead of
    (M, N).
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("cupy is not available; cannot use GPU path")

    pos1g = _cp.asarray(pos1, dtype=_cp.float64)
    pos2g = _cp.asarray(pos2_slice, dtype=_cp.float64)
    nvec1g = _cp.asarray(nvec1, dtype=_cp.float64)

    rx = pos1g[:, 0:1] - pos2g[:, 0]
    ry = pos1g[:, 1:2] - pos2g[:, 1]
    rz = pos1g[:, 2:3] - pos2g[:, 2]
    d = _cp.sqrt(rx * rx + ry * ry + rz * rz)
    d = _cp.maximum(d, _EPS)
    n_dot_r = (nvec1g[:, 0:1] * rx +
               nvec1g[:, 1:2] * ry +
               nvec1g[:, 2:3] * rz)
    if same:
        n1 = pos1g.shape[0]
        ncol = pos2g.shape[0]
        co = int(col_offset)
        # j range where (j + co) is a valid row in [0, n1)
        j_lo = max(0, -co)
        j_hi = min(ncol, n1 - co)
        if j_hi > j_lo:
            j_idx = _cp.arange(j_lo, j_hi)
            i_idx = j_idx + co
            d[i_idx, j_idx] = _EPS
            n_dot_r[i_idx, j_idx] = 0.0
    inv_d = 1.0 / d

    if want_r:
        return d, inv_d, n_dot_r, rx, ry, rz
    return d, inv_d, n_dot_r


def ret_phase_gpu(d, k):
    """exp(1j * k * d) on GPU (complex128 cupy array)."""
    return _cp.exp(1j * k * d)


def ret_G_pre_gpu(inv_d, area2):
    """Pre-phase G on GPU. inv_d: (M,N), area2: (N,)."""
    area2g = _cp.asarray(area2, dtype=_cp.float64)
    return (inv_d * area2g[None, :]).astype(_cp.complex128)


def ret_F_norm_pre_gpu(inv_d, inv_d2, n_dot_r, area2, k):
    """Pre-phase F (norm path) on GPU."""
    area2g = _cp.asarray(area2, dtype=_cp.float64)
    return (n_dot_r * (1j * k - inv_d) * inv_d2 * area2g[None, :]).astype(_cp.complex128)


def ret_F_cart_pre_gpu(inv_d, inv_d2, rx, ry, rz, nvec1, area2, k):
    """Pre-phase F (cart path) on GPU."""
    nvec1g = _cp.asarray(nvec1, dtype=_cp.float64)
    area2g = _cp.asarray(area2, dtype=_cp.float64)
    f_aux = (1j * k - inv_d) * inv_d2
    F = ((nvec1g[:, 0:1] * (f_aux * rx) +
          nvec1g[:, 1:2] * (f_aux * ry) +
          nvec1g[:, 2:3] * (f_aux * rz)) * area2g[None, :])
    return F.astype(_cp.complex128)


def ret_Gp_pre_gpu(inv_d, inv_d2, rx, ry, rz, area2, k):
    """Pre-phase Gp (M,3,N) on GPU."""
    area2g = _cp.asarray(area2, dtype=_cp.float64)
    f_aux = ((1j * k - inv_d) * inv_d2 * area2g[None, :])
    Gp_x = (rx * f_aux).astype(_cp.complex128)
    Gp_y = (ry * f_aux).astype(_cp.complex128)
    Gp_z = (rz * f_aux).astype(_cp.complex128)
    return _cp.stack([Gp_x, Gp_y, Gp_z], axis=1)


def apply_phase_2d_gpu(g, phase):
    """In-place g *= phase for cupy (M,N) complex matrices."""
    g *= phase
    return g


def apply_phase_3d_axis02_gpu(g, phase):
    """In-place g[m,:,n] *= phase[m,n]. g: (M,3,N), phase: (M,N) on GPU."""
    g *= phase[:, None, :]
    return g


def to_host(arr):
    """Bring a cupy array to host numpy; pass-through if already numpy."""
    if CUPY_AVAILABLE and isinstance(arr, _cp.ndarray):
        return _cp.asnumpy(arr)
    return arr
