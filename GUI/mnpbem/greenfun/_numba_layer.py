"""Numba-accelerated trilinear/bilinear interpolation kernels for layer
Green-function tabulation.

Activated via the ``MNPBEM_NUMBA`` environment variable.  Falls back to
``scipy.interpolate.RegularGridInterpolator`` when numba is unavailable
or the variable is unset.

The interpolator mimics RegularGridInterpolator(method='linear',
bounds_error=False, fill_value=None): inside the grid we use plain
multilinear interpolation, outside we extrapolate with the gradient of
the boundary cell (i.e. clamp the cell index but keep the fractional
weight unbounded).
"""

import os
import numpy as np

try:
    import numba
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _layer_parallel() -> bool:
    """Whether the layer interpolation kernels run multithreaded.

    The substrate BEM build interpolates the tabulated Green function at
    O(n^2) face-pair query points (≈7.3M points for a 2696-face dimer,
    ≈230M for a 15072-face dimer) for each of 5 components x 3 fields.
    Parallelising the per-point loop across cores gives a near-linear
    speed-up there.  Default ON; set ``MNPBEM_NUMBA_PARALLEL=0`` to force
    the serial kernels (e.g. when the host is oversubscribed).
    """
    return os.environ.get('MNPBEM_NUMBA_PARALLEL', '1').strip() not in (
        '0', 'false', 'False')


def _numba_enabled() -> bool:
    return _HAS_NUMBA and os.environ.get('MNPBEM_NUMBA', '').strip() not in ('', '0', 'false', 'False')


def _numba_layer_preferred() -> bool:
    """Numba multilinear interpolation is the default for the layer Green
    tabulation whenever numba is importable.

    The kernel is numerically identical to scipy's linear
    RegularGridInterpolator, but the scipy path rebuilds an interpolator
    object per component/field on every call — a heavy O(n^2) cost on the
    substrate BEM build (≈19s/wavelength on a 2696-face dimer).  Default
    ON; set ``MNPBEM_NUMBA=0`` to force the scipy reference path for
    debugging or bit-exact comparison.
    """
    if not _HAS_NUMBA:
        return False
    return os.environ.get('MNPBEM_NUMBA', '').strip() not in ('0', 'false', 'False')


# ---------------------------------------------------------------------------
# Helpers for index location (linear search; grids are small, n <= ~50)
# ---------------------------------------------------------------------------

if _HAS_NUMBA:

    @njit(cache=True, fastmath=False)
    def _locate_cell(grid: np.ndarray, x: float) -> tuple:
        """Locate the cell (i, t) such that grid[i] + t*(grid[i+1]-grid[i]) = x.

        Returns (i, t) clamped so that 0 <= i <= len(grid)-2, with t allowed
        to lie outside [0, 1] (linear extrapolation outside the grid).
        """
        n = grid.shape[0]
        if n < 2:
            return 0, 0.0

        # Binary search for first index i where grid[i+1] > x
        lo = 0
        hi = n - 2
        while lo < hi:
            mid = (lo + hi) // 2
            if grid[mid + 1] <= x:
                lo = mid + 1
            else:
                hi = mid
        i = lo
        denom = grid[i + 1] - grid[i]
        if denom == 0.0:
            t = 0.0
        else:
            t = (x - grid[i]) / denom
        return i, t

    # Serial inner loop: parallel adds ~1-2 ms launch overhead per call
    # which dominates for the typical BEM call sizes (n ~ a few thousand
    # called 90+ times per solve).  Parallel only wins for n >> 10^5.
    @njit(cache=True, parallel=False, fastmath=False)
    def _trilinear_complex(
            grid_r: np.ndarray,
            grid_z1: np.ndarray,
            grid_z2: np.ndarray,
            data: np.ndarray,
            r_q: np.ndarray,
            z1_q: np.ndarray,
            z2_q: np.ndarray,
            out: np.ndarray,
    ) -> None:
        """Trilinear interpolation of complex 3D data on a regular grid.

        data : (nr, nz1, nz2) complex128
        r_q, z1_q, z2_q : (n,) float64
        out : (n,) complex128 — filled in place.
        """
        n = r_q.shape[0]
        for q in range(n):
            ir, tr = _locate_cell(grid_r, r_q[q])
            iz, tz = _locate_cell(grid_z1, z1_q[q])
            iw, tw = _locate_cell(grid_z2, z2_q[q])

            c000 = data[ir,     iz,     iw]
            c100 = data[ir + 1, iz,     iw]
            c010 = data[ir,     iz + 1, iw]
            c110 = data[ir + 1, iz + 1, iw]
            c001 = data[ir,     iz,     iw + 1]
            c101 = data[ir + 1, iz,     iw + 1]
            c011 = data[ir,     iz + 1, iw + 1]
            c111 = data[ir + 1, iz + 1, iw + 1]

            # Interpolate along r
            c00 = c000 * (1.0 - tr) + c100 * tr
            c10 = c010 * (1.0 - tr) + c110 * tr
            c01 = c001 * (1.0 - tr) + c101 * tr
            c11 = c011 * (1.0 - tr) + c111 * tr
            # Along z1
            c0 = c00 * (1.0 - tz) + c10 * tz
            c1 = c01 * (1.0 - tz) + c11 * tz
            # Along z2
            out[q] = c0 * (1.0 - tw) + c1 * tw

    @njit(cache=True, parallel=False, fastmath=False)
    def _bilinear_complex(
            grid_r: np.ndarray,
            grid_z: np.ndarray,
            data: np.ndarray,
            r_q: np.ndarray,
            z_q: np.ndarray,
            out: np.ndarray,
    ) -> None:
        """Bilinear interpolation of complex 2D data on a regular grid.

        data : (nr, nz) complex128
        r_q, z_q : (n,) float64
        out : (n,) complex128 — filled in place.
        """
        n = r_q.shape[0]
        for q in range(n):
            ir, tr = _locate_cell(grid_r, r_q[q])
            iz, tz = _locate_cell(grid_z, z_q[q])

            c00 = data[ir,     iz]
            c10 = data[ir + 1, iz]
            c01 = data[ir,     iz + 1]
            c11 = data[ir + 1, iz + 1]

            c0 = c00 * (1.0 - tr) + c10 * tr
            c1 = c01 * (1.0 - tr) + c11 * tr
            out[q] = c0 * (1.0 - tz) + c1 * tz

    # Parallel variants: identical math, prange over the query points.
    # Used for the O(n^2) substrate face-pair grids where the per-point
    # loop dwarfs numba's thread-launch overhead.
    @njit(cache=True, parallel=True, fastmath=False)
    def _trilinear_complex_par(
            grid_r: np.ndarray,
            grid_z1: np.ndarray,
            grid_z2: np.ndarray,
            data: np.ndarray,
            r_q: np.ndarray,
            z1_q: np.ndarray,
            z2_q: np.ndarray,
            out: np.ndarray,
    ) -> None:
        n = r_q.shape[0]
        for q in prange(n):
            ir, tr = _locate_cell(grid_r, r_q[q])
            iz, tz = _locate_cell(grid_z1, z1_q[q])
            iw, tw = _locate_cell(grid_z2, z2_q[q])

            c000 = data[ir,     iz,     iw]
            c100 = data[ir + 1, iz,     iw]
            c010 = data[ir,     iz + 1, iw]
            c110 = data[ir + 1, iz + 1, iw]
            c001 = data[ir,     iz,     iw + 1]
            c101 = data[ir + 1, iz,     iw + 1]
            c011 = data[ir,     iz + 1, iw + 1]
            c111 = data[ir + 1, iz + 1, iw + 1]

            c00 = c000 * (1.0 - tr) + c100 * tr
            c10 = c010 * (1.0 - tr) + c110 * tr
            c01 = c001 * (1.0 - tr) + c101 * tr
            c11 = c011 * (1.0 - tr) + c111 * tr
            c0 = c00 * (1.0 - tz) + c10 * tz
            c1 = c01 * (1.0 - tz) + c11 * tz
            out[q] = c0 * (1.0 - tw) + c1 * tw

    @njit(cache=True, parallel=True, fastmath=False)
    def _bilinear_complex_par(
            grid_r: np.ndarray,
            grid_z: np.ndarray,
            data: np.ndarray,
            r_q: np.ndarray,
            z_q: np.ndarray,
            out: np.ndarray,
    ) -> None:
        n = r_q.shape[0]
        for q in prange(n):
            ir, tr = _locate_cell(grid_r, r_q[q])
            iz, tz = _locate_cell(grid_z, z_q[q])

            c00 = data[ir,     iz]
            c10 = data[ir + 1, iz]
            c01 = data[ir,     iz + 1]
            c11 = data[ir + 1, iz + 1]

            c0 = c00 * (1.0 - tr) + c10 * tr
            c1 = c01 * (1.0 - tr) + c11 * tr
            out[q] = c0 * (1.0 - tz) + c1 * tz

    # ----------------------------------------------------------------
    # Batched kernels: locate each query point's cell ONCE, then
    # interpolate K stacked data slabs.  The cell index + fractional
    # weight depend only on (r, z1, z2), so doing the binary search /
    # weight math once and reusing it across all components (5) x fields
    # (3) = 15 slabs removes a 15x redundancy from the substrate build's
    # O(n^2) interpolation — the dominant per-wavelength cost.
    # ----------------------------------------------------------------
    @njit(cache=True, parallel=True, fastmath=False)
    def _bilinear_complex_batch_par(
            grid_r: np.ndarray,
            grid_z: np.ndarray,
            data: np.ndarray,      # (K, nr, nz)
            r_q: np.ndarray,
            z_q: np.ndarray,
            out: np.ndarray,       # (K, n)
    ) -> None:
        n = r_q.shape[0]
        K = data.shape[0]
        for q in prange(n):
            ir, tr = _locate_cell(grid_r, r_q[q])
            iz, tz = _locate_cell(grid_z, z_q[q])
            w00 = (1.0 - tr) * (1.0 - tz)
            w10 = tr * (1.0 - tz)
            w01 = (1.0 - tr) * tz
            w11 = tr * tz
            for kk in range(K):
                out[kk, q] = (data[kk, ir,     iz]     * w00
                            + data[kk, ir + 1, iz]     * w10
                            + data[kk, ir,     iz + 1] * w01
                            + data[kk, ir + 1, iz + 1] * w11)

    @njit(cache=True, parallel=True, fastmath=False)
    def _trilinear_complex_batch_par(
            grid_r: np.ndarray,
            grid_z1: np.ndarray,
            grid_z2: np.ndarray,
            data: np.ndarray,      # (K, nr, nz1, nz2)
            r_q: np.ndarray,
            z1_q: np.ndarray,
            z2_q: np.ndarray,
            out: np.ndarray,       # (K, n)
    ) -> None:
        n = r_q.shape[0]
        K = data.shape[0]
        for q in prange(n):
            ir, tr = _locate_cell(grid_r, r_q[q])
            iz, tz = _locate_cell(grid_z1, z1_q[q])
            iw, tw = _locate_cell(grid_z2, z2_q[q])
            w000 = (1.0 - tr) * (1.0 - tz) * (1.0 - tw)
            w100 = tr * (1.0 - tz) * (1.0 - tw)
            w010 = (1.0 - tr) * tz * (1.0 - tw)
            w110 = tr * tz * (1.0 - tw)
            w001 = (1.0 - tr) * (1.0 - tz) * tw
            w101 = tr * (1.0 - tz) * tw
            w011 = (1.0 - tr) * tz * tw
            w111 = tr * tz * tw
            for kk in range(K):
                out[kk, q] = (data[kk, ir,     iz,     iw]     * w000
                            + data[kk, ir + 1, iz,     iw]     * w100
                            + data[kk, ir,     iz + 1, iw]     * w010
                            + data[kk, ir + 1, iz + 1, iw]     * w110
                            + data[kk, ir,     iz,     iw + 1] * w001
                            + data[kk, ir + 1, iz,     iw + 1] * w101
                            + data[kk, ir,     iz + 1, iw + 1] * w011
                            + data[kk, ir + 1, iz + 1, iw + 1] * w111)


def trilinear_complex_batch(grid, data_stack, points):
    """Locate-once batched multilinear interpolation of stacked complex data.

    grid       : tuple of 1D axis grids (len 2 or 3)
    data_stack : complex array (K, *grid_shape)
    points     : (n, ndim) query coordinates
    Returns (K, n) complex array.  Falls back to per-slab scipy when numba
    is unavailable.
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    K = data_stack.shape[0]
    n = points.shape[0]
    if not _numba_layer_preferred():
        out = np.empty((K, n), dtype=np.complex128)
        for kk in range(K):
            out[kk] = _scipy_fallback(grid, data_stack[kk], points)
        return out

    data_c = np.ascontiguousarray(data_stack, dtype=np.complex128)
    out = np.empty((K, n), dtype=np.complex128)
    ndim = data_c.ndim - 1
    if ndim == 2:
        _bilinear_complex_batch_par(
            np.ascontiguousarray(grid[0], dtype=np.float64),
            np.ascontiguousarray(grid[1], dtype=np.float64),
            data_c,
            np.ascontiguousarray(points[:, 0], dtype=np.float64),
            np.ascontiguousarray(points[:, 1], dtype=np.float64),
            out,
        )
    elif ndim == 3:
        _trilinear_complex_batch_par(
            np.ascontiguousarray(grid[0], dtype=np.float64),
            np.ascontiguousarray(grid[1], dtype=np.float64),
            np.ascontiguousarray(grid[2], dtype=np.float64),
            data_c,
            np.ascontiguousarray(points[:, 0], dtype=np.float64),
            np.ascontiguousarray(points[:, 1], dtype=np.float64),
            np.ascontiguousarray(points[:, 2], dtype=np.float64),
            out,
        )
    else:
        out = np.empty((K, n), dtype=np.complex128)
        for kk in range(K):
            out[kk] = _scipy_fallback(grid, data_stack[kk], points)
    return out


def trilinear_complex(grid, data, points):
    """Public wrapper that dispatches to numba when enabled, else RGI.

    grid   : tuple of 1D ndarrays (axis grids)
    data   : complex array with shape == (len(g) for g in grid)
    points : (n, ndim) ndarray of query coordinates
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    if not _numba_layer_preferred():
        return _scipy_fallback(grid, data, points)

    data_c = np.ascontiguousarray(data, dtype=np.complex128)
    out = np.empty(points.shape[0], dtype=np.complex128)
    # Multithread only when the query batch is large enough to amortise the
    # numba thread-launch overhead (small dipole/EELS calls stay serial).
    _par = _layer_parallel() and points.shape[0] >= 50000

    if data_c.ndim == 3:
        _kern = _trilinear_complex_par if _par else _trilinear_complex
        _kern(
            np.ascontiguousarray(grid[0], dtype=np.float64),
            np.ascontiguousarray(grid[1], dtype=np.float64),
            np.ascontiguousarray(grid[2], dtype=np.float64),
            data_c,
            np.ascontiguousarray(points[:, 0], dtype=np.float64),
            np.ascontiguousarray(points[:, 1], dtype=np.float64),
            np.ascontiguousarray(points[:, 2], dtype=np.float64),
            out,
        )
    elif data_c.ndim == 2:
        _kern = _bilinear_complex_par if _par else _bilinear_complex
        _kern(
            np.ascontiguousarray(grid[0], dtype=np.float64),
            np.ascontiguousarray(grid[1], dtype=np.float64),
            data_c,
            np.ascontiguousarray(points[:, 0], dtype=np.float64),
            np.ascontiguousarray(points[:, 1], dtype=np.float64),
            out,
        )
    else:
        return _scipy_fallback(grid, data, points)
    return out


def _scipy_fallback(grid, data, points):
    """Slow path identical to GreenTabLayer._interp_complex."""
    from scipy.interpolate import RegularGridInterpolator
    val_r = RegularGridInterpolator(
        grid, data.real, method='linear',
        bounds_error=False, fill_value=None)(points)
    val_i = RegularGridInterpolator(
        grid, data.imag, method='linear',
        bounds_error=False, fill_value=None)(points)
    return val_r + 1j * val_i
