"""
Numba-accelerated kernels for the M field-points x N source faces
near-field evaluation that drives MeshField.

The kernels operate on the per-wavelength dense fill of G, F and Gp that
sits inside the Green function evaluation loop for both the quasistatic
(CompGreenStat) and retarded (GreenRetRefined) paths. They are only used
when the source / target particles are *distinct* (p1 is not p2), which is
the meshfield / observer pattern -- BEM-self-block kernels live in the
sibling assembly modules (`_numba_kernels`, `_numba_ret_kernels`).

Activation:
  - default: enabled when numba is importable
  - disable by setting environment variable MNPBEM_NUMBA = 0

The retarded kernels return *pre-phase* matrices so that callers can apply
the refinement overlay (which targets pre-phase values) before multiplying
by exp(i k d). Phase application is itself wrapped in numba kernels.
"""

import os
import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def numba_enabled():
    """Return True iff numba kernels should be used."""
    if not NUMBA_AVAILABLE:
        return False
    return os.environ.get('MNPBEM_NUMBA', '1') != '0'


if NUMBA_AVAILABLE:

    # ---------------------------------------------------------------- stat

    @njit(parallel = True, fastmath = False, cache = True)
    def _stat_Gp(rx, ry, rz, inv_d3, area2):
        """
        Quasistatic Cartesian derivative Gp[m, k, n] = -r[m, n, k] / d^3 * area[n].

        Inputs are precomputed (M, N) arrays for rx, ry, rz and inv_d3 (= 1/d^3).
        Output: (M, 3, N) float64.
        """
        M = rx.shape[0]
        N = rx.shape[1]
        Gp = np.empty((M, 3, N), dtype = np.float64)
        for m in prange(M):
            for n in range(N):
                w = -area2[n] * inv_d3[m, n]
                Gp[m, 0, n] = rx[m, n] * w
                Gp[m, 1, n] = ry[m, n] * w
                Gp[m, 2, n] = rz[m, n] * w
        return Gp

    # ---------------------------------------------------------- ret kernels

    @njit(parallel = True, fastmath = False, cache = True)
    def _ret_phase(d, k):
        """exp(1j * k * d) elementwise. Returns (M, N) complex128."""
        M = d.shape[0]
        N = d.shape[1]
        out = np.empty((M, N), dtype = np.complex128)
        for m in prange(M):
            for n in range(N):
                out[m, n] = np.exp(1j * k * d[m, n])
        return out

    @njit(parallel = True, fastmath = False, cache = True)
    def _ret_G_pre(inv_d, area2):
        """Pre-phase G[m, n] = area[n] / d. Returns (M, N) complex128."""
        M = inv_d.shape[0]
        N = inv_d.shape[1]
        out = np.empty((M, N), dtype = np.complex128)
        for m in prange(M):
            for n in range(N):
                out[m, n] = inv_d[m, n] * area2[n]
        return out

    @njit(parallel = True, fastmath = False, cache = True)
    def _ret_F_norm_pre(inv_d, inv_d2, n_dot_r, area2, k):
        """Pre-phase F (norm path)."""
        M = inv_d.shape[0]
        N = inv_d.shape[1]
        out = np.empty((M, N), dtype = np.complex128)
        for m in prange(M):
            for n in range(N):
                out[m, n] = (n_dot_r[m, n] *
                             (1j * k - inv_d[m, n]) * inv_d2[m, n] *
                             area2[n])
        return out

    @njit(parallel = True, fastmath = False, cache = True)
    def _ret_F_cart_pre(inv_d, inv_d2, rx, ry, rz, nvec1, area2, k):
        """Pre-phase F (cart path: F = inner(nvec, Gp_pre))."""
        M = inv_d.shape[0]
        N = inv_d.shape[1]
        out = np.empty((M, N), dtype = np.complex128)
        for m in prange(M):
            nx = nvec1[m, 0]
            ny = nvec1[m, 1]
            nz = nvec1[m, 2]
            for n in range(N):
                f_aux = (1j * k - inv_d[m, n]) * inv_d2[m, n] * area2[n]
                out[m, n] = (nx * rx[m, n] + ny * ry[m, n] + nz * rz[m, n]) * f_aux
        return out

    @njit(parallel = True, fastmath = False, cache = True)
    def _ret_Gp_pre(inv_d, inv_d2, rx, ry, rz, area2, k):
        """Pre-phase Gp (M, 3, N) complex128."""
        M = inv_d.shape[0]
        N = inv_d.shape[1]
        out = np.empty((M, 3, N), dtype = np.complex128)
        for m in prange(M):
            for n in range(N):
                f_aux = (1j * k - inv_d[m, n]) * inv_d2[m, n] * area2[n]
                out[m, 0, n] = rx[m, n] * f_aux
                out[m, 1, n] = ry[m, n] * f_aux
                out[m, 2, n] = rz[m, n] * f_aux
        return out

    @njit(parallel = True, fastmath = False, cache = True)
    def _apply_phase_2d(g, phase):
        """In-place g *= phase, both (M, N) complex128."""
        M = g.shape[0]
        N = g.shape[1]
        for m in prange(M):
            for n in range(N):
                g[m, n] = g[m, n] * phase[m, n]

    @njit(parallel = True, fastmath = False, cache = True)
    def _apply_phase_3d_axis02(g, phase):
        """In-place g *= phase. g: (M, 3, N), phase: (M, N), broadcast on axis 1."""
        M = g.shape[0]
        N = g.shape[2]
        for m in prange(M):
            for n in range(N):
                p = phase[m, n]
                g[m, 0, n] = g[m, 0, n] * p
                g[m, 1, n] = g[m, 1, n] * p
                g[m, 2, n] = g[m, 2, n] * p


# ---------------------------------------------------------------- API

def stat_Gp(r, d_safe, area2):
    """
    Vectorized fallback wrapper for Gp = -r / d^3 * area.

    Parameters
    ----------
    r : (M, N, 3) float64
    d_safe : (M, N) float64
    area2 : (N,) float64

    Returns
    -------
    Gp : (M, 3, N) float64
    """
    if numba_enabled():
        rx = np.ascontiguousarray(r[:, :, 0], dtype = np.float64)
        ry = np.ascontiguousarray(r[:, :, 1], dtype = np.float64)
        rz = np.ascontiguousarray(r[:, :, 2], dtype = np.float64)
        inv_d3 = 1.0 / (np.ascontiguousarray(d_safe, dtype = np.float64) ** 3)
        area = np.ascontiguousarray(area2, dtype = np.float64)
        return _stat_Gp(rx, ry, rz, inv_d3, area)
    Gp = -r / (d_safe[:, :, np.newaxis] ** 3) * area2[np.newaxis, :, np.newaxis]
    return np.transpose(Gp, (0, 2, 1))


def ret_phase(d, k):
    """exp(1j * k * d) — (M, N) complex128."""
    if numba_enabled():
        return _ret_phase(np.ascontiguousarray(d, dtype = np.float64), complex(k))
    return np.exp(1j * k * d)


def ret_G_pre(inv_d, area2):
    if numba_enabled():
        return _ret_G_pre(np.ascontiguousarray(inv_d, dtype = np.float64),
                          np.ascontiguousarray(area2, dtype = np.float64))
    return inv_d * area2[np.newaxis, :] + 0j


def ret_F_norm_pre(inv_d, inv_d2, n_dot_r, area2, k):
    if numba_enabled():
        return _ret_F_norm_pre(
            np.ascontiguousarray(inv_d, dtype = np.float64),
            np.ascontiguousarray(inv_d2, dtype = np.float64),
            np.ascontiguousarray(n_dot_r, dtype = np.float64),
            np.ascontiguousarray(area2, dtype = np.float64),
            complex(k))
    return n_dot_r * (1j * k - inv_d) * inv_d2 * area2[np.newaxis, :]


def ret_F_cart_pre(inv_d, inv_d2, rx, ry, rz, nvec1, area2, k):
    if numba_enabled():
        return _ret_F_cart_pre(
            np.ascontiguousarray(inv_d, dtype = np.float64),
            np.ascontiguousarray(inv_d2, dtype = np.float64),
            np.ascontiguousarray(rx, dtype = np.float64),
            np.ascontiguousarray(ry, dtype = np.float64),
            np.ascontiguousarray(rz, dtype = np.float64),
            np.ascontiguousarray(nvec1, dtype = np.float64),
            np.ascontiguousarray(area2, dtype = np.float64),
            complex(k))
    f_aux = (1j * k - inv_d) * inv_d2
    return ((nvec1[:, 0:1] * (f_aux * rx) +
             nvec1[:, 1:2] * (f_aux * ry) +
             nvec1[:, 2:3] * (f_aux * rz)) * area2[np.newaxis, :])


def ret_Gp_pre(inv_d, inv_d2, rx, ry, rz, area2, k):
    if numba_enabled():
        return _ret_Gp_pre(
            np.ascontiguousarray(inv_d, dtype = np.float64),
            np.ascontiguousarray(inv_d2, dtype = np.float64),
            np.ascontiguousarray(rx, dtype = np.float64),
            np.ascontiguousarray(ry, dtype = np.float64),
            np.ascontiguousarray(rz, dtype = np.float64),
            np.ascontiguousarray(area2, dtype = np.float64),
            complex(k))
    f_aux = (1j * k - inv_d) * inv_d2
    Gp_x = f_aux * rx * area2[np.newaxis, :]
    Gp_y = f_aux * ry * area2[np.newaxis, :]
    Gp_z = f_aux * rz * area2[np.newaxis, :]
    return np.stack([Gp_x, Gp_y, Gp_z], axis = 1)


def apply_phase_2d(g, phase):
    """In-place g *= phase for (M, N) complex matrix."""
    if numba_enabled():
        _apply_phase_2d(g, phase)
        return g
    g *= phase
    return g


def apply_phase_3d_axis02(g, phase):
    """In-place g[m, :, n] *= phase[m, n] for g: (M, 3, N), phase: (M, N)."""
    if numba_enabled():
        _apply_phase_3d_axis02(g, phase)
        return g
    g *= phase[:, np.newaxis, :]
    return g
