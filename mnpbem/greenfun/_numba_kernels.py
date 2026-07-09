"""
Numba-accelerated kernels for quasistatic Green function assembly.

Used by CompGreenStat._compute_greenstat for the off-diagonal far-field
fill of G, F, and Gp matrices. Self-block (i == j) and refinement
overrides are still applied by the Python caller after these kernels run.

Activation:
  - default: enabled
  - disable by setting environment variable MNPBEM_NUMBA=0
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

    @njit(parallel=True, fastmath=False, cache=True)
    def _green_stat_GF(pos1, pos2, nvec1, area2, same):
        """
        Build G and F matrices for quasistatic Green function.

        G[i, j] = area2[j] / |pos1[i] - pos2[j]|
        F[i, j] = -dot(nvec1[i], pos1[i] - pos2[j]) * area2[j] / |...|^3

        When `same` is True, the i == j entries are zeroed (caller fills
        the diagonal via refinement / analytical correction).
        """
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        G = np.zeros((n1, n2))
        F = np.zeros((n1, n2))
        for i in prange(n1):
            p0 = pos1[i, 0]
            p1 = pos1[i, 1]
            p2 = pos1[i, 2]
            nx = nvec1[i, 0]
            ny = nvec1[i, 1]
            nz = nvec1[i, 2]
            for j in range(n2):
                if same and i == j:
                    continue
                rx = p0 - pos2[j, 0]
                ry = p1 - pos2[j, 1]
                rz = p2 - pos2[j, 2]
                d2 = rx * rx + ry * ry + rz * rz
                d = d2 ** 0.5
                if d < 2.220446049250313e-16:
                    d = 2.220446049250313e-16
                aj = area2[j]
                G[i, j] = aj / d
                ndotr = nx * rx + ny * ry + nz * rz
                F[i, j] = -ndotr * aj / (d * d * d)
        return G, F

    @njit(parallel=True, fastmath=False, cache=True)
    def _green_stat_GF_Gp(pos1, pos2, nvec1, area2, same):
        """
        Build G, F and Gp matrices in a single pass.

        Gp[i, k, j] = -(pos1[i,k] - pos2[j,k]) * area2[j] / |...|^3
        """
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        G = np.zeros((n1, n2))
        F = np.zeros((n1, n2))
        Gp = np.zeros((n1, 3, n2))
        for i in prange(n1):
            p0 = pos1[i, 0]
            p1 = pos1[i, 1]
            p2 = pos1[i, 2]
            nx = nvec1[i, 0]
            ny = nvec1[i, 1]
            nz = nvec1[i, 2]
            for j in range(n2):
                if same and i == j:
                    continue
                rx = p0 - pos2[j, 0]
                ry = p1 - pos2[j, 1]
                rz = p2 - pos2[j, 2]
                d2 = rx * rx + ry * ry + rz * rz
                d = d2 ** 0.5
                if d < 2.220446049250313e-16:
                    d = 2.220446049250313e-16
                aj = area2[j]
                d3 = d * d * d
                G[i, j] = aj / d
                ndotr = nx * rx + ny * ry + nz * rz
                F[i, j] = -ndotr * aj / d3
                inv_d3_a = aj / d3
                Gp[i, 0, j] = -rx * inv_d3_a
                Gp[i, 1, j] = -ry * inv_d3_a
                Gp[i, 2, j] = -rz * inv_d3_a
        return G, F, Gp


def green_stat_assemble(pos1, pos2, nvec1, area2, same, want_gp):
    """
    Numba-accelerated assembly of G, F (and optionally Gp).

    Falls back to numpy broadcasting when numba is unavailable or disabled.
    Returns (G, F, Gp) where Gp is None when want_gp is False.
    """
    pos1 = np.ascontiguousarray(pos1, dtype = np.float64)
    pos2 = np.ascontiguousarray(pos2, dtype = np.float64)
    nvec1 = np.ascontiguousarray(nvec1, dtype = np.float64)
    area2 = np.ascontiguousarray(area2, dtype = np.float64)

    if numba_enabled():
        if want_gp:
            G, F, Gp = _green_stat_GF_Gp(pos1, pos2, nvec1, area2, same)
            return G, F, Gp
        G, F = _green_stat_GF(pos1, pos2, nvec1, area2, same)
        return G, F, None

    return _green_stat_assemble_numpy(pos1, pos2, nvec1, area2, same, want_gp)


def _green_stat_assemble_numpy(pos1, pos2, nvec1, area2, same, want_gp):
    """Reference numpy implementation (used when MNPBEM_NUMBA=0)."""
    r = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]
    d = np.linalg.norm(r, axis = 2)
    d_safe = np.maximum(d, np.finfo(float).eps)

    G = (1.0 / d_safe) * area2[np.newaxis, :]
    n_dot_r = np.sum(nvec1[:, np.newaxis, :] * r, axis = 2)
    F = -n_dot_r / (d_safe ** 3) * area2[np.newaxis, :]

    if same:
        n = min(G.shape[0], G.shape[1])
        idx = np.arange(n)
        G[idx, idx] = 0.0
        F[idx, idx] = 0.0

    Gp = None
    if want_gp:
        Gp = -r / (d_safe[:, :, np.newaxis] ** 3) * area2[np.newaxis, :, np.newaxis]
        Gp = np.transpose(Gp, (0, 2, 1))
        if same:
            n = min(Gp.shape[0], Gp.shape[2])
            idx = np.arange(n)
            Gp[idx, :, idx] = 0.0

    return G, F, Gp
