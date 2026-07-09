"""
Verify that the numba-accelerated quasistatic Green function assembly
produces matrices that are bit-for-bit equivalent (within 1e-12) to the
numpy broadcasting fallback.
"""

import os
import importlib
import numpy as np
import pytest


def _build_compgreen(env_value, n_faces):
    if env_value is None:
        os.environ.pop('MNPBEM_NUMBA', None)
    else:
        os.environ['MNPBEM_NUMBA'] = env_value

    # Force the kernels module (and thus numba_enabled cache state) to
    # re-evaluate the env var on every call.
    from mnpbem.greenfun import _numba_kernels
    importlib.reload(_numba_kernels)
    from mnpbem.greenfun import compgreen_stat as _cgs_mod
    importlib.reload(_cgs_mod)

    from mnpbem import trisphere, EpsConst, ComParticle

    eps = [EpsConst(1.0), EpsConst(2.0)]
    p = trisphere(n_faces, 10.0)
    cp = ComParticle(eps, [p], [[2, 1]])
    g = _cgs_mod.CompGreenStat(cp, cp)
    return g


@pytest.mark.parametrize("n_faces", [144, 484])
def test_numba_matches_numpy_G_F(n_faces):
    g_slow = _build_compgreen('0', n_faces)
    g_fast = _build_compgreen('1', n_faces)

    G_slow, F_slow = g_slow.G, g_slow.F
    G_fast, F_fast = g_fast.G, g_fast.F

    assert G_slow.shape == G_fast.shape
    assert F_slow.shape == F_fast.shape

    diff_G = np.max(np.abs(G_fast - G_slow))
    diff_F = np.max(np.abs(F_fast - F_slow))

    assert diff_G < 1e-12, "max|G_numba - G_slow| = {}".format(diff_G)
    assert diff_F < 1e-12, "max|F_numba - F_slow| = {}".format(diff_F)


def test_numba_matches_numpy_Gp_cart():
    """Also verify Gp matches when deriv == 'cart'."""
    from mnpbem import trisphere, EpsConst, ComParticle
    from mnpbem.greenfun import _numba_kernels
    from mnpbem.greenfun import compgreen_stat as _cgs_mod

    n_faces = 144

    os.environ['MNPBEM_NUMBA'] = '0'
    importlib.reload(_numba_kernels)
    importlib.reload(_cgs_mod)
    eps = [EpsConst(1.0), EpsConst(2.0)]
    p = trisphere(n_faces, 10.0)
    cp = ComParticle(eps, [p], [[2, 1]])
    g_slow = _cgs_mod.CompGreenStat(cp, cp, deriv = 'cart')
    Gp_slow = g_slow._Gp_raw.copy()

    os.environ['MNPBEM_NUMBA'] = '1'
    importlib.reload(_numba_kernels)
    importlib.reload(_cgs_mod)
    eps = [EpsConst(1.0), EpsConst(2.0)]
    p = trisphere(n_faces, 10.0)
    cp = ComParticle(eps, [p], [[2, 1]])
    g_fast = _cgs_mod.CompGreenStat(cp, cp, deriv = 'cart')
    Gp_fast = g_fast._Gp_raw.copy()

    diff = np.max(np.abs(Gp_fast - Gp_slow))
    assert diff < 1e-12, "max|Gp_numba - Gp_slow| = {}".format(diff)
