"""
Verify that the closed-surface diagonal correction path is bit-identical
between the original (slow O(n^2) ``np.allclose`` matcher) and the
vectorised loc-matching introduced for M4 N6.

Also exercises the multi-particle ComParticle path to make sure the
``_handle_closed_surfaces`` accumulation produces the same F matrix as
before within 1e-12.
"""

import os
import importlib
import numpy as np
import pytest


def _build(env_value, n_faces, multi = False):
    if env_value is None:
        os.environ.pop('MNPBEM_NUMBA', None)
    else:
        os.environ['MNPBEM_NUMBA'] = env_value

    from mnpbem.greenfun import _numba_kernels
    importlib.reload(_numba_kernels)
    from mnpbem.greenfun import compgreen_stat as _cgs_mod
    importlib.reload(_cgs_mod)

    from mnpbem import trisphere, EpsConst, ComParticle

    eps = [EpsConst(1.0), EpsConst(4.0)]

    if multi:
        # two spheres -> ComParticle with two particles, each closed by index
        p_a = trisphere(n_faces, 5.0)
        p_b = trisphere(n_faces, 8.0)
        # shift second sphere so they don't overlap
        p_b.pos = p_b.pos + np.array([20.0, 0.0, 0.0])
        p_b.verts = p_b.verts + np.array([20.0, 0.0, 0.0])
        if hasattr(p_b, 'verts2') and p_b.verts2 is not None:
            p_b.verts2 = p_b.verts2 + np.array([20.0, 0.0, 0.0])
        cp = ComParticle(eps, [p_a, p_b], [[2, 1], [2, 1]], 1, 2)
    else:
        p = trisphere(n_faces, 10.0)
        cp = ComParticle(eps, [p], [[2, 1]], 1)

    return _cgs_mod.CompGreenStat(cp, cp)


@pytest.mark.parametrize("n_faces", [144, 256])
def test_closed_surface_F_identical(n_faces):
    """The closed-surface F matrix must match the reference within 1e-12."""
    g_slow = _build('0', n_faces)
    g_fast = _build('1', n_faces)

    diff_F = np.max(np.abs(g_fast.F - g_slow.F))
    diff_G = np.max(np.abs(g_fast.G - g_slow.G))

    assert diff_F < 1e-12, f"max|F_fast - F_slow| = {diff_F}"
    assert diff_G < 1e-12, f"max|G_fast - G_slow| = {diff_G}"


def test_closed_surface_multi_particle():
    """Multi-particle ComParticle: closed-surface accumulation also identical."""
    g_slow = _build('0', 144, multi = True)
    g_fast = _build('1', 144, multi = True)

    diff_F = np.max(np.abs(g_fast.F - g_slow.F))
    diff_G = np.max(np.abs(g_fast.G - g_slow.G))

    assert diff_F < 1e-12, f"multi: max|F_fast - F_slow| = {diff_F}"
    assert diff_G < 1e-12, f"multi: max|G_fast - G_slow| = {diff_G}"


def test_closedparticle_loc_matches_identity():
    """For ``closed = [1]`` over a single particle, ``loc`` must be identity."""
    from mnpbem import trisphere, EpsConst, ComParticle

    p = trisphere(144, 10.0)
    cp = ComParticle([EpsConst(1.0), EpsConst(2.0)], [p], [[2, 1]], 1)
    full, dir_val, loc = cp.closedparticle(1)
    assert full is not None
    assert dir_val == 1
    assert loc is not None
    assert np.array_equal(loc, np.arange(len(loc)))


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
