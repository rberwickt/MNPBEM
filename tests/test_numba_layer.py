"""Numba on/off regression tests for layer Green-function interpolation.

Verifies:
  * Trilinear / bilinear numba kernel matches RGI to 1e-12.
  * Stat layer Green function (substrate) matches under numba on/off.
  * Ret layer Green function (substrate, complex k) matches.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _set_numba(flag: bool):
    if flag:
        os.environ['MNPBEM_NUMBA'] = '1'
    else:
        os.environ.pop('MNPBEM_NUMBA', None)


def test_trilinear_complex_matches_rgi():
    from mnpbem.greenfun._numba_layer import trilinear_complex, _HAS_NUMBA
    if not _HAS_NUMBA:
        pytest.skip('numba not installed')

    rng = np.random.default_rng(42)
    nr, nz1, nz2 = 8, 6, 5
    r = np.linspace(0, 1, nr)
    z1 = np.linspace(-1, 1, nz1)
    z2 = np.linspace(0, 2, nz2)
    data = rng.standard_normal((nr, nz1, nz2)) + 1j * rng.standard_normal((nr, nz1, nz2))

    # Inside grid + a few extrapolation points
    n_pts = 200
    points = rng.uniform(-0.2, 1.2, size=(n_pts, 3))
    points[:, 1] = points[:, 1] * 2 - 1
    points[:, 2] = points[:, 2] * 2

    _set_numba(False)
    val_rgi = trilinear_complex((r, z1, z2), data, points)
    _set_numba(True)
    val_nb = trilinear_complex((r, z1, z2), data, points)
    _set_numba(False)

    assert np.allclose(val_rgi, val_nb, rtol=1e-12, atol=1e-12), \
        'trilinear numba/RGI mismatch: max |Δ| = {:e}'.format(
            np.max(np.abs(val_rgi - val_nb)))


def test_bilinear_complex_matches_rgi():
    from mnpbem.greenfun._numba_layer import trilinear_complex, _HAS_NUMBA
    if not _HAS_NUMBA:
        pytest.skip('numba not installed')

    rng = np.random.default_rng(7)
    nr, nz = 7, 5
    r = np.linspace(0, 1, nr)
    z = np.linspace(-1, 1, nz)
    data = rng.standard_normal((nr, nz)) + 1j * rng.standard_normal((nr, nz))

    points = rng.uniform(-0.2, 1.2, size=(150, 2))
    points[:, 1] = points[:, 1] * 2 - 1

    _set_numba(False)
    val_rgi = trilinear_complex((r, z), data, points)
    _set_numba(True)
    val_nb = trilinear_complex((r, z), data, points)
    _set_numba(False)

    assert np.allclose(val_rgi, val_nb, rtol=1e-12, atol=1e-12)


def _build_stat_layer_G(flag):
    from mnpbem.materials import EpsConst
    from mnpbem.geometry import trisphere, ComParticle, LayerStructure
    from mnpbem.greenfun import CompGreenStatLayer

    epstab = [EpsConst(1.0), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 2], [0.0])
    sphere = trisphere(144, 5.0)
    sphere.shift([0, 0, -sphere.pos[:, 2].min() + 1.0])
    p = ComParticle(epstab, [sphere], [[2, 1]], [1])

    _set_numba(flag)
    g = CompGreenStatLayer(p, p, layer)
    G = g.eval(500.0, 'G')
    _set_numba(False)
    return G


def test_layer_stat_numba_match():
    """Stat layer: numba on/off → same result to 1e-12."""
    G_off = _build_stat_layer_G(False)
    G_on = _build_stat_layer_G(True)
    assert np.all(np.isfinite(G_off))
    assert np.all(np.isfinite(G_on))
    diff = np.max(np.abs(G_off - G_on))
    assert diff < 1e-12, 'stat layer numba mismatch: {:e}'.format(diff)


def _build_ret_layer_Gout(flag):
    from mnpbem.materials import EpsConst, EpsTable
    from mnpbem.geometry import trisphere, ComParticle, LayerStructure
    from mnpbem.greenfun import CompGreenRetLayer, GreenTabLayer

    epstab = [EpsConst(1.0), EpsTable('gold.dat'), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 3], [0.0])
    sphere = trisphere(144, 20.0)
    sphere.shift([0, 0, -sphere.pos[:, 2].min() + 1.0])
    p = ComParticle(epstab, [sphere], [[2, 1]], [1])

    tab = layer.tabspace(p)
    gt = GreenTabLayer(layer, tab=tab)
    gt.set(np.array([500.0]))

    _set_numba(flag)
    g = CompGreenRetLayer(p, p, layer, greentab_obj=gt)
    # Outer surface (n_regions=2 -> i=1, j=1) exercises the tabulated
    # reflected Green function.
    G_out = g.eval(1, 1, 'G', 500.0)
    _set_numba(False)
    return G_out


def test_layer_ret_numba_match():
    """Ret layer with complex k: numba on/off → same result to 1e-12."""
    G_off = _build_ret_layer_Gout(False)
    G_on = _build_ret_layer_Gout(True)
    assert isinstance(G_off, dict)
    assert isinstance(G_on, dict)
    for name in G_off:
        assert np.all(np.isfinite(G_off[name]))
        assert np.all(np.isfinite(G_on[name]))
        diff = np.max(np.abs(G_off[name] - G_on[name]))
        assert diff < 1e-12, 'ret layer numba mismatch [{}]: {:e}'.format(name, diff)
