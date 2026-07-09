"""Regression tests for metal-substrate layer green path (Issue 3).

When a particle sits exactly on a metal substrate interface
(z_particle_bottom == z_layer), the legacy `_enlarge` clip used a
strict `<` comparison, leaving the enlarged z range with a value
matching the layer interface. The downstream `_zlinlogspace` then
evaluated log10(0) = -inf, which produced NaN tabulation grids and
finally an `IndexError: index out of bounds` inside
`_reflection_subs` when `indlayer` mapped the NaN to the lowermost
bin.

The fix uses `<=` / `>=` boundary checks in `_enlarge` so values
exactly on an interface are nudged inward by 1e-10. These tests
exercise the trigger cases (gold.dat substrate, particle on
interface, dipole below interface) and confirm tabulation /
green-function evaluation no longer raises.
"""

import os

import numpy as np
import pytest

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import LayerStructure, trisphere


def _build_layer(eps_substrate, z_interface = -21.0):
    eps_medium = EpsConst(1.0)
    eps_particle = EpsConst(2.0)
    epstab = [eps_medium, eps_particle, eps_substrate]
    layer = LayerStructure(epstab, [1, 3], [z_interface])
    return layer, epstab


def test_enlarge_clips_at_boundary():
    layer, _ = _build_layer(EpsConst(2.25))

    z_in = np.array([-21.0, 21.0])
    zlayer = np.array([-21.0, np.inf])
    out = layer._enlarge(z_in, zlayer, scale = 1.05, range_mode = None)

    assert out[0] > -21.0, 'z[0] must be strictly above the interface'
    assert out[0] - (-21.0) <= 1.0e-9
    assert np.isfinite(out).all()


def test_adjust_no_nan_on_interface():
    layer, _ = _build_layer(EpsConst(2.25))

    z1_in = np.array([-20.0, 20.0])
    z2_in = np.array([-20.0, 20.0])
    z1, z2 = layer._adjust(z1_in, z2_in, scale = 1.05)

    assert np.isfinite(z1).all(), 'z1 must not contain NaN/Inf'
    assert np.isfinite(z2).all(), 'z2 must not contain NaN/Inf'


def test_zlinlogspace_no_nan_on_interface():
    layer, _ = _build_layer(EpsConst(2.25))

    # Mimic the post-_adjust input that previously crashed (uppermost
    # case): z range starts exactly at the layer interface.
    z1_in = np.array([-20.0, 20.0])
    z2_in = np.array([-20.0, 20.0])
    z1, z2 = layer._adjust(z1_in, z2_in, scale = 1.05)

    grid = layer._zlinlogspace(float(z1[0]), float(z1[1]), 30, 'log')
    assert np.isfinite(grid).all(), 'zlinlogspace grid contains NaN/Inf'


def test_tabspace_metal_substrate_no_index_error():
    """Reproduce Issue 3 mini case: particle on metal substrate, tabspace
    + GreenTabLayer.set should not raise IndexError."""
    from mnpbem.greenfun import GreenTabLayer

    eps_gold = EpsTable('gold.dat')
    layer, _ = _build_layer(eps_gold, z_interface = -21.0)

    # Tiny particle sitting exactly on substrate (zmin = -21)
    p = trisphere(60, 4.0)
    p.shift([0.0, 0.0, -21.0 + 2.0])  # bottom near z = -21

    tab = layer.tabspace(p)

    gt = GreenTabLayer(layer, tab = tab)
    enei_tab = np.linspace(500.0, 700.0, 3)

    gt.set(enei_tab)

    # eval at a sample query that previously triggered the IndexError
    r_q = np.array([1.0, 5.0])
    z1_q = np.array([-20.5, -20.5])
    z2_q = np.array([-20.99, -20.99])
    G, Fr, Fz = gt.eval(600.0, r_q, z1_q, z2_q)

    assert np.isfinite(G).all()
    assert np.isfinite(Fr).all()
    assert np.isfinite(Fz).all()


def test_dielectric_substrate_unchanged():
    """The fix must not alter behavior on the well-tested glass case."""
    from mnpbem.greenfun import GreenTabLayer

    eps_glass = EpsConst(2.25)
    layer, _ = _build_layer(eps_glass, z_interface = 0.0)

    p = trisphere(60, 4.0)
    p.shift([0.0, 0.0, 3.0])

    tab = layer.tabspace(p)

    gt = GreenTabLayer(layer, tab = tab)
    gt.set(np.array([550.0]))

    r_q = np.array([1.0, 5.0])
    z1_q = np.array([2.5, 2.5])
    z2_q = np.array([0.01, 0.01])
    G, Fr, Fz = gt.eval(550.0, r_q, z1_q, z2_q)

    assert np.isfinite(G).all()
    assert np.isfinite(Fr).all()
    assert np.isfinite(Fz).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
