"""Tests for compgreen_ret_layer multi-particle (core-shell) on substrate.

v1.6.1 regression test for the _assembly shape-mismatch fix.

Background
----------
ComParticle with multi-material per particle (e.g. Au@Ag core-shell) on a
LayerStructure (substrate) triggered a (116, 116) vs (232, 232) shape
mismatch in `compgreen_ret_layer.CompGreenRetLayer._assembly`.  The
`_init_layer_indices` filter selects layer-touching faces (shell only),
giving ind1/ind2 of size 116, while `GreenRetLayer.G_comp` is built on
the full ComParticle and is sized (n_total=232, n_total=232).  The fix
restricts gr_val to the [ind1, ind2] sub-block.

This file pins:
1. End-to-end BEMRetLayer.solve completes for Au@Ag core-shell on glass.
2. Extinction is finite, real-positive.
3. Single-particle uniform-eps regression (no behavior change).
"""

import os
import sys

from typing import Any, Tuple

import numpy as np
import pytest

from GUI.mnpbem.materials import EpsConst, EpsTable
from GUI.mnpbem.geometry import trisphere, ComParticle, LayerStructure
from GUI.mnpbem.bem import BEMRetLayer
from GUI.mnpbem.simulation import PlaneWaveRetLayer
from GUI.mnpbem.greenfun import GreenTabLayer


_POL = np.array([[1.0, 0.0, 0.0]])
_DIR = np.array([[0.0, 0.0, -1.0]])


def _excite(layer: Any) -> PlaneWaveRetLayer:
    return PlaneWaveRetLayer(_POL, _DIR, layer)


def _au_sphere_on_glass() -> Tuple[Any, Any, list]:
    epstab = [EpsConst(1.0), EpsTable('gold.dat'), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 3], [0.0])
    sphere = trisphere(60, 12.0)
    sphere.shift([0.0, 0.0, -sphere.pos[:, 2].min() + 1.0])
    p = ComParticle(epstab, [sphere], [[2, 1]], [1])
    return p, layer, epstab


def _au_ag_core_shell_on_glass() -> Tuple[Any, Any, list]:
    epstab = [EpsConst(1.0),
            EpsTable('gold.dat'),
            EpsTable('silver.dat'),
            EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 4], [0.0])

    p_shell = trisphere(60, 12.0)
    p_core = trisphere(60, 8.0)

    p_shell.shift([0.0, 0.0, -p_shell.pos[:, 2].min() + 1.0])

    shell_z_mid = (p_shell.pos[:, 2].min() + p_shell.pos[:, 2].max()) / 2
    core_z_mid = (p_core.pos[:, 2].min() + p_core.pos[:, 2].max()) / 2
    p_core.shift([0.0, 0.0, shell_z_mid - core_z_mid])

    p = ComParticle(epstab, [p_shell, p_core], [[3, 1], [2, 3]], [1, 2])
    return p, layer, epstab


def _solve_extinction(p, layer, enei: float) -> float:
    enei_arr = np.array([enei])
    tab = layer.tabspace(p)
    gt = GreenTabLayer(layer, tab = tab)
    gt.set(enei_arr)

    bem = BEMRetLayer(p, layer, greentab = gt)
    exc = _excite(layer)

    sig, _ = bem.solve(exc(p, enei))
    ext = float(np.real(np.ravel(exc.extinction(sig)))[0])
    return ext


def test_core_shell_on_substrate_no_shape_mismatch():
    """Au@Ag core-shell on glass — pre-fix raised ValueError shape mismatch."""

    p, layer, _ = _au_ag_core_shell_on_glass()

    # Sanity check that ind1/ind2 are < total faces (the trigger condition)
    from GUI.mnpbem.greenfun.compgreen_ret_layer import CompGreenRetLayer
    cg = CompGreenRetLayer(p, p, layer)
    assert len(cg.ind1) < p.n, \
        '[error] Test setup invalid: ind1 should be a strict subset (got {} of {})'.format(
                len(cg.ind1), p.n)
    assert len(cg.ind2) < p.n, \
        '[error] Test setup invalid: ind2 should be a strict subset (got {} of {})'.format(
                len(cg.ind2), p.n)

    ext = _solve_extinction(p, layer, 550.0)
    assert np.isfinite(ext), \
        '[error] Au@Ag core-shell extinction not finite: {}'.format(ext)
    assert ext > 0.0, \
        '[error] Au@Ag core-shell extinction not positive: {}'.format(ext)


def test_core_shell_on_substrate_multi_wavelength():
    """Smoke at three wavelengths spanning resonances."""

    p, layer, _ = _au_ag_core_shell_on_glass()
    for enei in (450.0, 550.0, 700.0):
        ext = _solve_extinction(p, layer, enei)
        assert np.isfinite(ext) and ext > 0.0, \
            '[error] core-shell ext invalid at enei={}: {}'.format(enei, ext)


def test_single_au_sphere_on_substrate_unchanged():
    """Au sphere on glass (uniform eps, ind1 = range(n)) — fix is a no-op."""

    p, layer, _ = _au_sphere_on_glass()

    from GUI.mnpbem.greenfun.compgreen_ret_layer import CompGreenRetLayer
    cg = CompGreenRetLayer(p, p, layer)
    assert len(cg.ind1) == p.n, \
        '[error] Au sphere on glass should have ind1 = full range (got {} of {})'.format(
                len(cg.ind1), p.n)

    ext = _solve_extinction(p, layer, 600.0)
    assert np.isfinite(ext) and ext > 0.0, \
        '[error] Au sphere ext invalid: {}'.format(ext)


def test_assembly_subblock_consistency():
    """For uniform-eps single particle, gr_val sub-block == full block."""

    p, layer, _ = _au_sphere_on_glass()

    from GUI.mnpbem.greenfun.compgreen_ret_layer import CompGreenRetLayer
    cg = CompGreenRetLayer(p, p, layer)
    cg.gr.eval_components(600.0)

    # All faces touch the layer in this case
    assert len(cg.ind1) == p.n
    assert len(cg.ind2) == p.n

    for name in cg.gr.G_comp:
        gr_full = cg.gr.G_comp[name]
        gr_sub = gr_full[np.ix_(cg.ind1, cg.ind2)]
        assert np.allclose(gr_full, gr_sub), \
            '[error] sub-block mismatch for component <{}>'.format(name)
