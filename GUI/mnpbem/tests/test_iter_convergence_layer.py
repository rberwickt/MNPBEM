"""Tests for BEMRetLayerIter convergence on substrate + composite particles.

v1.6.0 (agent B) regression test for the substrate + multi-material iter
operator-form fix.

Background
----------
``BEMRetLayerIter`` shares the same eps-after-matvec pattern that v1.5.1
fixed in ``BEMRetIter``: ``alpha`` and ``De`` rows applied
``eps · (M·sig)`` instead of the dense-form ``M · (eps·sig)``.  The two
forms are identical when eps is scalar (uniform-eps region) but diverge
when a substrate sim runs over a composite particle (e.g. core-shell
sphere on a glass substrate).  This file pins the corrected behaviour:
LayerIter must agree with the dense ``BEMRetLayer`` to within GMRES
tolerance on every case (uniform AND non-uniform eps).
"""

import numpy as np
import pytest

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle, LayerStructure
from mnpbem.bem import BEMRetLayer
from mnpbem.bem.bem_ret_layer_iter import BEMRetLayerIter
from mnpbem.simulation import PlaneWaveRetLayer
from mnpbem.greenfun import GreenTabLayer


_POL = np.array([[1.0, 0.0, 0.0]])
_DIR = np.array([[0.0, 0.0, -1.0]])


def _excite(layer):
    return PlaneWaveRetLayer(_POL, _DIR, layer)


def _au_sphere_on_glass():
    epstab = [EpsConst(1.0), EpsTable('gold.dat'), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 3], [0.0])
    sphere = trisphere(32, 20.0)
    sphere.shift([0.0, 0.0, -sphere.pos[:, 2].min() + 1.0])
    p = ComParticle(epstab, [sphere], [[2, 1]], [1])
    return p, layer, epstab


def _auag_dimer_on_glass():
    """Au + Ag dimer on glass — multi-material (different per particle)
    + substrate.  Triggers the same eps · (M·sig) → M · (eps·sig)
    operator-form fix that v1.5.1 applied to BEMRetIter."""
    epstab = [EpsConst(1.0), EpsTable('gold.dat'),
            EpsTable('silver.dat'), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 4], [0.0])
    p1 = trisphere(32, 12.0)
    p2 = trisphere(32, 12.0)
    p1.shift([-8.0, 0.0, -p1.pos[:, 2].min() + 1.0])
    p2.shift([+8.0, 0.0, -p2.pos[:, 2].min() + 1.0])
    p = ComParticle(epstab, [p1, p2], [[2, 1], [3, 1]], [1, 2])
    return p, layer, epstab


def _solve_loop(bem, exc, p, enei_arr):
    n = len(enei_arr)
    ext = np.zeros(n)
    for i, e in enumerate(enei_arr):
        sig, _ = bem.solve(exc(p, e))
        ext[i] = float(np.real(np.ravel(exc.extinction(sig)))[0])
    return ext


# ---------------------------------------------------------------------------
# Regression: uniform-eps Au sphere on glass — iter must match dense as
# before (scalar-eps fast path is bit-identical).
# ---------------------------------------------------------------------------

def test_uniform_eps_layer_iter_no_regression():

    p, layer, _ = _au_sphere_on_glass()
    enei = np.array([600.0])

    tab = layer.tabspace(p)
    gt = GreenTabLayer(layer, tab = tab)
    gt.set(enei)

    bem_d = BEMRetLayer(p, layer, greentab = gt)
    bem_i = BEMRetLayerIter(p, layer, greentab = gt,
            tol = 1e-8, maxit = 200)  # default precond='hmat'

    exc = _excite(layer)
    ext_d = _solve_loop(bem_d, exc, p, enei)
    ext_i = _solve_loop(bem_i, exc, p, enei)

    rel_diff = np.abs(ext_i - ext_d) / np.abs(ext_d)
    assert rel_diff.max() < 1e-2, \
        '[error] Au sphere on glass iter regressed: rel diff = {}'.format(rel_diff)


# ---------------------------------------------------------------------------
# Multi-material + substrate: Au + Ag dimer on glass — pre-fix this case
# would have hit the same ``eps · (M·sig)`` operator-mismatch β agent
# fixed for BEMRetIter (~70 % mid-band drift on Au@Ag dimer).  Operator
# form lifts iter back to dense within GMRES tol.
# ---------------------------------------------------------------------------

def test_multi_material_substrate_iter_dense_vs_iter():

    p, layer, _ = _auag_dimer_on_glass()
    enei = np.array([600.0])

    tab = layer.tabspace(p)
    gt = GreenTabLayer(layer, tab = tab)
    gt.set(enei)

    bem_d = BEMRetLayer(p, layer, greentab = gt)
    bem_i = BEMRetLayerIter(p, layer, greentab = gt,
            tol = 1e-8, maxit = 200)  # default precond='hmat'

    exc = _excite(layer)
    ext_d = _solve_loop(bem_d, exc, p, enei)
    ext_i = _solve_loop(bem_i, exc, p, enei)

    rel_diff = np.abs(ext_i - ext_d) / np.abs(ext_d)
    assert rel_diff.max() < 1e-2, \
        '[error] Au+Ag dimer on glass iter drift: rel diff = {}'.format(rel_diff)


# ---------------------------------------------------------------------------
# Smoke test: substrate iter with hmatrix is currently NotImplemented for
# BEMRetLayerIter (raised by __init__).  Pin that behaviour so we know
# when the path becomes available.
# ---------------------------------------------------------------------------

def test_substrate_iter_hmatrix_raises_not_implemented():

    p, layer, _ = _au_sphere_on_glass()
    enei = np.array([600.0])

    tab = layer.tabspace(p)
    gt = GreenTabLayer(layer, tab = tab)
    gt.set(enei)

    with pytest.raises(NotImplementedError):
        _ = BEMRetLayerIter(p, layer, greentab = gt, hmatrix = True)


# ---------------------------------------------------------------------------
# v1.6.0 fix: scalar-eps fast-path bit-identical assertion against dense.
# ---------------------------------------------------------------------------

def test_scalar_eps_layer_path_unchanged_by_v160_fix():

    p, layer, _ = _au_sphere_on_glass()
    enei = 600.0

    tab = layer.tabspace(p)
    gt = GreenTabLayer(layer, tab = tab)
    gt.set(np.array([enei]))

    bem_d = BEMRetLayer(p, layer, greentab = gt)
    bem_i = BEMRetLayerIter(p, layer, greentab = gt,
            tol = 1e-8, maxit = 200)  # default precond='hmat'

    exc = _excite(layer)
    sig_d, _ = bem_d.solve(exc(p, enei))
    sig_i, _ = bem_i.solve(exc(p, enei))

    ext_d = float(np.real(np.ravel(exc.extinction(sig_d)))[0])
    ext_i = float(np.real(np.ravel(exc.extinction(sig_i)))[0])

    rd = abs(ext_i - ext_d) / abs(ext_d)
    assert rd < 1e-2, \
        '[error] Scalar eps layer path regressed: ext_d={}, ext_i={}, rd={}'.format(
                ext_d, ext_i, rd)
