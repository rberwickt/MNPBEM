"""v1.7 A3 — BEMStat family GPU audit tests.

Covers GPU-path smoke + regression for:
  - BEMStat (quasistatic, dense LU)
  - BEMStatLayer (with substrate)
  - BEMStatIter (iterative GMRES)
  - BEMStatEig (eigenmode expansion)

Bugs guarded:
  Bug A: BEMStat.clear() stale enei → cache hit with mat_lu=None → crash.
  Bug B: BEMStatLayer.clear() left _A_lu / _rhs_scale pinned (GPU mem leak).
  Bug C: BEMStatIter.clear() stale enei / _lambda → cache hit with
         _mat_lu=None → crash inside _mfun.
  Bug D: BEMStatEig.clear() did not exist — API parity gap.

GPU is set to device 2 (CUDA_VISIBLE_DEVICES=2) at the runner level.
The tests are written so they pass on the CPU fallback as well: the
correctness checks (clear-then-solve, eigenvalue sorting) do not
require the GPU path to be live.  Routing is exercised when
MNPBEM_GPU=1 + cupy is importable.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from GUI.mnpbem.materials.eps_const import EpsConst
from GUI.mnpbem.geometry import trisphere, ComParticle
from GUI.mnpbem.geometry.layer_structure import LayerStructure
from GUI.mnpbem.greenfun import CompStruct
from GUI.mnpbem.bem.bem_stat import BEMStat
from GUI.mnpbem.bem.bem_stat_layer import BEMStatLayer
from GUI.mnpbem.bem.bem_stat_iter import BEMStatIter
from GUI.mnpbem.bem.bem_stat_eig import BEMStatEig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sphere(nverts = 144, diameter = 10.0):
    sphere = trisphere(nverts, diameter)
    eps = [EpsConst(1.0), EpsConst(-10.0 + 1.0j)]
    p = ComParticle(eps, [sphere], [[2, 1]])
    return p


def _make_dimer(nverts = 144, diameter = 5.0, gap = 12.0):
    s1 = trisphere(nverts, diameter)
    s1.verts[:, 0] -= gap / 2
    s1.pos[:, 0] -= gap / 2
    s2 = trisphere(nverts, diameter)
    s2.verts[:, 0] += gap / 2
    s2.pos[:, 0] += gap / 2
    eps = [EpsConst(1.0), EpsConst(-10.0 + 1.0j), EpsConst(-10.0 + 1.0j)]
    p = ComParticle(eps, [s1, s2], [[2, 1], [3, 1]])
    return p


def _make_particle_on_layer(nverts = 144, diameter = 5.0, z_offset = 10.0):
    eps_layer = [EpsConst(1.0), EpsConst(2.25)]
    layer = LayerStructure(eps_layer, [1, 2], [0.0])
    sphere = trisphere(nverts, diameter)
    sphere.verts[:, 2] += z_offset
    sphere.pos[:, 2] += z_offset
    eps_p = [EpsConst(1.0), EpsConst(-10.0 + 1.0j)]
    p = ComParticle(eps_p, [sphere], [[2, 1]])
    return p, layer


def _planewave_phip(p, pol = (1.0, 0.0, 0.0)):
    return -p.nvec @ np.asarray(pol, dtype = float)


# ---------------------------------------------------------------------------
# BEMStat — clear()-then-solve regression (Bug A)
# ---------------------------------------------------------------------------


class TestBEMStatClearRegression(object):

    def test_clear_resets_enei(self):
        p = _make_sphere()
        bem = BEMStat(p, enei = 500.0)
        assert bem.enei == 500.0
        bem.clear()
        # v1.7 A3: enei must be cleared so the cache gate forces a rebuild.
        assert bem.enei is None
        assert bem.mat_lu is None

    def test_clear_then_solve_same_wl(self):
        # Pre-fix: TypeError "cannot unpack non-iterable NoneType".
        p = _make_sphere()
        bem = BEMStat(p, enei = 500.0)
        bem.clear()
        exc = CompStruct(p, 500.0, phip = _planewave_phip(p))
        sig, _ = bem.solve(exc)
        assert sig.sig.shape == (p.n,)
        assert np.isfinite(sig.sig).all()
        assert np.max(np.abs(sig.sig)) > 0


# ---------------------------------------------------------------------------
# BEMStatLayer — clear() drops _A_lu / _rhs_scale (Bug B)
# ---------------------------------------------------------------------------


class TestBEMStatLayerClearRegression(object):

    def test_clear_drops_factor(self):
        p, layer = _make_particle_on_layer()
        bem = BEMStatLayer(p, layer, enei = 500.0)
        assert bem._A_lu is not None
        assert bem._rhs_scale is not None
        bem.clear()
        # v1.7 A3: _A_lu and _rhs_scale must be released to free GPU
        # LU memory.  Pre-fix they stayed pinned across wavelength sweeps.
        assert bem._A_lu is None
        assert bem._rhs_scale is None
        assert bem.enei is None

    def test_clear_then_solve_same_wl(self):
        p, layer = _make_particle_on_layer()
        bem = BEMStatLayer(p, layer, enei = 500.0)
        bem.clear()
        exc = CompStruct(p, 500.0, phip = _planewave_phip(p))
        sig, _ = bem.solve(exc)
        assert sig.sig.shape == (p.n,)
        assert np.isfinite(sig.sig).all()


# ---------------------------------------------------------------------------
# BEMStatIter — clear() resets enei + _lambda (Bug C)
# ---------------------------------------------------------------------------


class TestBEMStatIterClearRegression(object):

    def test_clear_resets_state(self):
        p = _make_sphere()
        bem = BEMStatIter(p, enei = 500.0, precond = 'hmat')
        assert bem.enei == 500.0
        assert bem._lambda is not None
        bem.clear()
        # v1.7 A3: cache gate and wavelength-dependent fields cleared.
        assert bem.enei is None
        assert bem._mat_lu is None
        assert bem._lambda is None
        assert bem._schur_active is False
        assert bem._schur_op is None
        assert bem._hlu_object is None

    def test_clear_then_solve_same_wl(self):
        # Pre-fix: TypeError "'NoneType' object is not subscriptable".
        p = _make_sphere()
        bem = BEMStatIter(p, enei = 500.0, precond = 'hmat',
                tol = 1e-6, maxit = 200)
        bem.clear()
        exc = CompStruct(p, 500.0, phip = _planewave_phip(p))
        sig, _ = bem.solve(exc)
        assert sig.sig.shape == (p.n,)
        assert np.isfinite(sig.sig).all()


# ---------------------------------------------------------------------------
# BEMStatEig — clear() method exists + behaves (Bug D)
# ---------------------------------------------------------------------------


class TestBEMStatEigClearRegression(object):

    def test_clear_exists_and_resets(self):
        p = _make_sphere()
        bem = BEMStatEig(p, nev = 10, enei = 500.0)
        assert bem.enei == 500.0
        assert bem.mat is not None
        # Pre-fix: AttributeError.  v1.7 A3 added the method for parity.
        bem.clear()
        assert bem.enei is None
        assert bem.mat is None

    def test_clear_then_solve(self):
        p = _make_sphere()
        bem = BEMStatEig(p, nev = 10, enei = 500.0)
        bem.clear()
        exc = CompStruct(p, 500.0, phip = _planewave_phip(p))
        sig, _ = bem.solve(exc)
        assert sig.sig.shape == (p.n,)
        assert np.isfinite(sig.sig).all()


# ---------------------------------------------------------------------------
# Smoke matrix — all four solvers reach a sensible solution
# ---------------------------------------------------------------------------


class TestStatSmokeMatrix(object):

    def test_bemstat_sphere_planewave(self):
        p = _make_sphere(nverts = 144)
        bem = BEMStat(p)
        exc = CompStruct(p, 500.0, phip = _planewave_phip(p))
        sig, _ = bem.solve(exc)
        assert np.max(np.abs(sig.sig)) > 0
        assert np.isfinite(sig.sig).all()

    def test_bemstat_layer_planewave(self):
        p, layer = _make_particle_on_layer(nverts = 144)
        bem = BEMStatLayer(p, layer)
        exc = CompStruct(p, 500.0, phip = _planewave_phip(p))
        sig, _ = bem.solve(exc)
        assert np.max(np.abs(sig.sig)) > 0
        assert np.isfinite(sig.sig).all()

    def test_bemstat_iter_dimer(self):
        p = _make_dimer(nverts = 144)
        bem = BEMStatIter(p, tol = 1e-6, maxit = 200, precond = None)
        exc = CompStruct(p, 500.0, phip = _planewave_phip(p))
        sig, _ = bem.solve(exc)
        assert np.max(np.abs(sig.sig)) > 0
        assert np.isfinite(sig.sig).all()

    def test_bemstat_eig_eigenvalues_sorted(self):
        p = _make_sphere(nverts = 144)
        bem = BEMStatEig(p, nev = 20)
        ene_diag = np.diag(bem.ene)
        # plasmonmode sorts by ascending real part — guard against
        # regressions that drop the sort.
        diffs = np.diff(np.real(ene_diag))
        assert np.all(diffs >= -1e-10), '[regression] eigenvalues not sorted'


# ---------------------------------------------------------------------------
# Iter vs dense cross-check (small mesh)
# ---------------------------------------------------------------------------


class TestBEMStatIterMatchesDense(object):

    def test_dense_vs_iter_same_solution(self):
        p = _make_sphere(nverts = 144)
        bem_dense = BEMStat(p)
        bem_iter = BEMStatIter(p, tol = 1e-8, maxit = 400, precond = None)
        exc = CompStruct(p, 500.0, phip = _planewave_phip(p))
        sig_d, _ = bem_dense.solve(exc)
        sig_i, _ = bem_iter.solve(exc)
        rel = (np.linalg.norm(sig_d.sig - sig_i.sig)
                / np.linalg.norm(sig_d.sig))
        assert rel < 1e-5, '[error] iter vs dense mismatch: rel={:.3e}'.format(rel)
