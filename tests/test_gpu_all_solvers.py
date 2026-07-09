"""Verify GPU dispatch is wired into all BEM solvers + eig + mirror + iter.

Without a CUDA device the GPU path falls through to the CPU branch, so the
checks here focus on:

- import surface (every patched solver still loads cleanly)
- dispatch helpers exist for LU, dense solve, eigh and matmul
- end-to-end solve results are unchanged after the dispatch refactor (numerical
  equivalence to the direct solver on the small sphere fixture used by the
  iter tests)
- the dispatch package format is correctly threaded through ``[1].shape`` /
  ``[2]`` indexing in the layer iterative preconditioner
"""

import os
import sys

import numpy as np
import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mnpbem.materials import EpsConst, EpsDrude
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.bem import (
    BEMStat, BEMRet, BEMStatLayer, BEMRetLayer,
    BEMStatMirror, BEMRetMirror, BEMStatEig, BEMStatEigMirror,
)
from mnpbem.bem.bem_stat_iter import BEMStatIter
from mnpbem.bem.bem_ret_iter import BEMRetIter
from mnpbem.bem.bem_ret_layer_iter import BEMRetLayerIter
from mnpbem.simulation import PlaneWaveStat, PlaneWaveRet
from mnpbem.utils import gpu as gmod


def _make_sphere(nfaces=144, radius=10.0):
    epsm = EpsConst(1.0)
    epsAu = EpsDrude(eps0=10.0, wp=9.065, gammad=0.0708, name='gold')
    p = trisphere(nfaces, radius)
    return ComParticle([epsm, epsAu], [p], [[1, 2]], 1)


@pytest.fixture
def sphere():
    return _make_sphere()


class TestDispatchSurface(object):

    def test_helpers_exist(self):
        assert hasattr(gmod, 'lu_factor_dispatch')
        assert hasattr(gmod, 'lu_solve_dispatch')
        assert hasattr(gmod, 'solve_dispatch')
        assert hasattr(gmod, 'eigh_dispatch')
        assert hasattr(gmod, 'matmul_dispatch')

    def test_solve_dispatch_matches_scipy(self):
        rng = np.random.default_rng(0)
        N = 64
        A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        b = rng.standard_normal((N, 3)) + 1j * rng.standard_normal((N, 3))
        x = gmod.solve_dispatch(A, b)
        from scipy.linalg import solve as _scipy_solve
        ref = _scipy_solve(A, b, check_finite=False)
        assert np.allclose(x, ref, atol=1e-12, rtol=1e-12)

    def test_eigh_dispatch_matches_scipy(self):
        rng = np.random.default_rng(0)
        N = 32
        A = rng.standard_normal((N, N))
        A = 0.5 * (A + A.T)
        w, v = gmod.eigh_dispatch(A)
        from scipy.linalg import eigh as _scipy_eigh
        w_ref, v_ref = _scipy_eigh(A, check_finite=False)
        assert np.allclose(w, w_ref, atol=1e-12, rtol=1e-12)

    def test_matmul_dispatch_matches_numpy(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((50, 30))
        B = rng.standard_normal((30, 20))
        C = gmod.matmul_dispatch(A, B)
        assert np.allclose(C, A @ B, atol=1e-12, rtol=1e-12)


class TestSolverDispatchWired(object):
    """Each patched solver must import the dispatch helpers from utils.gpu."""

    @pytest.mark.parametrize('module_name', [
        'mnpbem.bem.bem_stat_iter',
        'mnpbem.bem.bem_ret_iter',
        'mnpbem.bem.bem_ret_layer_iter',
        'mnpbem.bem.bem_stat_mirror',
        'mnpbem.bem.bem_ret_mirror',
        'mnpbem.bem.bem_stat_eig',
        'mnpbem.bem.bem_stat_eig_mirror',
    ])
    def test_import(self, module_name):
        import importlib
        mod = importlib.import_module(module_name)
        # the dispatch helpers must be referenced from the module source
        src = open(mod.__file__).read()
        assert ('lu_factor_dispatch' in src
                or 'solve_dispatch' in src
                or 'matmul_dispatch' in src), \
            f'{module_name} does not reference any dispatch helper'


class TestEndToEndResultsUnchanged(object):
    """Iterative solver still matches the direct solver after dispatch refactor."""

    def test_stat_iter_matches_direct(self, sphere):
        bem_d = BEMStat(sphere)
        bem_i = BEMStatIter(sphere, solver='gmres', tol=1e-8, maxit=300,
                            precond='hmat')
        pw = PlaneWaveStat(np.array([[0.0, 0.0, 1.0]]))
        exc = pw(sphere, 550.0)
        sig_d, _ = bem_d.solve(exc)
        sig_i, _ = bem_i.solve(exc)
        rel = np.abs(sig_d.sig - sig_i.sig).max() / np.abs(sig_d.sig).max()
        assert rel < 1e-8

    def test_ret_iter_matches_direct(self, sphere):
        bem_d = BEMRet(sphere)
        bem_i = BEMRetIter(sphere, solver='gmres', tol=1e-8, maxit=300,
                           precond='hmat')
        pw = PlaneWaveRet(np.array([[0.0, 0.0, 1.0]]),
                          np.array([[1.0, 0.0, 0.0]]))
        exc = pw(sphere, 550.0)
        sig_d, _ = bem_d.solve(exc)
        sig_i, _ = bem_i.solve(exc)
        for fname in ('sig1', 'sig2'):
            a = sig_d.get(fname); b = sig_i.get(fname)
            rel = np.abs(a - b).max() / np.abs(a).max()
            assert rel < 1e-8

    def test_stat_eig_matches_direct(self, sphere):
        bem_d = BEMStat(sphere)
        bem_e = BEMStatEig(sphere, nev=40)
        pw = PlaneWaveStat(np.array([[0.0, 0.0, 1.0]]))
        exc = pw(sphere, 550.0)
        sig_d, _ = bem_d.solve(exc)
        sig_e, _ = bem_e.solve(exc)
        # eigenmode expansion truncates at nev modes -> looser tolerance
        rel = np.abs(sig_d.sig - sig_e.sig).max() / np.abs(sig_d.sig).max()
        assert rel < 5e-2


class TestLayerIterTuplePackFormat(object):
    """Regression: bem_ret_layer_iter relies on lu_piv[1].shape[0] (was [0])."""

    def test_lu_pkg_indexing(self):
        rng = np.random.default_rng(0)
        N = 16
        A = rng.standard_normal((N, N)) + np.eye(N)
        pkg = gmod.lu_factor_dispatch(A)
        assert pkg[0] in ('cpu', 'gpu')
        assert pkg[1].shape == (N, N)
