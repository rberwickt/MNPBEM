"""End-to-end integration tests for BEMIter iterative solver.

Verifies that BEMStatIter / BEMRetIter produce results matching the direct
BEMStat / BEMRet solvers on a small Au sphere, within tight tolerances.
"""
import os
import sys
import time

import numpy as np
import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mnpbem.materials import EpsConst, EpsDrude
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.bem import BEMStat, BEMStatIter, BEMRet, BEMRetIter, BEMIter
from mnpbem.simulation import PlaneWaveStat, PlaneWaveRet


def _make_sphere(nfaces=144, radius=10.0):
    epsm = EpsConst(1.0)
    epsAu = EpsDrude(eps0=10.0, wp=9.065, gammad=0.0708, name='gold')
    p = trisphere(nfaces, radius)
    return ComParticle([epsm, epsAu], [p], [[1, 2]], 1)


@pytest.fixture
def sphere():
    return _make_sphere()


class TestBEMIterBase(object):

    def test_default_solver_is_gmres(self):
        it = BEMIter()
        assert it.solver == 'gmres'
        assert it.precond == 'hmat'

    def test_options_returns_dict(self):
        op = BEMIter.options()
        assert op['solver'] == 'gmres'
        assert 'kmax' in op and 'htol' in op

    def test_solver_map_has_three(self):
        assert set(BEMIter.SOLVER_MAP.keys()) == {'gmres', 'cgs', 'bicgstab'}


class TestBEMStatIterEndToEnd(object):

    def test_matches_direct_solver(self, sphere):
        bem_d = BEMStat(sphere)
        bem_i = BEMStatIter(sphere, solver='gmres', tol=1e-8, maxit=300,
                            precond='hmat')
        pw = PlaneWaveStat(np.array([[0.0, 0.0, 1.0]]))
        exc = pw(sphere, 550.0)
        sig_d, _ = bem_d.solve(exc)
        sig_i, _ = bem_i.solve(exc)
        denom = np.abs(sig_d.sig).max()
        rel = np.abs(sig_d.sig - sig_i.sig).max() / denom
        assert rel < 1e-8

    def test_info_records_statistics(self, sphere):
        bem_i = BEMStatIter(sphere, solver='gmres', tol=1e-6, maxit=200,
                            precond='hmat')
        pw = PlaneWaveStat(np.array([[0.0, 0.0, 1.0]]))
        exc = pw(sphere, 500.0)
        bem_i.solve(exc)
        flag, relres, it = bem_i.info()
        assert len(flag) == 1
        assert flag[0] == 0
        assert relres[0] < 1e-5

    def test_bicgstab_also_converges(self, sphere):
        bem_d = BEMStat(sphere)
        bem_i = BEMStatIter(sphere, solver='bicgstab', tol=1e-8, maxit=300,
                            precond='hmat')
        pw = PlaneWaveStat(np.array([[0.0, 0.0, 1.0]]))
        exc = pw(sphere, 550.0)
        sig_d, _ = bem_d.solve(exc)
        sig_i, _ = bem_i.solve(exc)
        denom = np.abs(sig_d.sig).max()
        rel = np.abs(sig_d.sig - sig_i.sig).max() / denom
        assert rel < 1e-6


class TestBEMRetIterEndToEnd(object):

    def test_matches_direct_solver(self, sphere):
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
            denom = np.abs(a).max()
            rel = np.abs(a - b).max() / denom
            assert rel < 1e-8, '{} rel diff = {:.3e}'.format(fname, rel)

    def test_iterative_faster_or_comparable(self, sphere):
        bem_d = BEMRet(sphere)
        bem_i = BEMRetIter(sphere, solver='gmres', tol=1e-6, maxit=300,
                           precond='hmat')
        pw = PlaneWaveRet(np.array([[0.0, 0.0, 1.0]]),
                          np.array([[1.0, 0.0, 0.0]]))
        exc = pw(sphere, 550.0)

        t = time.perf_counter(); bem_d.solve(exc); td = time.perf_counter() - t
        t = time.perf_counter(); bem_i.solve(exc); ti = time.perf_counter() - t
        # For n=144 sphere the iterative path has precomputed Green overhead
        # already at construction; per-solve iter should not be catastrophically
        # slower than direct.
        assert ti < 20.0 * td, 'iter too slow: direct={:.3f}s iter={:.3f}s'.format(td, ti)


class TestMaxitZero(object):

    def test_precond_only_mode(self, sphere):
        # maxit=0 + precond='hmat' -> apply preconditioner as the solver
        bem_d = BEMStat(sphere)
        bem_i = BEMStatIter(sphere, solver='gmres', maxit=0, precond='hmat')
        pw = PlaneWaveStat(np.array([[0.0, 0.0, 1.0]]))
        exc = pw(sphere, 550.0)
        sig_d, _ = bem_d.solve(exc)
        sig_i, _ = bem_i.solve(exc)
        denom = np.abs(sig_d.sig).max()
        rel = np.abs(sig_d.sig - sig_i.sig).max() / denom
        # Preconditioner is (-Lambda - F) factorised, which for stat BEM IS
        # the full system matrix -> should match direct to machine precision.
        assert rel < 1e-10
