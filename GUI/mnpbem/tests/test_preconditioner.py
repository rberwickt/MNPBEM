import os
import sys

from typing import Any, Tuple

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mnpbem.materials import EpsConst
from mnpbem.geometry import ComParticle, trisphere
from mnpbem.bem import BEMRetIter, BEMStatIter
from mnpbem.bem.preconditioner import HMatrixLUPreconditioner
from mnpbem.simulation.planewave_ret import PlaneWaveRet
from mnpbem.simulation.planewave_stat import PlaneWaveStat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sphere(n_face: int = 144,
        diameter: float = 10.0,
        eps_in: complex = -10.0 + 0.5j) -> ComParticle:

    eps_b = EpsConst(1.0)
    eps_m = EpsConst(eps_in)
    p = trisphere(n_face, diameter)
    cp = ComParticle([eps_b, eps_m], [p], [[2, 1]])
    return cp


# ---------------------------------------------------------------------------
# Standalone class tests
# ---------------------------------------------------------------------------

class TestHMatrixLUPreconditioner(object):

    def test_dense_mode_solves_random(self) -> None:

        rng = np.random.default_rng(42)
        n = 50
        A = rng.standard_normal((n, n)) + np.eye(n) * 5.0
        b = rng.standard_normal(n)
        precond = HMatrixLUPreconditioner(A, mode = 'dense')
        x = precond.solve(b)
        assert np.linalg.norm(A @ x - b) < 1e-10, '[error] dense LU residual too large'

    def test_dense_mode_complex(self) -> None:

        rng = np.random.default_rng(7)
        n = 32
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)) + np.eye(n) * 5.0
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        precond = HMatrixLUPreconditioner(A, mode = 'dense')
        x = precond.solve(b)
        assert np.linalg.norm(A @ x - b) < 1e-10

    def test_auto_picks_dense_for_small(self) -> None:

        n = 100
        A = np.eye(n) + 0.01 * np.random.default_rng(0).standard_normal((n, n))
        precond = HMatrixLUPreconditioner(A, mode = 'auto')
        # n < threshold => dense
        assert precond.mode == 'dense'

    def test_matvec_alias(self) -> None:

        rng = np.random.default_rng(1)
        n = 16
        A = rng.standard_normal((n, n)) + np.eye(n) * 4.0
        b = rng.standard_normal(n)
        p = HMatrixLUPreconditioner(A, mode = 'dense')
        assert np.allclose(p.matvec(b), p.solve(b))
        assert np.allclose(p @ b, p.solve(b))


# ---------------------------------------------------------------------------
# BEMRetIter integration: 'auto' vs 'none' on a small sphere
# ---------------------------------------------------------------------------

class TestBEMRetIterPreconditioner(object):

    def _setup(self,
            n_face: int = 144) -> Tuple[Any, Any]:

        cp = _make_sphere(n_face = n_face, diameter = 10.0)
        enei = 600.0
        pol = np.array([[1.0, 0.0, 0.0]])
        dirn = np.array([[0.0, 0.0, 1.0]])
        exc = PlaneWaveRet(pol = pol, dir = dirn)
        return cp, exc(cp, enei)

    def test_auto_matches_none(self) -> None:

        cp, exc_struct = self._setup(n_face = 144)

        bem_no = BEMRetIter(cp, hmatrix = True, htol = 1e-8,
                kmax = [4, 100], cleaf = 32,
                tol = 1e-8, maxit = 400, preconditioner = 'none')
        sig_no, bem_no_done = bem_no.solve(exc_struct)

        bem_auto = BEMRetIter(cp, hmatrix = True, htol = 1e-8,
                kmax = [4, 100], cleaf = 32,
                tol = 1e-8, maxit = 400, preconditioner = 'auto')
        sig_auto, bem_auto_done = bem_auto.solve(exc_struct)

        for name in ('sig1', 'sig2', 'h1', 'h2'):
            v_no = getattr(sig_no, name)
            v_auto = getattr(sig_auto, name)
            denom = np.linalg.norm(v_no)
            err = np.linalg.norm(v_auto - v_no) / denom if denom > 0 else 0.0
            assert err < 1e-3, '[error] auto preconditioner mismatch on <{}>: rel={:.3e}'.format(name, err)

    def test_hlu_dense_explicit(self) -> None:

        cp, exc_struct = self._setup(n_face = 144)

        bem = BEMRetIter(cp, hmatrix = True, htol = 1e-8,
                kmax = [4, 100], cleaf = 32,
                tol = 1e-7, maxit = 400, preconditioner = 'hlu_dense')
        sig, _ = bem.solve(exc_struct)
        assert np.isfinite(sig.sig1).all()
        assert np.linalg.norm(sig.sig1) > 0

    def test_none_keeps_v13_behaviour(self) -> None:

        cp, exc_struct = self._setup(n_face = 144)
        bem = BEMRetIter(cp, hmatrix = True, htol = 1e-8,
                kmax = [4, 100], cleaf = 32,
                tol = 1e-7, maxit = 400, preconditioner = 'none')
        # Should run without building HLU.
        assert bem._hlu_object is None
        sig, _ = bem.solve(exc_struct)
        assert bem._hlu_object is None
        assert np.isfinite(sig.sig1).all()


# ---------------------------------------------------------------------------
# BEMStatIter integration
# ---------------------------------------------------------------------------

class TestBEMStatIterPreconditioner(object):

    def _setup(self,
            n_face: int = 144) -> Tuple[Any, Any]:

        cp = _make_sphere(n_face = n_face, diameter = 10.0)
        pol = np.array([[1.0, 0.0, 0.0]])
        exc = PlaneWaveStat(pol = pol)
        return cp, exc(cp, 600.0)

    def test_auto_matches_none(self) -> None:

        cp, exc_struct = self._setup(n_face = 144)

        bem_no = BEMStatIter(cp, hmatrix = True, htol = 1e-8,
                kmax = [4, 100], cleaf = 32,
                tol = 1e-8, maxit = 400, preconditioner = 'none')
        sig_no, _ = bem_no.solve(exc_struct)

        bem_auto = BEMStatIter(cp, hmatrix = True, htol = 1e-8,
                kmax = [4, 100], cleaf = 32,
                tol = 1e-8, maxit = 400, preconditioner = 'auto')
        sig_auto, _ = bem_auto.solve(exc_struct)

        denom = np.linalg.norm(sig_no.sig)
        err = np.linalg.norm(sig_auto.sig - sig_no.sig) / denom if denom > 0 else 0.0
        assert err < 1e-3, '[error] BEMStatIter HLU mismatch: rel={:.3e}'.format(err)


