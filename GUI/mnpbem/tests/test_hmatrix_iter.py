import os
import sys

from typing import Tuple

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mnpbem.materials import EpsConst
from mnpbem.geometry import ComParticle, trisphere
from mnpbem.bem import BEMRetIter, BEMStatIter, BEMRetLayerIter
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
# Cross-check: dense BEMRetIter vs hmatrix=True
# ---------------------------------------------------------------------------

class TestBEMRetIterHMatrix(object):

    def test_small_sphere_dense_vs_hmatrix(self) -> None:

        cp = _make_sphere(n_face = 144, diameter = 10.0)
        enei = 600.0
        pol = np.array([[1.0, 0.0, 0.0]])
        dirn = np.array([[0.0, 0.0, 1.0]])
        exc = PlaneWaveRet(pol = pol, dir = dirn)
        exc_struct = exc(cp, enei)

        # Dense path (existing default)
        bem_dense = BEMRetIter(cp, tol = 1e-8, maxit = 200)
        sig_dense, _ = bem_dense.solve(exc_struct)

        # H-matrix path
        bem_hmat = BEMRetIter(cp, tol = 1e-8, maxit = 200,
                hmatrix = True, htol = 1e-8, kmax = [4, 100], cleaf = 32)
        sig_hmat, _ = bem_hmat.solve(exc_struct)

        # ACA tol drives the per-block error; use a relaxed tolerance.
        for name in ('sig1', 'sig2', 'h1', 'h2'):
            v_dense = getattr(sig_dense, name)
            v_hmat = getattr(sig_hmat, name)
            denom = np.linalg.norm(v_dense)
            err = np.linalg.norm(v_hmat - v_dense) / denom if denom > 0 else 0.0
            assert err < 1e-3, '[error] BEMRetIter hmatrix mismatch on <{}>: rel={:.3e}'.format(name, err)

    def test_options_are_consumed(self) -> None:

        cp = _make_sphere(n_face = 32)
        # Should not raise: hmatrix + standard options accepted.
        bem = BEMRetIter(cp, hmatrix = True, htol = 1e-6,
                kmax = [4, 50], cleaf = 32, eta = 2.5, tol = 1e-6)
        assert bem._hmatrix is True

    def test_hmatrix_default_disables_precond(self) -> None:

        cp = _make_sphere(n_face = 32)
        bem = BEMRetIter(cp, hmatrix = True)
        # default precond should fall to None (no dense LU)
        assert bem.precond is None

    def test_hmatrix_with_refun_raises(self) -> None:

        cp = _make_sphere(n_face = 32)
        with pytest.raises(NotImplementedError):
            BEMRetIter(cp, hmatrix = True, refun = lambda g, G, H: (G, H))


# ---------------------------------------------------------------------------
# Cross-check: dense BEMStatIter vs hmatrix=True
# ---------------------------------------------------------------------------

class TestBEMStatIterHMatrix(object):

    def test_small_sphere_dense_vs_hmatrix(self) -> None:

        cp = _make_sphere(n_face = 144, diameter = 10.0)
        enei = 600.0
        pol = np.array([[1.0, 0.0, 0.0]])
        dirn = np.array([[0.0, 0.0, 1.0]])
        # Quasistatic excitation
        exc = PlaneWaveStat(pol = pol)
        exc_struct = exc(cp, enei)

        bem_dense = BEMStatIter(cp, tol = 1e-8, maxit = 200, precond = None)
        sig_dense, _ = bem_dense.solve(exc_struct)

        bem_hmat = BEMStatIter(cp, tol = 1e-8, maxit = 200,
                hmatrix = True, htol = 1e-8, kmax = [4, 100], cleaf = 32)
        sig_hmat, _ = bem_hmat.solve(exc_struct)

        denom = np.linalg.norm(sig_dense.sig)
        err = np.linalg.norm(sig_hmat.sig - sig_dense.sig) / denom if denom > 0 else 0.0
        assert err < 1e-3, '[error] BEMStatIter hmatrix mismatch: rel={:.3e}'.format(err)


# ---------------------------------------------------------------------------
# Layered iter solver: hmatrix must raise NotImplementedError
# ---------------------------------------------------------------------------

class TestBEMRetLayerIterHMatrix(object):

    def test_layered_iter_hmatrix_raises(self) -> None:

        cp = _make_sphere(n_face = 32)
        with pytest.raises(NotImplementedError):
            BEMRetLayerIter(cp, layer = None, hmatrix = True)


# ---------------------------------------------------------------------------
# Larger-mesh smoke test: hmatrix only (dense reference too costly here)
# ---------------------------------------------------------------------------

class TestBEMRetIterHMatrixSmoke(object):

    def test_medium_sphere_runs(self) -> None:

        cp = _make_sphere(n_face = 1024, diameter = 20.0)
        enei = 600.0
        pol = np.array([[1.0, 0.0, 0.0]])
        dirn = np.array([[0.0, 0.0, 1.0]])
        exc = PlaneWaveRet(pol = pol, dir = dirn)
        exc_struct = exc(cp, enei)

        bem = BEMRetIter(cp, hmatrix = True, htol = 1e-6,
                kmax = [4, 100], cleaf = 64, tol = 1e-5, maxit = 400)
        sig, _ = bem.solve(exc_struct)

        # Just check the solver returned a non-trivial solution.
        assert np.isfinite(sig.sig1).all()
        assert np.linalg.norm(sig.sig1) > 0
