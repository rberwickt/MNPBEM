import os
import sys

from typing import Tuple

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mnpbem.materials import EpsConst, EpsDrude, make_nonlocal_pair
from mnpbem.geometry import ComParticle, trisphere
from mnpbem.greenfun import CompStruct, coverlayer
from mnpbem.bem import BEMStat, BEMStatIter, BEMRetIter
from mnpbem.bem.schur_iter_helpers import (
        SchurIterOperator,
        detect_iter_partition,
        schur_iter_memory_estimate,
)
from scipy.sparse.linalg import gmres


# ---------------------------------------------------------------------------
# Pure-math tests on the LinearOperator
# ---------------------------------------------------------------------------

class TestSchurIterOperatorMath(object):

    def test_lu_dense_path_machine_precision(self) -> None:

        np.random.seed(0)
        n = 30
        A = np.random.randn(n, n) + 1j * np.random.randn(n, n) + 5.0 * np.eye(n)
        b = np.random.randn(n) + 1j * np.random.randn(n)
        sig_full = np.linalg.solve(A, b)

        shell = np.array([0, 3, 7, 12, 19, 25])
        core = np.array([i for i in range(n) if i not in set(shell)])

        op = SchurIterOperator(
                lambda x: A @ x,
                shell, core,
                nfaces = n, components = 1, dtype = complex,
                g_ss_solver = 'lu_dense')

        b_eff = op.reduce_rhs(b)
        sig_core, info = gmres(op, b_eff, rtol = 1e-12, maxiter = 200,
                restart = min(op._n_core, 50))
        sig_recovered = op.recover_full(sig_core, b)

        assert info == 0
        rel = np.linalg.norm(sig_recovered - sig_full) / np.linalg.norm(sig_full)
        assert rel < 1e-10, '[error] lu_dense path mismatch: rel={}'.format(rel)

    def test_gmres_inner_path(self) -> None:

        np.random.seed(1)
        n = 30
        A = np.random.randn(n, n) + 1j * np.random.randn(n, n) + 5.0 * np.eye(n)
        b = np.random.randn(n) + 1j * np.random.randn(n)
        sig_full = np.linalg.solve(A, b)

        shell = np.array([1, 4, 9, 17])
        core = np.array([i for i in range(n) if i not in set(shell)])

        op = SchurIterOperator(
                lambda x: A @ x,
                shell, core,
                nfaces = n, components = 1, dtype = complex,
                g_ss_solver = 'gmres', inner_tol = 1e-12)

        b_eff = op.reduce_rhs(b)
        sig_core, info = gmres(op, b_eff, rtol = 1e-10, maxiter = 200,
                restart = min(op._n_core, 50))
        sig_recovered = op.recover_full(sig_core, b)

        assert info == 0
        rel = np.linalg.norm(sig_recovered - sig_full) / np.linalg.norm(sig_full)
        assert rel < 1e-8, '[error] gmres inner path mismatch: rel={}'.format(rel)

    def test_components_8_lifting(self) -> None:

        # 8N system that is block-diagonal across the 8 components, with
        # each component having the same per-face shell/core structure.
        # The lifted partition must keep the systems decoupled.
        np.random.seed(2)
        nfaces = 12
        comps = 8
        N = nfaces * comps
        A_block = np.random.randn(nfaces, nfaces) + 1j * np.random.randn(nfaces, nfaces) + 5.0 * np.eye(nfaces)
        # Block-diag A across components.
        A_full = np.zeros((N, N), dtype = complex)
        for k in range(comps):
            A_full[k * nfaces:(k + 1) * nfaces, k * nfaces:(k + 1) * nfaces] = A_block

        b = np.random.randn(N) + 1j * np.random.randn(N)
        sig_full = np.linalg.solve(A_full, b)

        shell_face = np.array([0, 3, 7])
        core_face = np.array([i for i in range(nfaces) if i not in set(shell_face)])

        op = SchurIterOperator(
                lambda x: A_full @ x,
                shell_face, core_face,
                nfaces = nfaces, components = comps, dtype = complex,
                g_ss_solver = 'lu_dense')

        b_eff = op.reduce_rhs(b)
        sig_core, info = gmres(op, b_eff, rtol = 1e-12, maxiter = 400,
                restart = min(op._n_core, 50))
        sig_recovered = op.recover_full(sig_core, b)

        rel = np.linalg.norm(sig_recovered - sig_full) / np.linalg.norm(sig_full)
        assert rel < 1e-9, '[error] 8-component lifting mismatch: rel={}'.format(rel)


# ---------------------------------------------------------------------------
# BEMStatIter integration
# ---------------------------------------------------------------------------

def _build_nonlocal_sphere(n_faces: int = 144,
        diameter: float = 10.0) -> Tuple[ComParticle, object]:

    eps_b = EpsConst(1.0)
    core_eps, shell_eps = make_nonlocal_pair('gold',
            eps_embed = eps_b,
            delta_d = 0.05)

    delta_d = shell_eps.delta_d
    p_core = trisphere(n_faces, diameter - 2 * delta_d)
    p_shell = coverlayer.shift(p_core, delta_d)

    epstab = [eps_b, core_eps, shell_eps]
    inds = [[3, 1], [2, 3]]
    cp = ComParticle(epstab, [p_shell, p_core], inds, 1, 2)
    refun = coverlayer.refine(cp, [[1, 2]])
    return cp, refun


class TestBEMStatIterSchur(object):

    def test_schur_dense_matches_no_schur(self) -> None:

        cp, _ = _build_nonlocal_sphere(n_faces = 144)
        enei = 600.0
        nfaces = cp.nfaces

        bem_full = BEMStatIter(cp, tol = 1e-10, maxit = 400, precond = None)
        bem_schur = BEMStatIter(cp, schur = True, tol = 1e-10, maxit = 400, precond = None)

        rng = np.random.default_rng(42)
        phip = rng.standard_normal(nfaces) + 1j * rng.standard_normal(nfaces)
        exc = CompStruct(cp, enei, phip = phip)

        sig_full, _ = bem_full / exc
        sig_schur, _ = bem_schur / exc

        assert bem_schur._schur_active is True

        rel = np.linalg.norm(sig_full.sig - sig_schur.sig) / np.linalg.norm(sig_full.sig)
        assert rel < 1e-8, '[error] BEMStatIter Schur mismatch: rel={}'.format(rel)

    def test_schur_with_hmatrix(self) -> None:

        cp, _ = _build_nonlocal_sphere(n_faces = 144)
        enei = 600.0
        nfaces = cp.nfaces

        bem_full = BEMStatIter(cp, tol = 1e-8, maxit = 400, precond = None)
        bem_schur = BEMStatIter(cp,
                schur = True, hmatrix = True,
                htol = 1e-8, kmax = [4, 100], cleaf = 32,
                tol = 1e-8, maxit = 400)

        rng = np.random.default_rng(1)
        phip = rng.standard_normal(nfaces) + 1j * rng.standard_normal(nfaces)
        exc = CompStruct(cp, enei, phip = phip)

        sig_full, _ = bem_full / exc
        sig_schur, _ = bem_schur / exc

        assert bem_schur._schur_active is True

        rel = np.linalg.norm(sig_full.sig - sig_schur.sig) / np.linalg.norm(sig_full.sig)
        # ACA htol=1e-8 dominates the error.
        assert rel < 1e-4, '[error] BEMStatIter Schur+hmatrix mismatch: rel={}'.format(rel)

    def test_schur_false_with_hmatrix_unchanged(self) -> None:

        # Regression: schur=False + hmatrix=True must keep working as in v1.3.0.
        cp, _ = _build_nonlocal_sphere(n_faces = 144)
        enei = 600.0
        nfaces = cp.nfaces

        bem = BEMStatIter(cp,
                hmatrix = True,
                htol = 1e-6, kmax = [4, 100], cleaf = 32,
                tol = 1e-6, maxit = 400)

        rng = np.random.default_rng(7)
        phip = rng.standard_normal(nfaces) + 1j * rng.standard_normal(nfaces)
        exc = CompStruct(cp, enei, phip = phip)

        sig, _ = bem / exc
        assert bem._schur_active is False
        assert np.isfinite(sig.sig).all()


# ---------------------------------------------------------------------------
# BEMRetIter integration
# ---------------------------------------------------------------------------

class TestBEMRetIterSchur(object):

    def test_schur_dense_matches_no_schur(self) -> None:

        cp, _ = _build_nonlocal_sphere(n_faces = 60)
        enei = 600.0
        nfaces = cp.nfaces

        bem_full = BEMRetIter(cp, tol = 1e-6, maxit = 400)
        bem_schur = BEMRetIter(cp, schur = True, tol = 1e-6, maxit = 400, precond = None)

        rng = np.random.default_rng(42)
        phi1p = rng.standard_normal(nfaces) + 1j * rng.standard_normal(nfaces)
        a1p = rng.standard_normal((nfaces, 3)) + 1j * rng.standard_normal((nfaces, 3))
        exc = CompStruct(cp, enei, phi1p = phi1p, a1p = a1p)

        sig_full, _ = bem_full / exc
        sig_schur, _ = bem_schur / exc

        assert bem_schur._schur_active is True

        for name in ('sig1', 'sig2', 'h1', 'h2'):
            aa = getattr(sig_full, name)
            bb = getattr(sig_schur, name)
            denom = np.linalg.norm(aa)
            rel = np.linalg.norm(aa - bb) / max(denom, 1e-30)
            assert rel < 5e-3, '[error] BEMRetIter Schur <{}> mismatch: rel={}'.format(name, rel)

    def test_schur_with_hmatrix(self) -> None:

        cp, _ = _build_nonlocal_sphere(n_faces = 60)
        enei = 600.0
        nfaces = cp.nfaces

        bem_ref = BEMRetIter(cp,
                hmatrix = True,
                htol = 1e-6, kmax = [4, 100], cleaf = 32,
                tol = 1e-6, maxit = 400)
        bem_schur = BEMRetIter(cp,
                schur = True, hmatrix = True,
                htol = 1e-6, kmax = [4, 100], cleaf = 32,
                tol = 1e-6, maxit = 400)

        rng = np.random.default_rng(11)
        phi1p = rng.standard_normal(nfaces) + 1j * rng.standard_normal(nfaces)
        a1p = rng.standard_normal((nfaces, 3)) + 1j * rng.standard_normal((nfaces, 3))
        exc = CompStruct(cp, enei, phi1p = phi1p, a1p = a1p)

        sig_ref, _ = bem_ref / exc
        sig_schur, _ = bem_schur / exc

        assert bem_schur._schur_active is True

        for name in ('sig1', 'sig2', 'h1', 'h2'):
            aa = getattr(sig_ref, name)
            bb = getattr(sig_schur, name)
            denom = np.linalg.norm(aa)
            rel = np.linalg.norm(aa - bb) / max(denom, 1e-30)
            assert rel < 5e-3, '[error] BEMRetIter Schur+hmatrix <{}> mismatch: rel={}'.format(name, rel)

    def test_schur_false_with_hmatrix_unchanged(self) -> None:

        cp, _ = _build_nonlocal_sphere(n_faces = 60)
        enei = 600.0
        nfaces = cp.nfaces

        bem = BEMRetIter(cp,
                hmatrix = True,
                htol = 1e-6, kmax = [4, 100], cleaf = 32,
                tol = 1e-6, maxit = 400)

        rng = np.random.default_rng(13)
        phi1p = rng.standard_normal(nfaces) + 1j * rng.standard_normal(nfaces)
        a1p = rng.standard_normal((nfaces, 3)) + 1j * rng.standard_normal((nfaces, 3))
        exc = CompStruct(cp, enei, phi1p = phi1p, a1p = a1p)

        sig, _ = bem / exc
        assert bem._schur_active is False
        assert np.isfinite(sig.sig1).all()


# ---------------------------------------------------------------------------
# Memory estimator
# ---------------------------------------------------------------------------

def test_schur_iter_memory_estimate_components() -> None:

    info = schur_iter_memory_estimate(
            nfaces_total = 1000, nfaces_shell = 500, components = 8)
    assert info['nfaces_total'] == 1000
    assert info['nfaces_core'] == 500
    assert info['components'] == 8
    assert info['full_dof'] == 8000
    assert info['core_dof'] == 4000
    assert info['shell_dof'] == 4000
    assert info['krylov_reduction_ratio'] == pytest.approx(0.5)


def test_detect_iter_partition_redirect() -> None:

    # detect_iter_partition is a thin re-export of the v1.2.0 detector.
    cp, _ = _build_nonlocal_sphere(n_faces = 60)
    partition = detect_iter_partition(cp)
    assert partition is not None
    s, c = partition
    assert s.size > 0 and c.size > 0
    assert s.size + c.size == cp.nfaces
