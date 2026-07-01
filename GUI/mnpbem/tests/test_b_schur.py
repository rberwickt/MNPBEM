import os
import sys
import time

from typing import Any, Tuple

import numpy as np
from GUI.mnpbem.greenfun import CompStruct
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from GUI.mnpbem.materials import EpsConst, EpsDrude, EpsTable, make_nonlocal_pair
from GUI.mnpbem.geometry import ComParticle, trisphere
from GUI.mnpbem.greenfun import coverlayer
from GUI.mnpbem.bem import BEMRet, BEMRetIter
from GUI.mnpbem.bem.schur_iter_helpers import SchurIterOperator


# ---------------------------------------------------------------------------
# Helpers — particle factories matching test_schur_iter / test_iter_convergence
# patterns so we can compare against the same reference solutions.
# ---------------------------------------------------------------------------

def _build_nonlocal_sphere(n_faces: int = 60,
        diameter: float = 10.0) -> Tuple[ComParticle, Any]:

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


def _build_auag_dimer() -> ComParticle:

    # Au@Ag core-shell dimer (composite particle, non-uniform eps within
    # region; cross-conn != 0).  This is the case the β v1.5.1 operator-form
    # fix targets.  No cover layer — Schur reduction is N/A here, so
    # ``schur=True`` automatically degrades to the full path.
    epstab = [EpsConst(1.77), EpsTable('gold.dat'), EpsTable('silver.dat')]
    core_d, shell_t = 5.0, 1.5
    outer_d = core_d + 2.0 * shell_t
    gap = 0.6
    half = (outer_d + gap) / 2.0
    p1_shell = trisphere(60, outer_d); p1_core = trisphere(60, core_d)
    p1_shell.shift([-half, 0.0, 0.0]); p1_core.shift([-half, 0.0, 0.0])
    p2_shell = trisphere(60, outer_d); p2_core = trisphere(60, core_d)
    p2_shell.shift([+half, 0.0, 0.0]); p2_core.shift([+half, 0.0, 0.0])
    inds = [[3, 1], [2, 3], [3, 1], [2, 3]]
    return ComParticle(epstab, [p1_shell, p1_core, p2_shell, p2_core],
            inds, 1, 2, interp = 'curv')


def _build_au_sphere_uniform() -> ComParticle:

    # Au sphere with scalar EpsDrude — uniform eps within region.  Used for
    # the pointwise-vs-operator equivalence check (verification 2).
    epstab = [EpsConst(1.0), EpsDrude.gold()]
    p_sph = trisphere(60, 8.0)
    return ComParticle(epstab, [p_sph], [[2, 1]], 1, interp = 'curv')


# ---------------------------------------------------------------------------
# 검증 1: dense BEMRet vs B-Schur (iter+hmat) on small core-shell mesh
#
# The dense BEMRet (no Schur) and BEMRetIter(schur=True, hmatrix=True) must
# produce equivalent surface charges on a 60-face nonlocal core-shell sphere.
# H-matrix htol=1e-6 dominates the residual, so we check rel_diff < 5e-3.
# ---------------------------------------------------------------------------

class TestVerification1DenseVsBSchur(object):

    def test_dense_vs_b_schur_iter_hmat(self) -> None:

        cp, _ = _build_nonlocal_sphere(n_faces = 60)
        nfaces = cp.nfaces
        enei = 600.0

        rng = np.random.default_rng(42)
        phi1p = rng.standard_normal(nfaces) + 1j * rng.standard_normal(nfaces)
        a1p = rng.standard_normal((nfaces, 3)) + 1j * rng.standard_normal((nfaces, 3))
        exc = CompStruct(cp, enei, phi1p = phi1p, a1p = a1p)

        # Dense BEMRet ground truth (no schur in dense path on this geometry —
        # nonlocal cover layer triggers v1.2.0 dense Schur automatically;
        # match the iter+hmat path against the *full* dense solution
        # without dense Schur for an apples-to-apples comparison.)
        bem_dense = BEMRet(cp, schur = False)
        sig_dense, _ = bem_dense.solve(exc)

        bem_b_schur = BEMRetIter(cp,
                schur = True, hmatrix = True,
                htol = 1e-6, kmax = [4, 100], cleaf = 32,
                tol = 1e-6, maxit = 400)
        sig_b_schur, _ = bem_b_schur / exc

        assert bem_b_schur._schur_active is True
        op_info = bem_b_schur._schur_op.info()
        assert op_info['eps_form'] == 'operator', \
                '[error] expected operator-form Schur, got <{}>'.format(op_info['eps_form'])

        for name in ('sig1', 'sig2', 'h1', 'h2'):
            aa = getattr(sig_dense, name)
            bb = getattr(sig_b_schur, name)
            denom = np.linalg.norm(aa)
            rel = np.linalg.norm(aa - bb) / max(denom, 1e-30)
            # ACA htol=1e-6 + GMRES tol=1e-6 → expected drift ~1e-4..1e-3.
            assert rel < 5e-3, \
                    '[error] dense vs B-Schur <{}> mismatch: rel={}'.format(name, rel)


# ---------------------------------------------------------------------------
# 검증 2: uniform-eps pointwise vs operator equivalence
#
# When eps is scalar within the region, pointwise and operator forms produce
# identical _afun output (β v1.5.1 fast path explicitly preserves this).  The
# B-Schur reduction must therefore yield identical results regardless of
# eps_form choice.  We check both BEMRetIter(schur=True, hmatrix=False)
# variants.
# ---------------------------------------------------------------------------

class TestVerification2PointwiseVsOperator(object):

    def test_uniform_eps_pointwise_equals_operator(self) -> None:

        # We use the nonlocal cover layer geometry so schur is actually
        # active.  EpsNonlocal evaluates to a scalar at each enei (one
        # value per material), so eps1/eps2 are constant *per material
        # region*.  Within a region they are scalar from _afun's POV when
        # the per-face array reduces to a single distinct value — which
        # happens here since each region has one material.
        cp, _ = _build_nonlocal_sphere(n_faces = 60)
        nfaces = cp.nfaces
        enei = 600.0

        rng = np.random.default_rng(7)
        phi1p = rng.standard_normal(nfaces) + 1j * rng.standard_normal(nfaces)
        a1p = rng.standard_normal((nfaces, 3)) + 1j * rng.standard_normal((nfaces, 3))
        exc = CompStruct(cp, enei, phi1p = phi1p, a1p = a1p)

        # Pointwise (legacy)
        bem_pw = BEMRetIter(cp,
                schur = True,
                schur_eps_form = 'pointwise',
                tol = 1e-10, maxit = 400, precond = None)
        # Operator
        bem_op = BEMRetIter(cp,
                schur = True,
                schur_eps_form = 'operator',
                tol = 1e-10, maxit = 400, precond = None)

        sig_pw, _ = bem_pw / exc
        sig_op, _ = bem_op / exc

        assert bem_pw._schur_op.info()['eps_form'] == 'pointwise'
        assert bem_op._schur_op.info()['eps_form'] == 'operator'

        for name in ('sig1', 'sig2', 'h1', 'h2'):
            aa = getattr(sig_pw, name)
            bb = getattr(sig_op, name)
            denom = np.linalg.norm(aa)
            rel = np.linalg.norm(aa - bb) / max(denom, 1e-30)
            # Both paths solve the same A_full system; the only difference
            # is the A_ss inversion strategy (lu_dense vs gmres).  GMRES
            # tol=1e-10 → drift ~1e-8 between the two.
            assert rel < 1e-5, \
                    '[error] pointwise vs operator <{}> drift: rel={}'.format(name, rel)


# ---------------------------------------------------------------------------
# 검증 3: Au@Ag dimer iter+hmat — β v1.5.1 fix preserved (no schur regression)
#
# When schur is off, the only thing that matters is that BEMRetIter still
# reproduces the dense BEMRet result on the Au@Ag dimer (the case the β fix
# was designed to fix).  The B-Schur changes must not regress the existing
# iter+hmat path.
# ---------------------------------------------------------------------------

class TestVerification3AuAgNoRegression(object):

    def test_auag_iter_hmat_no_regression(self) -> None:

        cp = _build_auag_dimer()
        nfaces = cp.nfaces

        enei_test = np.array([380.0, 540.0, 700.0])
        pol = np.array([1.0, 0.0, 0.0])
        dirn = np.array([0.0, 0.0, 1.0])

        from GUI.mnpbem.simulation import PlaneWaveRet
        exc = PlaneWaveRet(pol, dirn)

        bem_dense = BEMRet(cp)
        bem_iter = BEMRetIter(cp,
                hmatrix = True, htol = 1e-6, kmax = [4, 100],
                tol = 1e-10, maxit = 400, precond = None)

        ext_d = np.zeros(len(enei_test))
        ext_i = np.zeros(len(enei_test))
        for k, enei in enumerate(enei_test):
            sig_d, bem_dense = bem_dense.solve(exc.potential(cp, enei))
            sig_i, bem_iter = bem_iter.solve(exc.potential(cp, enei))
            ext_d[k] = float(np.real(np.ravel(exc.extinction(sig_d)))[0])
            ext_i[k] = float(np.real(np.ravel(exc.extinction(sig_i)))[0])

        rel_diff = np.abs(ext_i - ext_d) / np.abs(ext_d)
        # β v1.5.1 fix lifted Au@Ag dimer drift from ~70% → < 1e-3.  This
        # test pins that the B-Schur changes don't regress this.
        assert rel_diff.max() < 1e-3, \
                '[error] Au@Ag iter+hmat regressed: rel_diff={}'.format(rel_diff)


# ---------------------------------------------------------------------------
# 검증 4: 60-face nonlocal core-shell + schur+iter+hmat — GMRES converges
#
# The pre-fix (v1.5.0) test ran in 6:30; post-β-fix (v1.5.1) it hung 25+ min.
# B-Schur must restore convergence within a reasonable time budget.
# ---------------------------------------------------------------------------

class TestVerification4NonlocalSchurConverges(object):

    def test_nonlocal_schur_iter_hmat_converges(self) -> None:

        cp, _ = _build_nonlocal_sphere(n_faces = 60)
        nfaces = cp.nfaces
        enei = 600.0

        rng = np.random.default_rng(11)
        phi1p = rng.standard_normal(nfaces) + 1j * rng.standard_normal(nfaces)
        a1p = rng.standard_normal((nfaces, 3)) + 1j * rng.standard_normal((nfaces, 3))
        exc = CompStruct(cp, enei, phi1p = phi1p, a1p = a1p)

        # Reference: BEMRetIter without schur (β fix path; known fast).
        bem_ref = BEMRetIter(cp,
                hmatrix = True, htol = 1e-6, kmax = [4, 100], cleaf = 32,
                tol = 1e-6, maxit = 400)
        t0 = time.time()
        sig_ref, _ = bem_ref / exc
        t_ref = time.time() - t0

        # B-Schur path
        bem_b_schur = BEMRetIter(cp,
                schur = True, hmatrix = True,
                htol = 1e-6, kmax = [4, 100], cleaf = 32,
                tol = 1e-6, maxit = 400)
        t0 = time.time()
        sig_schur, _ = bem_b_schur / exc
        t_schur = time.time() - t0

        # Convergence: must complete in well under the v1.5.1 hang time.
        # Pre-β-fix v1.5.0 took 6:30 (390 s); post-β-fix hung 25+ min.
        # Allow a generous 12 minute budget on this CI-class machine; the
        # 60-face mesh actually converges in about 6:15 in our local run.
        assert t_schur < 720.0, \
                '[error] B-Schur did not converge in 12 min: t={:.1f}s'.format(t_schur)

        # Result equivalence: B-Schur must match reference.
        for name in ('sig1', 'sig2', 'h1', 'h2'):
            aa = getattr(sig_ref, name)
            bb = getattr(sig_schur, name)
            denom = np.linalg.norm(aa)
            rel = np.linalg.norm(aa - bb) / max(denom, 1e-30)
            assert rel < 5e-3, \
                    '[error] B-Schur vs ref <{}> drift: rel={}'.format(name, rel)

        op_info = bem_b_schur._schur_op.info()
        assert op_info['eps_form'] == 'operator'
        # n_shell_dof = n_shell_faces * 8 = 116 * 8 = 928 ≤ 4096 → lu_dense.
        assert op_info['g_ss_solver'] == 'lu_dense', \
                '[error] expected lu_dense for 928-shell-dof case, got <{}>'.format(
                        op_info['g_ss_solver'])

        print('[info] B-Schur 검증 4: ref={:.1f}s, schur={:.1f}s'.format(
                t_ref, t_schur))


# ---------------------------------------------------------------------------
# Sanity: ensure SchurIterOperator constructor accepts the new kwargs
# without breaking the legacy positional API.
# ---------------------------------------------------------------------------

def test_schur_iter_operator_legacy_kwargs_still_work() -> None:

    np.random.seed(0)
    n = 20
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n) + 5.0 * np.eye(n)

    shell = np.array([0, 3, 7])
    core = np.array([i for i in range(n) if i not in set(shell)])

    # Legacy call pattern (no eps_form, no eps_diag).
    op = SchurIterOperator(
            lambda x: A @ x,
            shell, core,
            nfaces = n, components = 1, dtype = complex,
            g_ss_solver = 'lu_dense')
    info = op.info()
    assert info['eps_form'] == 'pointwise'
    assert info['has_eps_diag'] is False
