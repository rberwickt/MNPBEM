"""
Schur-complement BEM solver tests (MNPBEM v1.2.0).

Verifies that activating Schur reduction on EpsNonlocal cover-layer
geometries produces results that are numerically identical (up to
machine precision) to the standard full-block solve, while also
shrinking the dominant LU-factor matrix to the core-only block.
"""

import numpy as np
import pytest

from mnpbem.materials import EpsConst, EpsDrude, EpsNonlocal, make_nonlocal_pair
from mnpbem.geometry import ComParticle, trisphere
from mnpbem.greenfun import CompStruct, coverlayer
from mnpbem.bem import BEMStat, BEMRet
from mnpbem.bem.schur_helpers import (
    schur_eliminate, detect_shell_core_partition, schur_memory_estimate,
)


# ---------------------------------------------------------------------------
# Pure-math tests for the helpers
# ---------------------------------------------------------------------------

def test_schur_eliminate_matches_full_block_solve():
    np.random.seed(0)
    n = 30
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    A = A + 5.0 * np.eye(n)  # make it non-singular and well-conditioned
    b = np.random.randn(n) + 1j * np.random.randn(n)

    # Reference: solve full system.
    sig_full_ref = np.linalg.solve(A, b)

    # Schur with arbitrary partition.
    shell = np.array([0, 3, 7, 12, 19, 25])
    core = np.array([i for i in range(n) if i not in set(shell)])

    A_eff, reduce_rhs, recover = schur_eliminate(A, shell, core)
    assert A_eff.shape == (len(core), len(core))

    b_eff = reduce_rhs(b)
    sig_core = np.linalg.solve(A_eff, b_eff)
    sig_full = recover(sig_core, b)

    rel = np.linalg.norm(sig_full - sig_full_ref) / np.linalg.norm(sig_full_ref)
    assert rel < 1e-12, "Schur recover differs from full solve: rel = {}".format(rel)


def test_schur_eliminate_supports_2d_rhs():
    np.random.seed(1)
    n = 20
    A = np.random.randn(n, n) + 5.0 * np.eye(n)
    B = np.random.randn(n, 4)

    sig_full_ref = np.linalg.solve(A, B)

    shell = np.array([1, 5, 11])
    core = np.array([i for i in range(n) if i not in set(shell)])

    A_eff, reduce_rhs, recover = schur_eliminate(A, shell, core)
    B_eff = reduce_rhs(B)
    sig_core = np.linalg.solve(A_eff, B_eff)
    sig_full = recover(sig_core, B)

    rel = np.linalg.norm(sig_full - sig_full_ref) / np.linalg.norm(sig_full_ref)
    assert rel < 1e-12


def test_schur_eliminate_empty_shell():
    np.random.seed(2)
    n = 8
    A = np.random.randn(n, n) + 5.0 * np.eye(n)
    b = np.random.randn(n)

    shell = np.array([], dtype = int)
    core = np.arange(n)

    A_eff, reduce_rhs, recover = schur_eliminate(A, shell, core)
    assert A_eff.shape == (n, n)
    np.testing.assert_allclose(A_eff, A)

    b_eff = reduce_rhs(b)
    np.testing.assert_allclose(b_eff, b)

    sig_core = np.linalg.solve(A, b)
    sig_full = recover(sig_core, b)
    np.testing.assert_allclose(sig_full, sig_core)


def test_detect_shell_core_partition_no_coverlayer():
    eps_b = EpsConst(1.0)
    eps_m = EpsDrude.gold()
    p = trisphere(144, 10.0)
    cp = ComParticle([eps_b, eps_m], [p], [[2, 1]])

    partition = detect_shell_core_partition(cp)
    assert partition is None


def test_detect_shell_core_partition_with_coverlayer():
    eps_b = EpsConst(1.0)
    core_eps, shell_eps = make_nonlocal_pair('gold',
            eps_embed = eps_b,
            delta_d = 0.05)

    diameter = 10.0
    delta_d = shell_eps.delta_d
    p_core = trisphere(144, diameter - 2 * delta_d)
    p_shell = coverlayer.shift(p_core, delta_d)

    epstab = [eps_b, core_eps, shell_eps]
    inds = [[3, 1], [2, 3]]    # shell <-> embed (3,1), core <-> shell (2,3)
    cp = ComParticle(epstab, [p_shell, p_core], inds, 1, 2)

    partition = detect_shell_core_partition(cp)
    assert partition is not None

    shell_idx, core_idx = partition
    assert shell_idx.size > 0
    assert core_idx.size > 0
    # The full face count must be split exactly between the two sets.
    assert set(np.concatenate([shell_idx, core_idx]).tolist()) == set(range(cp.nfaces))
    # The two index sets must be disjoint.
    assert set(shell_idx.tolist()).isdisjoint(set(core_idx.tolist()))


def test_schur_memory_estimate_shape():
    info = schur_memory_estimate(nfaces_total = 1000, nfaces_shell = 500)
    assert info['nfaces_total'] == 1000
    assert info['nfaces_shell'] == 500
    assert info['nfaces_core'] == 500
    # Reduced matrix is 1/4 the size when shell == core (since 500^2 = 1/4 of 1000^2).
    assert info['reduction_ratio'] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Integration: BEMStat / BEMRet Schur vs full match at machine precision
# ---------------------------------------------------------------------------

def _build_nonlocal_sphere(n_faces = 144, diameter = 10.0):
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


def _make_bemstat_excitation(cp, enei):
    nfaces = cp.nfaces
    rng = np.random.default_rng(42)
    phip = rng.standard_normal(nfaces) + 1j * rng.standard_normal(nfaces)
    return CompStruct(cp, enei, phip = phip)


def test_bemstat_schur_matches_full_machine_precision():
    cp, refun = _build_nonlocal_sphere()
    enei = 600.0

    bem_full = BEMStat(cp, refun = refun)
    bem_full(enei)

    bem_schur = BEMStat(cp, refun = refun, schur = True)
    bem_schur(enei)
    assert bem_schur._schur_active

    exc = _make_bemstat_excitation(cp, enei)

    sig_full, _ = bem_full / exc
    sig_schur, _ = bem_schur / exc

    rel = np.linalg.norm(sig_full.sig - sig_schur.sig) / np.linalg.norm(sig_full.sig)
    assert rel < 1e-10, "BEMStat Schur differs from full: rel = {}".format(rel)


def test_bemstat_schur_reduced_matrix_smaller():
    cp, refun = _build_nonlocal_sphere()
    enei = 600.0

    bem_full = BEMStat(cp, refun = refun)
    bem_full(enei)

    bem_schur = BEMStat(cp, refun = refun, schur = True)
    bem_schur(enei)

    # The full LU stores an (N+M, N+M) matrix; Schur reduces to (M, M).
    full_lu = bem_full.mat_lu
    schur_lu = bem_schur.mat_lu

    # GPU LU is wrapped as ('cpu'/'gpu', lu, piv); CPU LU is (lu, piv).
    def _lu_matrix(lu_obj):
        if isinstance(lu_obj, tuple) and len(lu_obj) == 3 and lu_obj[0] in ('cpu', 'gpu'):
            return np.asarray(lu_obj[1])
        return np.asarray(lu_obj[0])

    n_full = _lu_matrix(full_lu).shape[0]
    n_schur = _lu_matrix(schur_lu).shape[0]

    assert n_schur < n_full
    assert n_schur == bem_schur._core_idx.size
    assert n_full - n_schur == bem_schur._shell_idx.size


def test_bemstat_schur_no_coverlayer_passthrough():
    """Schur=True on a particle without an EpsNonlocal cover layer must
    fall back transparently to the full BEM solve (partition=None)."""
    eps_b = EpsConst(1.0)
    eps_m = EpsDrude.gold()
    p = trisphere(144, 10.0)
    cp = ComParticle([eps_b, eps_m], [p], [[2, 1]])

    bem = BEMStat(cp, schur = True)
    bem(600.0)
    assert bem._schur_active is False
    # mat_lu must still be sized to nfaces.
    nfaces = cp.nfaces
    if isinstance(bem.mat_lu, tuple) and len(bem.mat_lu) == 3 and bem.mat_lu[0] in ('cpu', 'gpu'):
        assert bem.mat_lu[1].shape[0] == nfaces
    else:
        assert bem.mat_lu[0].shape[0] == nfaces


def test_bemret_schur_matches_full_machine_precision():
    cp, refun = _build_nonlocal_sphere()
    enei = 600.0

    bem_full = BEMRet(cp, refun = refun)
    bem_full.init(enei)

    bem_schur = BEMRet(cp, refun = refun, schur = True)
    bem_schur.init(enei)
    assert bem_schur._schur_active

    # Build a deterministic excitation. BEMRet expects a dict-like with
    # phi/a/phip/ap entries.
    nfaces = cp.nfaces
    rng = np.random.default_rng(7)
    phi1p = rng.standard_normal(nfaces) + 1j * rng.standard_normal(nfaces)
    a1p = rng.standard_normal((nfaces, 3)) + 1j * rng.standard_normal((nfaces, 3))

    exc = {
        'enei': enei,
        'phi1p': phi1p,
        'a1p': a1p,
        'p': cp,
    }

    sig_full, _ = bem_full.solve(exc)
    sig_schur, _ = bem_schur.solve(exc)

    # All four output fields should match.
    for field in ('sig1', 'sig2', 'h1', 'h2'):
        a = getattr(sig_full, field)
        b = getattr(sig_schur, field)
        norm_a = np.linalg.norm(a)
        rel = np.linalg.norm(a - b) / max(norm_a, 1e-30)
        assert rel < 1e-9, "{}: rel = {}".format(field, rel)


def test_bemret_schur_reduced_matrix_smaller():
    cp, refun = _build_nonlocal_sphere()
    enei = 600.0

    bem_full = BEMRet(cp, refun = refun)
    bem_full.init(enei)

    bem_schur = BEMRet(cp, refun = refun, schur = True)
    bem_schur.init(enei)

    def _lu_matrix(lu_obj):
        if isinstance(lu_obj, tuple) and len(lu_obj) == 3 and lu_obj[0] in ('cpu', 'gpu'):
            return np.asarray(lu_obj[1])
        return np.asarray(lu_obj[0])

    n_full = _lu_matrix(bem_full.Sigma_lu).shape[0]
    n_schur = _lu_matrix(bem_schur.Sigma_lu).shape[0]
    assert n_schur < n_full

    info = schur_memory_estimate(nfaces_total = n_full, nfaces_shell = n_full - n_schur)
    # sanity: reduced matrix is smaller than the full matrix
    assert info['reduction_ratio'] < 1.0


def test_bemret_schur_no_coverlayer_passthrough():
    eps_b = EpsConst(1.0)
    eps_m = EpsDrude.gold()
    p = trisphere(144, 10.0)
    cp = ComParticle([eps_b, eps_m], [p], [[2, 1]])

    bem = BEMRet(cp, schur = True)
    bem.init(600.0)
    assert bem._schur_active is False


# ---------------------------------------------------------------------------
# Iterative path: v1.5.0 enabled <schur> on BEMStatIter / BEMRetIter
# ---------------------------------------------------------------------------
# v1.2.0 raised NotImplementedError when schur=True met the iterative path.
# v1.5.0 ships SchurIterOperator (mnpbem/bem/schur_iter_helpers.py) which
# combines Schur reduction with hmatrix=True.  The two tests below assert
# that schur=True no longer raises *and* falls back transparently when no
# EpsNonlocal cover layer is present.  Functional correctness is covered
# by mnpbem/tests/test_schur_iter.py.

def test_bemstatiter_schur_no_coverlayer_passthrough():
    from mnpbem.bem import BEMStatIter

    eps_b = EpsConst(1.0)
    eps_m = EpsDrude.gold()
    p = trisphere(144, 10.0)
    cp = ComParticle([eps_b, eps_m], [p], [[2, 1]])

    bem = BEMStatIter(cp, schur = True)
    bem._init_matrices(600.0)
    assert bem._schur_active is False


def test_bemretiter_schur_no_coverlayer_passthrough():
    from mnpbem.bem import BEMRetIter

    eps_b = EpsConst(1.0)
    eps_m = EpsDrude.gold()
    p = trisphere(144, 10.0)
    cp = ComParticle([eps_b, eps_m], [p], [[2, 1]])

    bem = BEMRetIter(cp, schur = True, precond = None)
    bem._init_matrices(600.0)
    assert bem._schur_active is False
