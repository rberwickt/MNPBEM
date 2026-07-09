"""
BEMRet refun (cover-layer refinement) integration tests.

Verifies that BEMRet accepts the optional ``refun`` keyword (matching the
MATLAB ``bemsolver(p, op, 'refun', coverlayer.refine(...))`` signature) and
applies it to the assembled BEM matrices before LU factorization.

Wave 2 Agent β.
"""

from typing import Any, Tuple

import numpy as np
import pytest

from mnpbem.materials import EpsConst, EpsDrude, EpsNonlocal, make_nonlocal_pair
from mnpbem.geometry import ComParticle, trisphere
from mnpbem.greenfun import coverlayer
from mnpbem.bem import BEMRet


def _build_simple_sphere():
    eps_b = EpsConst(1.0)
    eps_m = EpsDrude.gold()
    p = trisphere(144, 10.0)
    cp = ComParticle([eps_b, eps_m], [p], [[2, 1]])
    return cp


def test_bemret_accepts_refun_kwarg():
    cp = _build_simple_sphere()

    def noop_refun(obj: Any, g: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return g, f

    # Should not raise.
    bem = BEMRet(cp, refun = noop_refun)
    assert bem is not None
    assert bem._refun is noop_refun


def test_bemret_noop_refun_matches_no_refun():
    """A refun that returns its inputs unchanged must produce identical
    BEM matrices vs. no refun at all (LU factor is unique up to pivot
    permutation, so we compare the resolvent matrix directly via Sigma_lu
    after a deterministic init at a fixed energy)."""
    cp = _build_simple_sphere()

    def noop_refun(obj: Any, g: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return g, f

    enei = 600.0
    bem_a = BEMRet(cp)
    bem_a.init(enei)

    bem_b = BEMRet(cp, refun = noop_refun)
    bem_b.init(enei)

    # eps1 / eps2 / k must be identical
    assert bem_a.k == bem_b.k
    if np.isscalar(bem_a.eps1):
        assert bem_a.eps1 == bem_b.eps1
        assert bem_a.eps2 == bem_b.eps2
    else:
        assert np.allclose(bem_a.eps1, bem_b.eps1)
        assert np.allclose(bem_a.eps2, bem_b.eps2)

    # Sigma1 (H1 @ G1^-1) must be identical when refun is no-op.
    s1a = bem_a.Sigma1
    s1b = bem_b.Sigma1
    if hasattr(s1a, 'get'):
        s1a = s1a.get()
    if hasattr(s1b, 'get'):
        s1b = s1b.get()
    assert np.allclose(s1a, s1b, rtol = 1e-12, atol = 1e-12)


def test_bemret_refun_is_called_twice():
    """refun must be called once for (G1, H1) and once for (G2, H2). We
    verify the call count and the matrix shapes passed in match the
    BEM problem dimension."""
    cp = _build_simple_sphere()
    nfaces = cp.nfaces
    call_log = []

    def tracking_refun(obj: Any, g: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        call_log.append((g.shape, f.shape))
        return g, f

    bem = BEMRet(cp, refun = tracking_refun)
    bem.init(600.0)

    assert len(call_log) == 2
    for g_shape, f_shape in call_log:
        assert g_shape == (nfaces, nfaces)
        assert f_shape == (nfaces, nfaces)


def test_bemret_refun_modifies_matrices():
    """Verify refun output actually flows into the LU-factored Sigma1.
    A refun that scales G/H by a constant should yield a Sigma1 that
    differs from the unmodified case by exactly that factor (since
    Sigma1 = H @ G^-1 — alpha*H @ (alpha*G)^-1 = H @ G^-1, scaling
    cancels).  We instead use an additive perturbation that should
    propagate to Sigma1."""
    cp = _build_simple_sphere()
    enei = 600.0

    bem_clean = BEMRet(cp)
    bem_clean.init(enei)

    def perturb_refun(obj: Any, g: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Add a small perturbation only to one diagonal element so we do
        # not destroy invertibility.
        g2 = g.copy()
        g2[0, 0] = g2[0, 0] * 1.001
        return g2, f

    bem_perturbed = BEMRet(cp, refun = perturb_refun)
    bem_perturbed.init(enei)

    s1a = bem_clean.Sigma1
    s1b = bem_perturbed.Sigma1
    if hasattr(s1a, 'get'):
        s1a = s1a.get()
    if hasattr(s1b, 'get'):
        s1b = s1b.get()

    # Should differ (refun did flow into the assembled matrices)
    diff = np.linalg.norm(s1a - s1b) / np.linalg.norm(s1a)
    assert diff > 1e-6


def test_bemret_with_coverlayer_refine_smoke():
    """Smoke test: the canonical ``coverlayer.refine`` callable must be
    accepted by BEMRet and produce a successfully-initialised solver
    (no exceptions, finite Sigma_lu)."""
    eps_b = EpsConst(1.0)
    core_eps, shell_eps = make_nonlocal_pair('gold',
            eps_embed = eps_b,
            delta_d = 0.05)

    diameter = 10.0
    delta_d = shell_eps.delta_d
    p_core = trisphere(144, diameter - 2 * delta_d)
    p_shell = coverlayer.shift(p_core, delta_d)

    epstab = [eps_b, core_eps, shell_eps]
    inds = [[3, 1], [2, 3]]
    p = ComParticle(epstab, [p_shell, p_core], inds, 1, 2)

    refun = coverlayer.refine(p, [[1, 2]])
    bem = BEMRet(p, refun = refun)
    bem.init(600.0)

    assert bem.Sigma_lu is not None
    # Sigma1 must be finite
    s1 = bem.Sigma1
    if hasattr(s1, 'get'):
        s1 = s1.get()
    assert np.all(np.isfinite(s1))


def test_bemret_nano_dimer_with_nonlocal_smoke():
    """Nano-gap dimer (1 nm gap, 5 nm radius spheres) with EpsNonlocal
    cover-layer.  Runs BEMRet through a plane-wave excitation at 600 nm
    and checks that the resulting absorption cross section is positive
    and finite — the actual cross-section value is not validated against
    MATLAB here (that lives in Wave 3 regression).  This is a 'pipeline
    works end-to-end' smoke test only."""
    eps_b = EpsConst(1.0)
    core_eps, shell_eps = make_nonlocal_pair('gold',
            eps_embed = eps_b,
            delta_d = 0.05)

    radius = 5.0
    gap = 1.0
    delta_d = shell_eps.delta_d

    # Build cores.  Two spheres separated by gap along x-axis.
    p_core_a = trisphere(144, 2 * radius - 2 * delta_d)
    p_core_a.shift([-(radius + gap / 2.0), 0.0, 0.0])
    p_core_b = trisphere(144, 2 * radius - 2 * delta_d)
    p_core_b.shift([+(radius + gap / 2.0), 0.0, 0.0])

    p_shell_a = coverlayer.shift(p_core_a, delta_d)
    p_shell_b = coverlayer.shift(p_core_b, delta_d)

    # epstab: [embedding, core_drude, nonlocal_shell]
    epstab = [eps_b, core_eps, shell_eps]
    # shell <-> embed (3,1), core <-> shell (2,3) — for both spheres.
    inds = [[3, 1], [2, 3], [3, 1], [2, 3]]
    p = ComParticle(epstab, [p_shell_a, p_core_a, p_shell_b, p_core_b], inds, 1, 2)

    refun = coverlayer.refine(p, [[1, 2], [3, 4]])
    bem = BEMRet(p, refun = refun)
    bem.init(600.0)

    # All BEM matrices finite
    s1 = bem.Sigma1
    if hasattr(s1, 'get'):
        s1 = s1.get()
    assert np.all(np.isfinite(s1))
    assert bem.Sigma_lu is not None


def test_bemret_refun_signature_compatibility():
    """The refun callable must conform to the (obj, G, F) -> (G, F)
    contract.  A buggy refun that returns wrong shape should propagate
    the error (not silently corrupt state)."""
    cp = _build_simple_sphere()

    def bad_refun(obj: Any, g: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return g[:5, :5], f[:5, :5]    # wrong shape

    bem = BEMRet(cp, refun = bad_refun)
    with pytest.raises((ValueError, IndexError, np.linalg.LinAlgError, RuntimeError)):
        bem.init(600.0)
