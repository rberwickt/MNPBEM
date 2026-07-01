import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mnpbem.materials.eps_const import EpsConst
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.bem.plasmonmode import plasmonmode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sphere_particle(n_verts = 32, diameter = 10.0):
    sphere = trisphere(n_verts, diameter)
    eps = [EpsConst(1.0), EpsConst(-10.0 + 1.0j)]
    p = ComParticle(eps, [sphere], [[2, 1]])
    return p


# ---------------------------------------------------------------------------
# Tests -- basic behaviour
# ---------------------------------------------------------------------------


class TestPlasmonmodeBasic(object):

    def test_returns_correct_shapes(self):
        p = _make_sphere_particle()
        nev = 5
        ene, ur, ul = plasmonmode(p, nev = nev)

        assert ene.shape == (nev,)
        assert ur.shape == (p.nfaces, nev)
        assert ul.shape == (nev, p.nfaces)

    def test_eigenvalues_are_real(self):
        p = _make_sphere_particle()
        nev = 5
        ene, ur, ul = plasmonmode(p, nev = nev)

        # Eigenvalues may be returned as complex with negligible imag part
        # (np.linalg.eig returns complex even for real symmetric output).
        if np.iscomplexobj(ene):
            assert np.allclose(ene.imag, 0.0, atol = 1e-10), \
                '[error] plasmonmode eigenvalues have non-trivial imag part'
        else:
            assert ene.dtype in (np.float64, np.float32)
        assert np.all(np.isfinite(ene))

    def test_eigenvalues_sorted_ascending(self):
        p = _make_sphere_particle()
        nev = 10
        ene, ur, ul = plasmonmode(p, nev = nev)

        for i in range(len(ene) - 1):
            assert ene[i] <= ene[i + 1] + 1e-10

    def test_eigenvectors_finite(self):
        p = _make_sphere_particle()
        nev = 5
        ene, ur, ul = plasmonmode(p, nev = nev)

        assert np.all(np.isfinite(ur))
        assert np.all(np.isfinite(ul))


# ---------------------------------------------------------------------------
# Tests -- orthogonality
# ---------------------------------------------------------------------------


class TestPlasmonmodeOrthogonality(object):

    def test_biorthogonality(self):
        p = _make_sphere_particle()
        nev = 5
        ene, ur, ul = plasmonmode(p, nev = nev)

        # ul @ ur should be close to identity
        overlap = ul @ ur  # (nev, nev)
        np.testing.assert_allclose(overlap, np.eye(nev), atol = 1e-8)

    def test_eigenvectors_satisfy_eigenvalue_equation(self):
        p = _make_sphere_particle()
        nev = 5
        ene, ur, ul = plasmonmode(p, nev = nev)

        # Reconstruct F to verify F @ ur[:,k] ~ ene[k] * ur[:,k]
        from mnpbem.greenfun import CompGreenStat
        g = CompGreenStat(p, p)
        F = g.F

        for k in range(nev):
            Fur = F @ ur[:, k]
            # project onto left eigenvector
            lam = (ul[k, :] @ Fur) / (ul[k, :] @ ur[:, k])
            np.testing.assert_allclose(lam.real, ene[k], atol = 1e-6)


# ---------------------------------------------------------------------------
# Tests -- sphere-specific properties
# ---------------------------------------------------------------------------


class TestPlasmonmodeSphere(object):

    def test_degenerate_eigenvalues_for_sphere(self):
        p = _make_sphere_particle(n_verts = 144, diameter = 10.0)
        nev = 10
        ene, ur, ul = plasmonmode(p, nev = nev)

        # For a sphere the F eigenvalues come in degenerate multiplets
        # (2l+1 degeneracy per angular momentum l).
        # ene[0] is a singlet, ene[1:4] form the l=1 dipole triplet,
        # ene[4:9] form the l=2 quadrupole quintet.
        # The spread within the l=1 triplet should be much smaller than
        # the gap between the singlet and the triplet.
        spread_dipole = abs(ene[3] - ene[1])
        gap_singlet_to_dipole = abs(ene[1] - ene[0])

        assert spread_dipole < 0.1 * gap_singlet_to_dipole

    def test_default_nev(self):
        p = _make_sphere_particle()
        ene, ur, ul = plasmonmode(p)
        # default nev is 20 but clamped to nfaces - 1
        assert ene.shape[0] == min(20, p.nfaces - 1)
        assert ur.shape[1] == ene.shape[0]
        assert ul.shape[0] == ene.shape[0]

    def test_nev_exceeds_matrix_size(self):
        p = _make_sphere_particle()
        nev = 1000
        ene, ur, ul = plasmonmode(p, nev = nev)

        assert ene.shape[0] <= p.nfaces - 1
        assert ur.shape == (p.nfaces, ene.shape[0])
        assert ul.shape == (ene.shape[0], p.nfaces)
