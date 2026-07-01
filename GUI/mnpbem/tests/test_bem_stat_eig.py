"""
Tests for BEMStatEig (quasistatic BEM with eigenmode expansion).

Tests:
  - Construction
  - Init with wavelength
  - Eigenmode computation (verify eigenvalues are sorted)
  - Solve produces same result as BEMStat (within tolerance)
  - Decay rate computation
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mnpbem.materials.eps_const import EpsConst
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.greenfun import CompStruct
from mnpbem.bem.bem_stat import BEMStat
from mnpbem.bem.bem_stat_eig import BEMStatEig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sphere_particle(n_verts = 32, diameter = 10.0):
    """Create a simple gold-like sphere particle."""
    sphere = trisphere(n_verts, diameter)
    eps = [EpsConst(1.0), EpsConst(-10.0 + 1.0j)]
    p = ComParticle(eps, [sphere], [[2, 1]])
    return p


def _make_excitation(p, enei = 500.0):
    """Create a plane-wave-like excitation (phip = -nvec dot pol)."""
    pol = np.array([1.0, 0.0, 0.0])
    phip = -p.nvec @ pol
    exc = CompStruct(p, enei, phip = phip)
    return exc


# ---------------------------------------------------------------------------
# Tests -- construction
# ---------------------------------------------------------------------------


class TestBEMStatEigConstruction(object):

    def test_basic_construction(self):
        p = _make_sphere_particle()
        bem = BEMStatEig(p, nev = 5)
        assert bem.p is p
        assert bem.nev == 5
        assert bem.enei is None
        assert bem.mat is None

    def test_construction_with_default_nev(self):
        p = _make_sphere_particle()
        bem = BEMStatEig(p)
        assert bem.nev <= 20
        assert bem.nev > 0

    def test_construction_stores_eigenmodes(self):
        p = _make_sphere_particle()
        nev = 5
        bem = BEMStatEig(p, nev = nev)
        assert bem.ene.shape == (nev, nev)
        assert bem.ur.shape == (p.nfaces, nev)
        assert bem.ul.shape == (nev, p.nfaces)

    def test_construction_unit_matrix_shape(self):
        p = _make_sphere_particle()
        nev = 5
        bem = BEMStatEig(p, nev = nev)
        assert bem.unit.shape == (nev ** 2, p.np)

    def test_green_function_created(self):
        p = _make_sphere_particle()
        bem = BEMStatEig(p, nev = 5)
        assert bem.g is not None

    def test_repr(self):
        p = _make_sphere_particle()
        bem = BEMStatEig(p, nev = 5)
        r = repr(bem)
        assert 'BEMStatEig' in r
        assert 'nev=5' in r
        assert 'not initialized' in r

    def test_repr_initialized(self):
        p = _make_sphere_particle()
        bem = BEMStatEig(p, nev = 5, enei = 500.0)
        r = repr(bem)
        assert 'BEMStatEig' in r
        assert '500' in r

    def test_class_attributes(self):
        assert BEMStatEig.name == 'bemsolver'
        assert BEMStatEig.needs['sim'] == 'stat'
        assert BEMStatEig.needs['nev'] is True


# ---------------------------------------------------------------------------
# Tests -- init with wavelength
# ---------------------------------------------------------------------------


class TestBEMStatEigInit(object):

    def test_init_with_enei_in_constructor(self):
        p = _make_sphere_particle()
        bem = BEMStatEig(p, nev = 5, enei = 500.0)
        assert bem.enei == 500.0
        assert bem.mat is not None

    def test_init_via_call(self):
        p = _make_sphere_particle()
        bem = BEMStatEig(p, nev = 5)
        result = bem(600.0)
        assert result is bem
        assert bem.enei == 600.0
        assert bem.mat is not None

    def test_init_caches_wavelength(self):
        p = _make_sphere_particle()
        bem = BEMStatEig(p, nev = 5, enei = 500.0)
        mat_first = bem.mat.copy()
        bem(500.0)
        np.testing.assert_array_equal(bem.mat, mat_first)

    def test_init_updates_for_new_wavelength(self):
        from mnpbem.materials.eps_drude import EpsDrude
        sphere = trisphere(32, 10.0)
        eps = [EpsConst(1.0), EpsDrude.gold()]
        p = ComParticle(eps, [sphere], [[2, 1]])
        bem = BEMStatEig(p, nev = 5, enei = 400.0)
        mat_first = bem.mat.copy()
        bem(700.0)
        assert bem.enei == 700.0
        assert not np.allclose(bem.mat, mat_first)

    def test_mat_shape(self):
        p = _make_sphere_particle()
        bem = BEMStatEig(p, nev = 5, enei = 500.0)
        assert bem.mat.shape == (p.nfaces, p.nfaces)


# ---------------------------------------------------------------------------
# Tests -- eigenmode properties
# ---------------------------------------------------------------------------


class TestBEMStatEigEigenmodes(object):

    def test_eigenvalues_sorted_ascending(self):
        p = _make_sphere_particle()
        nev = 10
        bem = BEMStatEig(p, nev = nev)
        ene_diag = np.diag(bem.ene)
        for i in range(len(ene_diag) - 1):
            assert ene_diag[i].real <= ene_diag[i + 1].real + 1e-10

    def test_eigenvectors_finite(self):
        p = _make_sphere_particle()
        bem = BEMStatEig(p, nev = 5)
        assert np.all(np.isfinite(bem.ur))
        assert np.all(np.isfinite(bem.ul))

    def test_biorthogonality(self):
        p = _make_sphere_particle()
        nev = 5
        bem = BEMStatEig(p, nev = nev)
        overlap = bem.ul @ bem.ur  # (nev, nev)
        np.testing.assert_allclose(overlap, np.eye(nev), atol = 1e-8)

    def test_ene_is_diagonal(self):
        p = _make_sphere_particle()
        nev = 5
        bem = BEMStatEig(p, nev = nev)
        # off-diagonal elements should be zero
        off_diag = bem.ene - np.diag(np.diag(bem.ene))
        np.testing.assert_allclose(off_diag, 0, atol = 1e-15)

    def test_nev_clamped_for_small_particle(self):
        p = _make_sphere_particle(n_verts = 8)
        bem = BEMStatEig(p, nev = 1000)
        assert bem.nev <= p.nfaces - 1


# ---------------------------------------------------------------------------
# Tests -- solve
# ---------------------------------------------------------------------------


class TestBEMStatEigSolve(object):

    def test_solve_returns_compstruct_and_solver(self):
        p = _make_sphere_particle()
        bem = BEMStatEig(p, nev = 10)
        exc = _make_excitation(p, enei = 500.0)
        sig, solver = bem.solve(exc)
        assert isinstance(sig, CompStruct)
        assert solver is bem

    def test_solve_sets_enei(self):
        p = _make_sphere_particle()
        bem = BEMStatEig(p, nev = 10)
        exc = _make_excitation(p, enei = 500.0)
        bem.solve(exc)
        assert bem.enei == 500.0

    def test_solve_sig_shape(self):
        p = _make_sphere_particle()
        bem = BEMStatEig(p, nev = 10)
        exc = _make_excitation(p, enei = 500.0)
        sig, _ = bem.solve(exc)
        assert sig.sig.shape == (p.nfaces,)

    def test_solve_sig_finite(self):
        p = _make_sphere_particle()
        bem = BEMStatEig(p, nev = 10)
        exc = _make_excitation(p, enei = 500.0)
        sig, _ = bem.solve(exc)
        assert np.all(np.isfinite(sig.sig))

    def test_truediv_same_as_solve(self):
        p = _make_sphere_particle()
        bem = BEMStatEig(p, nev = 10)
        exc = _make_excitation(p, enei = 500.0)
        sig1, _ = bem.solve(exc)
        sig2, _ = bem.__truediv__(exc)
        np.testing.assert_allclose(sig1.sig, sig2.sig, atol = 1e-15)


# ---------------------------------------------------------------------------
# Tests -- agreement with BEMStat
# ---------------------------------------------------------------------------


class TestBEMStatEigVsBEMStat(object):

    def test_solve_agrees_with_bemstat(self):
        """BEMStatEig should approximate BEMStat result when enough
        eigenmodes are used."""
        p = _make_sphere_particle(n_verts = 32, diameter = 10.0)
        nev = min(p.nfaces - 2, 30)

        bem_direct = BEMStat(p)
        bem_eig = BEMStatEig(p, nev = nev)

        exc = _make_excitation(p, enei = 500.0)

        sig_direct, _ = bem_direct.solve(exc)
        sig_eig, _ = bem_eig.solve(exc)

        # With sufficient eigenmodes, results should be close
        # Tolerance is moderate because eigenmode truncation introduces error
        np.testing.assert_allclose(
            sig_eig.sig.real, sig_direct.sig.real,
            rtol = 0.1, atol = 1e-6
        )

    def test_more_eigenmodes_closer_to_direct(self):
        """Using more eigenmodes should reduce the error relative to
        the direct BEMStat solution."""
        p = _make_sphere_particle(n_verts = 32, diameter = 10.0)
        nfaces = p.nfaces

        bem_direct = BEMStat(p)
        exc = _make_excitation(p, enei = 500.0)
        sig_direct, _ = bem_direct.solve(exc)

        nev_small = min(5, nfaces - 2)
        nev_large = min(15, nfaces - 2)

        bem_small = BEMStatEig(p, nev = nev_small)
        bem_large = BEMStatEig(p, nev = nev_large)

        sig_small, _ = bem_small.solve(exc)
        sig_large, _ = bem_large.solve(exc)

        err_small = np.linalg.norm(sig_small.sig - sig_direct.sig)
        err_large = np.linalg.norm(sig_large.sig - sig_direct.sig)

        assert err_large <= err_small + 1e-10

    def test_solve_multiple_wavelengths(self):
        """BEMStatEig and BEMStat should agree at multiple wavelengths."""
        p = _make_sphere_particle(n_verts = 32, diameter = 10.0)
        nev = min(p.nfaces - 2, 30)

        bem_direct = BEMStat(p)
        bem_eig = BEMStatEig(p, nev = nev)

        for wl in [400.0, 500.0, 600.0, 700.0]:
            exc = _make_excitation(p, enei = wl)
            sig_direct, _ = bem_direct.solve(exc)
            sig_eig, _ = bem_eig.solve(exc)

            np.testing.assert_allclose(
                sig_eig.sig.real, sig_direct.sig.real,
                rtol = 0.15, atol = 1e-6,
                err_msg = 'Mismatch at wavelength {}'.format(wl)
            )


# ---------------------------------------------------------------------------
# Tests -- potential and field
# ---------------------------------------------------------------------------


class TestBEMStatEigPotentialField(object):

    def test_potential_returns_compstruct(self):
        p = _make_sphere_particle()
        bem = BEMStatEig(p, nev = 10)
        exc = _make_excitation(p, enei = 500.0)
        sig, _ = bem.solve(exc)
        pot = bem.potential(sig, 2)
        assert isinstance(pot, CompStruct)

    def test_mul_returns_compstruct(self):
        p = _make_sphere_particle()
        bem = BEMStatEig(p, nev = 10)
        exc = _make_excitation(p, enei = 500.0)
        sig, _ = bem.solve(exc)
        phi = bem * sig
        assert isinstance(phi, CompStruct)
        assert hasattr(phi, 'phi1')
        assert hasattr(phi, 'phi2')


# ---------------------------------------------------------------------------
# Tests -- decay rate computation
# ---------------------------------------------------------------------------


class TestBEMStatEigDecayRate(object):

    def test_absorption_finite(self):
        """Compute absorption using PlaneWaveStat with BEMStatEig
        and verify the result is finite and positive."""
        from mnpbem.simulation.planewave_stat import PlaneWaveStat

        p = _make_sphere_particle(n_verts = 32, diameter = 10.0)
        nev = min(p.nfaces - 2, 15)

        pol = np.array([1.0, 0.0, 0.0])
        pw = PlaneWaveStat(pol)

        enei = 500.0
        bem = BEMStatEig(p, nev = nev)
        exc = pw.potential(p, enei)
        sig, _ = bem.solve(exc)
        cabs = pw.absorption(sig)

        assert np.isfinite(cabs)
        assert cabs.real > 0

    def test_absorption_agrees_with_bemstat(self):
        """Absorption from BEMStatEig should be close to BEMStat result."""
        from mnpbem.simulation.planewave_stat import PlaneWaveStat

        p = _make_sphere_particle(n_verts = 32, diameter = 10.0)
        nev = min(p.nfaces - 2, 15)

        pol = np.array([1.0, 0.0, 0.0])
        pw = PlaneWaveStat(pol)
        enei = 500.0

        # direct solver
        bem_direct = BEMStat(p)
        exc = pw.potential(p, enei)
        sig_direct, _ = bem_direct.solve(exc)
        abs_direct = pw.absorption(sig_direct)

        # eigenmode solver
        bem_eig = BEMStatEig(p, nev = nev)
        sig_eig, _ = bem_eig.solve(exc)
        abs_eig = pw.absorption(sig_eig)

        np.testing.assert_allclose(
            abs_eig.real, abs_direct.real, rtol = 0.15
        )
