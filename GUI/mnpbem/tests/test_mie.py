import sys
import types

import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn, factorial

# ---------------------------------------------------------------------------
# Bypass the broken top-level mnpbem.__init__ so the mie subpackage
# (and the lightweight materials subpackage) can be imported in isolation.
# ---------------------------------------------------------------------------
if "mnpbem" not in sys.modules:
    _stub = types.ModuleType("mnpbem")
    _stub.__path__ = ["mnpbem"]
    sys.modules["mnpbem"] = _stub

from mnpbem.mie.spherical_harmonics import spharm, sphtable, vecspharm
from mnpbem.mie.mie_gans import MieGans
from mnpbem.mie.mie_stat import MieStat
from mnpbem.mie.mie_ret import MieRet, _riccatibessel, _miecoefficients
from mnpbem.mie.mie_solver import mie_solver
from mnpbem.materials.eps_const import EpsConst
from mnpbem.materials.eps_drude import EpsDrude


# ============================================================================
# Helpers -- simple dielectric functions for testing
# ============================================================================

def _scalar_epsin(enei):
    return (-10.0 + 0.5j)


def _scalar_epsout(enei):
    return 1.0


def _smart_epsin(enei):
    # Returns an array for array input, scalar for scalar input
    # This mimics MATLAB behavior where epsin(enei) returns same shape
    enei = np.asarray(enei, dtype=float)
    result = (-10.0 + 0.5j) * np.ones_like(enei, dtype=complex)
    if result.ndim == 0:
        return complex(result)
    return result


def _smart_epsout(enei):
    enei = np.asarray(enei, dtype=float)
    result = 1.0 * np.ones_like(enei, dtype=complex)
    if result.ndim == 0:
        return complex(result)
    return result


# ============================================================================
# sphtable
# ============================================================================

class TestSphtable(object):

    def test_z_mode_length(self):
        # For key='z', each l contributes 3 entries (m=-1,0,1) -> 3*lmax
        ltab, mtab = sphtable(5)
        assert len(ltab) == 15
        assert len(mtab) == 15

    def test_z_mode_content_lmax1(self):
        ltab, mtab = sphtable(1)
        np.testing.assert_array_equal(ltab, [1, 1, 1])
        np.testing.assert_array_equal(mtab, [-1, 0, 1])

    def test_z_mode_content_lmax3(self):
        ltab, mtab = sphtable(3)
        expected_l = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        expected_m = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
        np.testing.assert_array_equal(ltab, expected_l)
        np.testing.assert_array_equal(mtab, expected_m)

    def test_full_mode_length(self):
        # For key='full', each l contributes (2l+1) entries
        # lmax=3: 3 + 5 + 7 = 15
        ltab, mtab = sphtable(3, "full")
        assert len(ltab) == 15
        assert len(mtab) == 15

    def test_full_mode_content_lmax2(self):
        ltab, mtab = sphtable(2, "full")
        expected_l = [1, 1, 1, 2, 2, 2, 2, 2]
        expected_m = [-1, 0, 1, -2, -1, 0, 1, 2]
        np.testing.assert_array_equal(ltab, expected_l)
        np.testing.assert_array_equal(mtab, expected_m)

    def test_dtypes(self):
        ltab, mtab = sphtable(5)
        assert ltab.dtype == np.int64
        assert mtab.dtype == np.int64

    def test_lmax_large(self):
        ltab, mtab = sphtable(50, "full")
        # total entries = sum(2l+1, l=1..50) = 50*52 = 2600
        assert len(ltab) == 50 * 52


# ============================================================================
# spharm
# ============================================================================

class TestSpharm(object):

    def test_Y10_at_pole(self):
        # Y_1^0(theta=0, phi=0) = sqrt(3/(4*pi)) * cos(0) = sqrt(3/(4*pi))
        ltab = np.array([1])
        mtab = np.array([0])
        y = spharm(ltab, mtab, np.array([0.0]), np.array([0.0]))
        expected = np.sqrt(3.0 / (4 * np.pi))
        assert np.real(y[0, 0]) == pytest.approx(expected, rel=1e-10)
        assert np.imag(y[0, 0]) == pytest.approx(0.0, abs=1e-15)

    def test_Y10_at_equator(self):
        # Y_1^0(theta=pi/2, phi=0) = sqrt(3/(4*pi)) * cos(pi/2) ~ 0
        ltab = np.array([1])
        mtab = np.array([0])
        y = spharm(ltab, mtab, np.array([np.pi / 2]), np.array([0.0]))
        assert np.abs(y[0, 0]) == pytest.approx(0.0, abs=1e-14)

    def test_Y11_at_pole(self):
        # Y_1^1(theta=0, phi=0) = -sqrt(3/(8*pi)) * sin(0) * e^(i*0) = 0
        ltab = np.array([1])
        mtab = np.array([1])
        y = spharm(ltab, mtab, np.array([0.0]), np.array([0.0]))
        assert np.abs(y[0, 0]) == pytest.approx(0.0, abs=1e-14)

    def test_Y11_at_equator(self):
        # Y_1^1(theta=pi/2, phi=0)
        # = -sqrt(3/(8*pi)) * sin(pi/2) * exp(i*0) = -sqrt(3/(8*pi))
        ltab = np.array([1])
        mtab = np.array([1])
        y = spharm(ltab, mtab, np.array([np.pi / 2]), np.array([0.0]))
        expected = -np.sqrt(3.0 / (8 * np.pi))
        assert np.real(y[0, 0]) == pytest.approx(expected, rel=1e-10)

    def test_negative_m_conjugate_relation(self):
        # Y_l^{-m} = (-1)^m * conj(Y_l^m)
        theta = np.array([0.7])
        phi = np.array([1.3])
        l_val = 3
        m_val = 2
        y_pos = spharm(np.array([l_val]), np.array([m_val]), theta, phi)
        y_neg = spharm(np.array([l_val]), np.array([-m_val]), theta, phi)
        expected = (-1) ** m_val * np.conj(y_pos[0, 0])
        assert np.real(y_neg[0, 0]) == pytest.approx(np.real(expected), rel=1e-10)
        assert np.imag(y_neg[0, 0]) == pytest.approx(np.imag(expected), rel=1e-10)

    def test_normalization(self):
        # For a single (l, m), integration of |Y|^2 over the sphere = 1
        # Use midpoint-rule numerical integration on a grid
        # spharm expects theta and phi arrays of the same length (paired coordinates)
        n_theta = 200
        n_phi = 400
        theta_edges = np.linspace(0, np.pi, n_theta + 1)
        phi_edges = np.linspace(0, 2 * np.pi, n_phi + 1)

        dtheta = theta_edges[1] - theta_edges[0]
        dphi = phi_edges[1] - phi_edges[0]
        theta_mid = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        phi_mid = 0.5 * (phi_edges[:-1] + phi_edges[1:])

        l_val = 2
        m_val = 1
        # Build a full grid of (theta, phi) pairs
        theta_grid, phi_grid = np.meshgrid(theta_mid, phi_mid, indexing="ij")
        theta_flat = theta_grid.ravel()
        phi_flat = phi_grid.ravel()

        y = spharm(np.array([l_val]), np.array([m_val]), theta_flat, phi_flat)
        # y shape: (1, n_theta * n_phi)
        y_grid = np.abs(y[0, :]).reshape(n_theta, n_phi) ** 2

        # Integrate: sum over phi (constant dphi), then sum over theta with sin(theta)*dtheta
        integral = np.sum(y_grid * np.sin(theta_mid)[:, np.newaxis]) * dtheta * dphi
        assert integral == pytest.approx(1.0, rel=1e-3)

    def test_output_shape(self):
        ltab = np.array([1, 1, 2])
        mtab = np.array([0, 1, -1])
        theta = np.array([0.3, 0.7, 1.2, 1.8])
        phi = np.array([0.1, 0.5, 1.0, 2.0])
        y = spharm(ltab, mtab, theta, phi)
        assert y.shape == (3, 4)

    def test_multiple_l_values(self):
        ltab, mtab = sphtable(3)
        theta = np.array([0.5])
        phi = np.array([0.0])
        y = spharm(ltab, mtab, theta, phi)
        assert y.shape == (9, 1)
        # All values should be finite
        assert np.all(np.isfinite(y))


# ============================================================================
# vecspharm
# ============================================================================

class TestVecspharm(object):

    def test_output_shapes(self):
        ltab, mtab = sphtable(3)
        theta = np.array([0.5, 1.0])
        phi = np.array([0.0, np.pi / 4])
        x, y = vecspharm(ltab, mtab, theta, phi)
        assert x.shape == (9, 2, 3)
        assert y.shape == (9, 2)

    def test_y_equals_spharm(self):
        ltab, mtab = sphtable(2)
        theta = np.array([0.3, 0.8])
        phi = np.array([0.1, 1.5])
        x, y = vecspharm(ltab, mtab, theta, phi)
        y_direct = spharm(ltab, mtab, theta, phi)
        np.testing.assert_allclose(y, y_direct, atol=1e-14)

    def test_finite_output(self):
        ltab, mtab = sphtable(5)
        theta = np.array([0.1, 0.5, 1.0, 1.5])
        phi = np.array([0.0, 0.5, 1.0, 2.0])
        x, y = vecspharm(ltab, mtab, theta, phi)
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))

    def test_single_angle(self):
        ltab, mtab = sphtable(2)
        theta = np.array([0.5])
        phi = np.array([0.0])
        x, y = vecspharm(ltab, mtab, theta, phi)
        assert x.shape == (6, 1, 3)
        assert y.shape == (6, 1)


# ============================================================================
# Riccati-Bessel functions
# ============================================================================

class TestRiccatiBessel(object):

    def test_j_against_scipy(self):
        # j_l(z) for real z should agree with scipy spherical_jn
        z = 5.0
        l_arr = np.array([1, 2, 3, 4, 5])
        j, h, zjp, zhp = _riccatibessel(z, l_arr)
        for idx, l_val in enumerate(l_arr):
            expected = spherical_jn(l_val, z)
            assert np.real(j[idx]) == pytest.approx(expected, rel=1e-10)
            assert abs(np.imag(j[idx])) < 1e-14

    def test_h_definition(self):
        # h_l = j_l + i*y_l for real z
        z = 3.0
        l_arr = np.array([1, 2, 3])
        j, h, zjp, zhp = _riccatibessel(z, l_arr)
        for idx, l_val in enumerate(l_arr):
            j_scipy = spherical_jn(l_val, z)
            y_scipy = spherical_yn(l_val, z)
            h_expected = j_scipy + 1j * y_scipy
            assert np.real(h[idx]) == pytest.approx(np.real(h_expected), rel=1e-10)
            assert np.imag(h[idx]) == pytest.approx(np.imag(h_expected), rel=1e-10)

    def test_derivative_numerical(self):
        # Check [z*j_l(z)]' numerically via finite differences
        z = 4.0
        l_arr = np.array([1, 2])
        eps = 1e-7

        _, _, zjp, _ = _riccatibessel(z, l_arr)

        # Numerical derivative of z * j_l(z)
        for idx, l_val in enumerate(l_arr):
            jh, _, _, _ = _riccatibessel(z + eps, l_arr)
            jl, _, _, _ = _riccatibessel(z - eps, l_arr)
            # z * j(z) at z +/- eps
            f_plus = (z + eps) * jh[idx]
            f_minus = (z - eps) * jl[idx]
            num_deriv = (f_plus - f_minus) / (2 * eps)
            assert np.real(zjp[idx]) == pytest.approx(np.real(num_deriv), rel=1e-5)

    def test_table_assignment(self):
        # When ltab has repeated values, output should be indexable correctly
        ltab = np.array([1, 1, 2, 2, 3])
        z = 5.0
        j, h, zjp, zhp = _riccatibessel(z, ltab)
        assert j[0] == j[1]
        assert j[2] == j[3]
        assert j[0] != j[4]

    def test_complex_argument(self):
        # Should handle complex z without errors
        z = 3.0 + 1.5j
        l_arr = np.array([1, 2, 3])
        j, h, zjp, zhp = _riccatibessel(z, l_arr)
        assert np.all(np.isfinite(j))
        assert np.all(np.isfinite(h))
        assert np.all(np.isfinite(zjp))
        assert np.all(np.isfinite(zhp))


# ============================================================================
# Mie coefficients
# ============================================================================

class TestMieCoefficients(object):

    def test_perfect_conductor_a(self):
        # For a perfect electric conductor (|epsr| -> inf), a_l -> 1 asymptotically
        # Use large but finite epsr to approximate
        k = 0.01
        diameter = 100.0
        epsr = -1e6 + 0j
        mur = 1.0
        ltab = np.array([1, 2])
        a, b, c, d = _miecoefficients(k, diameter, epsr, mur, ltab)
        # For very large |epsr|, |a| should be close to a finite value
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(b))

    def test_nonmagnetic_mur1(self):
        # Standard Mie coefficients with mur=1 should be finite
        k = 0.05
        diameter = 20.0
        epsr = -10.0 + 0.5j
        ltab = np.array([1, 2, 3, 4, 5])
        a, b, c, d = _miecoefficients(k, diameter, epsr, 1.0, ltab)
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(c))
        assert np.all(np.isfinite(d))

    def test_output_shapes(self):
        k = 0.05
        diameter = 20.0
        epsr = -10.0 + 0.5j
        ltab = np.array([1, 1, 2, 2, 3])
        a, b, c, d = _miecoefficients(k, diameter, epsr, 1.0, ltab)
        assert a.shape == (5,)
        assert b.shape == (5,)
        assert c.shape == (5,)
        assert d.shape == (5,)

    def test_repeated_ltab(self):
        # Repeated l values should give identical Mie coefficients
        k = 0.05
        diameter = 20.0
        epsr = -10.0 + 0.5j
        ltab = np.array([1, 1, 2, 2, 3])
        a, b, c, d = _miecoefficients(k, diameter, epsr, 1.0, ltab)
        assert a[0] == a[1]
        assert a[2] == a[3]


# ============================================================================
# MieGans
# ============================================================================

class TestMieGans(object):

    def test_sphere_depolarization_factors(self):
        # For a sphere (equal axes), all depolarization factors = 1/3
        mie = MieGans(_scalar_epsin, _scalar_epsout, np.array([10.0, 10.0, 10.0]))
        assert mie._L1 == pytest.approx(1.0 / 3, rel=1e-4)
        assert mie._L2 == pytest.approx(1.0 / 3, rel=1e-4)
        assert mie._L3 == pytest.approx(1.0 / 3, rel=1e-4)

    def test_depolarization_sum_to_one(self):
        # L1 + L2 + L3 = 1 for any ellipsoid
        mie = MieGans(_scalar_epsin, _scalar_epsout, np.array([5.0, 10.0, 20.0]))
        total = mie._L1 + mie._L2 + mie._L3
        assert total == pytest.approx(1.0, rel=1e-4)

    def test_prolate_depolarization(self):
        # For a prolate spheroid (a = b < c), L3 < L1 = L2
        mie = MieGans(_scalar_epsin, _scalar_epsout, np.array([10.0, 10.0, 40.0]))
        assert mie._L1 == pytest.approx(mie._L2, rel=1e-4)
        assert mie._L3 < mie._L1

    def test_extinction_equals_sca_plus_abs(self):
        mie = MieGans(_scalar_epsin, _scalar_epsout, np.array([10.0, 10.0, 10.0]))
        enei = np.array([500.0])
        pol = np.array([1.0, 0.0, 0.0])
        ext = mie.extinction(enei, pol)
        sca = mie.scattering(enei, pol)
        abso = mie.absorption(enei, pol)
        assert ext == pytest.approx(sca + abso, rel=1e-10)

    def test_sphere_polarization_independence(self):
        # For a sphere, cross sections should be the same regardless of polarization
        mie = MieGans(_scalar_epsin, _scalar_epsout, np.array([10.0, 10.0, 10.0]))
        enei = np.array([500.0])
        ext_x = mie.extinction(enei, np.array([1.0, 0.0, 0.0]))
        ext_y = mie.extinction(enei, np.array([0.0, 1.0, 0.0]))
        ext_z = mie.extinction(enei, np.array([0.0, 0.0, 1.0]))
        assert ext_x == pytest.approx(ext_y, rel=1e-4)
        assert ext_y == pytest.approx(ext_z, rel=1e-4)

    def test_ellipsoid_polarization_dependence(self):
        # For an ellipsoid, different polarizations give different cross sections
        mie = MieGans(_scalar_epsin, _scalar_epsout, np.array([5.0, 10.0, 20.0]))
        enei = np.array([500.0])
        ext_x = mie.extinction(enei, np.array([1.0, 0.0, 0.0]))
        ext_z = mie.extinction(enei, np.array([0.0, 0.0, 1.0]))
        # They should differ significantly for an asymmetric ellipsoid
        assert ext_x != pytest.approx(ext_z, rel=0.1)

    def test_positive_cross_sections(self):
        mie = MieGans(_scalar_epsin, _scalar_epsout, np.array([10.0, 10.0, 10.0]))
        enei = np.array([500.0])
        pol = np.array([1.0, 0.0, 0.0])
        assert mie.scattering(enei, pol) > 0
        assert mie.absorption(enei, pol) > 0
        assert mie.extinction(enei, pol) > 0

    def test_repr(self):
        mie = MieGans(_scalar_epsin, _scalar_epsout, np.array([10.0, 10.0, 10.0]))
        r = repr(mie)
        assert "MieGans" in r

    def test_multiple_wavelengths(self):
        mie = MieGans(_smart_epsin, _smart_epsout, np.array([10.0, 10.0, 10.0]))
        enei = np.array([400.0, 500.0, 600.0])
        pol = np.array([1.0, 0.0, 0.0])
        ext = mie.extinction(enei, pol)
        assert ext.shape == (3,)
        assert np.all(np.isfinite(ext))


# ============================================================================
# MieStat
# ============================================================================

class TestMieStat(object):

    def test_extinction_equals_sca_plus_abs(self):
        mie = MieStat(_scalar_epsin, _scalar_epsout, 10.0)
        enei = np.array([500.0])
        ext = mie.extinction(enei)
        sca = mie.scattering(enei)
        abso = mie.absorption(enei)
        assert ext[0] == pytest.approx(sca[0] + abso[0], rel=1e-10)

    def test_positive_cross_sections(self):
        mie = MieStat(_scalar_epsin, _scalar_epsout, 10.0)
        enei = np.array([500.0])
        assert mie.scattering(enei)[0] > 0
        assert mie.absorption(enei)[0] > 0
        assert mie.extinction(enei)[0] > 0

    def test_cross_section_scales_with_size(self):
        # Absorption ~ r^3, scattering ~ r^6 in quasistatic limit
        mie_small = MieStat(_scalar_epsin, _scalar_epsout, 10.0)
        mie_large = MieStat(_scalar_epsin, _scalar_epsout, 20.0)
        enei = np.array([500.0])

        # absorption should scale as (diameter/2)^3 ~ diameter^3
        ratio_abs = mie_large.absorption(enei)[0] / mie_small.absorption(enei)[0]
        assert ratio_abs == pytest.approx(8.0, rel=1e-3)  # (20/10)^3 = 8

        # scattering should scale as (diameter/2)^6 ~ diameter^6
        ratio_sca = mie_large.scattering(enei)[0] / mie_small.scattering(enei)[0]
        assert ratio_sca == pytest.approx(64.0, rel=1e-3)  # (20/10)^6 = 64

    def test_multiple_wavelengths(self):
        mie = MieStat(_smart_epsin, _smart_epsout, 10.0)
        enei = np.array([400.0, 500.0, 600.0])
        ext = mie.extinction(enei)
        assert ext.shape == (3,)
        assert np.all(np.isfinite(ext))

    def test_gold_drude_resonance(self):
        # Gold Drude nanosphere should have a plasmon resonance
        epsin = EpsDrude.gold()
        epsout = EpsConst(1.0)
        mie = MieStat(epsin, epsout, 20.0)
        enei = np.linspace(400, 700, 100)
        ext = mie.extinction(enei)
        # Find the peak wavelength
        peak_idx = np.argmax(ext)
        peak_wavelength = enei[peak_idx]
        # Gold plasmon resonance should be around 500-550 nm for Drude model
        assert 450 < peak_wavelength < 580

    def test_matches_mie_gans_sphere(self):
        # For a sphere, MieStat and MieGans (with equal axes) should give
        # similar absorption (absorption in quasistatic limit is the same formula)
        diameter = 10.0
        ax = np.array([diameter, diameter, diameter])
        mie_gans = MieGans(_scalar_epsin, _scalar_epsout, ax)
        mie_stat = MieStat(_scalar_epsin, _scalar_epsout, diameter)
        enei = np.array([500.0])

        # MieGans absorption with pol=[1,0,0] should match MieStat absorption
        abs_gans = mie_gans.absorption(enei, np.array([1.0, 0.0, 0.0]))
        abs_stat = mie_stat.absorption(enei)
        assert abs_gans == pytest.approx(abs_stat[0], rel=1e-3)

    def test_decayrate_output_shape(self):
        mie = MieStat(_scalar_epsin, _scalar_epsout, 10.0)
        z = np.array([10.0, 15.0, 20.0])
        tot, rad = mie.decayrate(500.0, z)
        assert tot.shape == (3, 2)
        assert rad.shape == (3, 2)

    def test_decayrate_radiative_positive(self):
        mie = MieStat(_scalar_epsin, _scalar_epsout, 10.0)
        z = np.array([10.0, 20.0])
        tot, rad = mie.decayrate(500.0, z)
        assert np.all(rad > 0)

    def test_decayrate_tot_geq_rad(self):
        # Total decay rate >= radiative decay rate (nonradiative >= 0)
        mie = MieStat(_scalar_epsin, _scalar_epsout, 10.0)
        z = np.array([10.0, 20.0])
        tot, rad = mie.decayrate(500.0, z)
        # This should hold for lossy particles
        assert np.all(tot >= rad - 1e-10)

    def test_decayrate_approaches_1_far_away(self):
        # Far from the sphere, the decay rate should approach free-space (1)
        mie = MieStat(_scalar_epsin, _scalar_epsout, 10.0)
        z_far = np.array([1000.0])
        tot, rad = mie.decayrate(500.0, z_far)
        # Both x and z orientations
        assert tot[0, 0] == pytest.approx(1.0, abs=0.1)
        assert tot[0, 1] == pytest.approx(1.0, abs=0.1)
        assert rad[0, 0] == pytest.approx(1.0, abs=0.1)
        assert rad[0, 1] == pytest.approx(1.0, abs=0.1)

    def test_lmax_parameter(self):
        mie10 = MieStat(_scalar_epsin, _scalar_epsout, 10.0, lmax=10)
        mie30 = MieStat(_scalar_epsin, _scalar_epsout, 10.0, lmax=30)
        # Both should give similar extinction (dipolar mode dominates for small particles)
        enei = np.array([500.0])
        ext10 = mie10.extinction(enei)
        ext30 = mie30.extinction(enei)
        assert ext10[0] == pytest.approx(ext30[0], rel=1e-6)

    def test_repr(self):
        mie = MieStat(_scalar_epsin, _scalar_epsout, 10.0)
        r = repr(mie)
        assert "MieStat" in r
        assert "10" in r


# ============================================================================
# MieRet
# ============================================================================

class TestMieRet(object):

    def test_extinction_equals_sca_plus_abs(self):
        mie = MieRet(_smart_epsin, _smart_epsout, 10.0)
        enei = np.array([500.0])
        ext = mie.extinction(enei)
        sca = mie.scattering(enei)
        abso = mie.absorption(enei)
        assert ext[0] == pytest.approx(sca[0] + abso[0], rel=1e-10)

    def test_positive_cross_sections(self):
        mie = MieRet(_smart_epsin, _smart_epsout, 10.0)
        enei = np.array([500.0])
        assert mie.scattering(enei)[0] > 0
        assert mie.absorption(enei)[0] > 0
        assert mie.extinction(enei)[0] > 0

    def test_multiple_wavelengths(self):
        mie = MieRet(_smart_epsin, _smart_epsout, 10.0)
        enei = np.array([400.0, 500.0, 600.0])
        ext = mie.extinction(enei)
        sca = mie.scattering(enei)
        abso = mie.absorption(enei)
        assert ext.shape == (3,)
        for i in range(3):
            assert ext[i] == pytest.approx(sca[i] + abso[i], rel=1e-10)

    def test_gold_drude_resonance(self):
        epsin = EpsDrude.gold()
        epsout = EpsConst(1.0)
        mie = MieRet(epsin, epsout, 20.0)
        enei = np.linspace(400, 700, 100)
        ext = mie.extinction(enei)
        peak_idx = np.argmax(ext)
        peak_wavelength = enei[peak_idx]
        # Should be similar to MieStat peak for small particles
        assert 450 < peak_wavelength < 600

    def test_large_particle_retardation_shift(self):
        # For larger particles, MieRet resonance should red-shift compared to MieStat
        epsin = EpsDrude.gold()
        epsout = EpsConst(1.0)
        enei = np.linspace(400, 800, 200)

        mie_stat = MieStat(epsin, epsout, 80.0)
        mie_ret = MieRet(epsin, epsout, 80.0)

        ext_stat = mie_stat.extinction(enei)
        ext_ret = mie_ret.extinction(enei)

        peak_stat = enei[np.argmax(ext_stat)]
        peak_ret = enei[np.argmax(ext_ret)]

        # Retardation red-shifts the resonance
        assert peak_ret >= peak_stat - 5  # allow small tolerance

    def test_decayrate_output_shape(self):
        mie = MieRet(_smart_epsin, _smart_epsout, 10.0)
        z = np.array([10.0, 15.0])
        tot, rad = mie.decayrate(500.0, z)
        assert tot.shape == (2, 2)
        assert rad.shape == (2, 2)

    def test_decayrate_radiative_positive(self):
        mie = MieRet(_smart_epsin, _smart_epsout, 10.0)
        z = np.array([10.0, 20.0])
        tot, rad = mie.decayrate(500.0, z)
        assert np.all(rad > 0)

    def test_purcell_factor_near_surface(self):
        # Close to a metallic sphere, the Purcell factor (tot decay rate / free-space)
        # should be > 1 due to enhanced local density of states
        mie = MieRet(_smart_epsin, _smart_epsout, 10.0)
        # z = 6 nm from center (just 1 nm above surface of r=5nm sphere)
        z = np.array([6.0])
        tot, rad = mie.decayrate(500.0, z)
        # At least one orientation should show enhancement
        assert np.max(tot) > 1.0

    def test_decayrate_approaches_1_far_away(self):
        mie = MieRet(_smart_epsin, _smart_epsout, 10.0)
        z_far = np.array([5000.0])
        tot, rad = mie.decayrate(500.0, z_far)
        assert tot[0, 0] == pytest.approx(1.0, abs=0.1)
        assert tot[0, 1] == pytest.approx(1.0, abs=0.1)

    def test_lmax_parameter(self):
        mie10 = MieRet(_smart_epsin, _smart_epsout, 10.0, lmax=10)
        mie30 = MieRet(_smart_epsin, _smart_epsout, 10.0, lmax=30)
        enei = np.array([500.0])
        ext10 = mie10.extinction(enei)
        ext30 = mie30.extinction(enei)
        # For small particles, low lmax is sufficient
        assert ext10[0] == pytest.approx(ext30[0], rel=1e-4)

    def test_repr(self):
        mie = MieRet(_smart_epsin, _smart_epsout, 10.0)
        r = repr(mie)
        assert "MieRet" in r
        assert "10" in r


# ============================================================================
# MieStat vs MieRet: quasistatic limit
# ============================================================================

class TestQuasistaticLimit(object):

    def test_small_particle_extinction(self):
        # For very small particles, MieRet should agree with MieStat
        diameter = 2.0  # 2 nm
        mie_s = MieStat(_smart_epsin, _smart_epsout, diameter)
        mie_r = MieRet(_smart_epsin, _smart_epsout, diameter)
        enei = np.array([500.0])
        ext_s = mie_s.extinction(enei)
        ext_r = mie_r.extinction(enei)
        assert ext_r[0] == pytest.approx(ext_s[0], rel=1e-3)

    def test_small_particle_absorption(self):
        diameter = 2.0
        mie_s = MieStat(_smart_epsin, _smart_epsout, diameter)
        mie_r = MieRet(_smart_epsin, _smart_epsout, diameter)
        enei = np.array([500.0])
        abs_s = mie_s.absorption(enei)
        abs_r = mie_r.absorption(enei)
        assert abs_r[0] == pytest.approx(abs_s[0], rel=1e-3)

    def test_small_particle_scattering(self):
        diameter = 2.0
        mie_s = MieStat(_smart_epsin, _smart_epsout, diameter)
        mie_r = MieRet(_smart_epsin, _smart_epsout, diameter)
        enei = np.array([500.0])
        sca_s = mie_s.scattering(enei)
        sca_r = mie_r.scattering(enei)
        assert sca_r[0] == pytest.approx(sca_s[0], rel=1e-3)

    def test_small_particle_multiple_wavelengths(self):
        diameter = 2.0
        mie_s = MieStat(_smart_epsin, _smart_epsout, diameter)
        mie_r = MieRet(_smart_epsin, _smart_epsout, diameter)
        enei = np.array([400.0, 500.0, 600.0, 700.0])
        ext_s = mie_s.extinction(enei)
        ext_r = mie_r.extinction(enei)
        for i in range(len(enei)):
            assert ext_r[i] == pytest.approx(ext_s[i], rel=1e-3)


# ============================================================================
# mie_solver factory
# ============================================================================

class TestMieSolver(object):

    def test_stat_returns_miestat(self):
        mie = mie_solver(_scalar_epsin, _scalar_epsout, 10.0, sim="stat")
        assert isinstance(mie, MieStat)

    def test_ret_returns_mieret(self):
        mie = mie_solver(_smart_epsin, _smart_epsout, 10.0, sim="ret")
        assert isinstance(mie, MieRet)

    def test_invalid_sim_raises(self):
        with pytest.raises(ValueError):
            mie_solver(_scalar_epsin, _scalar_epsout, 10.0, sim="invalid")

    def test_default_is_stat(self):
        mie = mie_solver(_scalar_epsin, _scalar_epsout, 10.0)
        assert isinstance(mie, MieStat)

    def test_lmax_forwarded(self):
        mie = mie_solver(_scalar_epsin, _scalar_epsout, 10.0, sim="stat", lmax=30)
        assert isinstance(mie, MieStat)
        # Check that lmax was applied (ltab should have 30*3=90 entries)
        assert len(mie._ltab) == 90

    def test_lmax_forwarded_ret(self):
        mie = mie_solver(_smart_epsin, _smart_epsout, 10.0, sim="ret", lmax=15)
        assert isinstance(mie, MieRet)
        assert len(mie._ltab) == 45

    def test_functional_consistency(self):
        # mie_solver should give the same results as direct construction
        enei = np.array([500.0])

        mie1 = mie_solver(_smart_epsin, _smart_epsout, 10.0, sim="ret", lmax=20)
        mie2 = MieRet(_smart_epsin, _smart_epsout, 10.0, lmax=20)
        ext1 = mie1.extinction(enei)
        ext2 = mie2.extinction(enei)
        assert ext1[0] == pytest.approx(ext2[0], rel=1e-12)


# ============================================================================
# MieStat.loss (EELS)
# ============================================================================

class TestMieStatLoss(object):

    def test_output_shape(self):
        mie = MieStat(_smart_epsin, _smart_epsout, 10.0)
        b = np.array([1.0, 2.0, 5.0])
        enei = np.array([400.0, 500.0])
        prob = mie.loss(b, enei)
        assert prob.shape == (3, 2)

    def test_positive_loss(self):
        mie = MieStat(_smart_epsin, _smart_epsout, 10.0)
        b = np.array([1.0, 5.0])
        enei = np.array([500.0])
        prob = mie.loss(b, enei)
        assert np.all(prob > 0)

    def test_loss_decreases_with_distance(self):
        mie = MieStat(_smart_epsin, _smart_epsout, 10.0)
        b = np.array([1.0, 5.0, 10.0, 20.0])
        enei = np.array([500.0])
        prob = mie.loss(b, enei)
        # Loss should decrease with increasing impact parameter
        for i in range(len(b) - 1):
            assert prob[i, 0] > prob[i + 1, 0]

    def test_negative_impact_parameter_raises(self):
        mie = MieStat(_scalar_epsin, _scalar_epsout, 10.0)
        with pytest.raises(AssertionError):
            mie.loss(np.array([-1.0]), np.array([500.0]))


# ============================================================================
# MieRet.loss (EELS)
# ============================================================================

class TestMieRetLoss(object):

    def test_output_shapes(self):
        mie = MieRet(_smart_epsin, _smart_epsout, 10.0)
        b = np.array([1.0, 5.0])
        enei = np.array([400.0, 500.0])
        prob, prad = mie.loss(b, enei)
        assert prob.shape == (2, 2)
        assert prad.shape == (2, 2)

    def test_positive_loss(self):
        mie = MieRet(_smart_epsin, _smart_epsout, 10.0)
        b = np.array([1.0, 5.0])
        enei = np.array([500.0])
        prob, prad = mie.loss(b, enei)
        assert np.all(prob > 0)
        assert np.all(prad >= 0)

    def test_loss_decreases_with_distance(self):
        mie = MieRet(_smart_epsin, _smart_epsout, 10.0)
        b = np.array([1.0, 5.0, 10.0, 20.0])
        enei = np.array([500.0])
        prob, prad = mie.loss(b, enei)
        for i in range(len(b) - 1):
            assert prob[i, 0] > prob[i + 1, 0]

    def test_negative_impact_parameter_raises(self):
        mie = MieRet(_smart_epsin, _smart_epsout, 10.0)
        with pytest.raises(AssertionError):
            mie.loss(np.array([-1.0]), np.array([500.0]))


# ============================================================================
# MieGans agreement with MieStat for spherical geometry
# ============================================================================

class TestMieGansVsMieStat(object):

    def test_absorption_sphere_agreement(self):
        # MieGans with equal axes should give the same absorption as MieStat
        diameter = 10.0
        ax = np.array([diameter, diameter, diameter])
        mie_g = MieGans(_scalar_epsin, _scalar_epsout, ax)
        mie_s = MieStat(_scalar_epsin, _scalar_epsout, diameter)
        enei = np.array([500.0])
        pol = np.array([1.0, 0.0, 0.0])

        abs_g = mie_g.absorption(enei, pol)
        abs_s = mie_s.absorption(enei)
        assert abs_g == pytest.approx(abs_s[0], rel=1e-3)

    def test_scattering_sphere_agreement(self):
        diameter = 10.0
        ax = np.array([diameter, diameter, diameter])
        mie_g = MieGans(_scalar_epsin, _scalar_epsout, ax)
        mie_s = MieStat(_scalar_epsin, _scalar_epsout, diameter)
        enei = np.array([500.0])
        pol = np.array([1.0, 0.0, 0.0])

        sca_g = mie_g.scattering(enei, pol)
        sca_s = mie_s.scattering(enei)
        assert sca_g == pytest.approx(sca_s[0], rel=1e-3)


# ============================================================================
# Edge cases and robustness
# ============================================================================

class TestEdgeCases(object):

    def test_miegans_with_tuple_dielectric(self):
        # Dielectric function that returns a tuple (eps, k)
        def epsin_tuple(enei):
            enei = np.asarray(enei, dtype=float)
            eps = -10.0 + 0.5j
            k = 2 * np.pi / enei * np.sqrt(eps) if np.any(enei > 0) else 0.0
            return (eps, k)

        def epsout_tuple(enei):
            return (1.0, 2 * np.pi / enei if np.any(np.asarray(enei) > 0) else 0.0)

        mie = MieGans(epsin_tuple, epsout_tuple, np.array([10.0, 10.0, 10.0]))
        enei = np.array([500.0])
        pol = np.array([1.0, 0.0, 0.0])
        ext = mie.extinction(enei, pol)
        assert np.isfinite(ext)
        assert ext > 0

    def test_mieret_single_wavelength(self):
        mie = MieRet(_smart_epsin, _smart_epsout, 10.0)
        enei = np.array([500.0])
        ext = mie.extinction(enei)
        assert ext.shape == (1,)
        assert ext[0] > 0

    def test_spharm_out_of_range_m(self):
        # m > l should give zero
        ltab = np.array([1])
        mtab = np.array([5])  # m=5 > l=1
        theta = np.array([0.5])
        phi = np.array([0.3])
        y = spharm(ltab, mtab, theta, phi)
        assert np.abs(y[0, 0]) == pytest.approx(0.0, abs=1e-14)

    def test_mieret_decayrate_single_enei_assertion(self):
        mie = MieRet(_smart_epsin, _smart_epsout, 10.0)
        # Should raise assertion for multiple enei values
        with pytest.raises(AssertionError):
            mie.decayrate(np.array([400.0, 500.0]), np.array([10.0]))

    def test_miestat_decayrate_single_enei_assertion(self):
        mie = MieStat(_scalar_epsin, _scalar_epsout, 10.0)
        with pytest.raises(AssertionError):
            mie.decayrate(np.array([400.0, 500.0]), np.array([10.0]))


# ============================================================================
# Cross-check Mie coefficients with analytical limits
# ============================================================================

class TestMieAnalytical(object):

    def test_small_particle_a1_dominates(self):
        # For very small particles (x = k*a << 1), only l=1 coefficients matter
        k = 2 * np.pi / 500.0  # vacuum
        diameter = 2.0  # 2 nm sphere, x = k*a = 0.013 << 1
        epsr = -10.0 + 0.5j
        ltab = np.array([1, 2, 3, 4, 5])
        a, b, c, d = _miecoefficients(k, diameter, epsr, 1.0, ltab)
        # a[0] (l=1) should be much larger than higher orders
        assert np.abs(a[0]) > 10 * np.abs(a[1])
        assert np.abs(a[0]) > 100 * np.abs(a[2])

    def test_extinction_efficiency_reasonable(self):
        # Extinction efficiency Q_ext = C_ext / (pi * a^2) should be O(1) for resonance
        epsin = EpsDrude.gold()
        epsout = EpsConst(1.0)
        diameter = 50.0
        mie = MieRet(epsin, epsout, diameter)
        enei = np.linspace(400, 800, 200)
        ext = mie.extinction(enei)
        a = diameter / 2
        Q_ext = ext / (np.pi * a ** 2)
        # Max Q_ext should be in a physically reasonable range (0.1 to 20)
        assert 0.01 < np.max(Q_ext) < 50.0

    def test_optical_theorem_mieret(self):
        # The optical theorem states that ext = sca + abs.
        # This is already tested above, but here we test with EpsDrude gold
        # over many wavelengths for thoroughness.
        epsin = EpsDrude.gold()
        epsout = EpsConst(1.0)
        mie = MieRet(epsin, epsout, 50.0)
        enei = np.linspace(400, 800, 50)
        ext = mie.extinction(enei)
        sca = mie.scattering(enei)
        abso = mie.absorption(enei)
        np.testing.assert_allclose(ext, sca + abso, rtol=1e-10)

    def test_nonabsorbing_sphere_abs_zero(self):
        # A dielectric sphere with real eps should have abs ~ 0
        def epsin_real(enei):
            enei = np.asarray(enei, dtype=float)
            result = (4.0 + 0j) * np.ones_like(enei, dtype=complex)
            if result.ndim == 0:
                return complex(result)
            return result

        mie = MieRet(epsin_real, _smart_epsout, 50.0)
        enei = np.array([500.0])
        abso = mie.absorption(enei)
        sca = mie.scattering(enei)
        # Absorption should be essentially zero for a lossless dielectric
        assert np.abs(abso[0]) < 1e-10 * sca[0]
