"""
Tests for DipoleRet and PlaneWaveRet scattering/extinction methods.

Verifies that SpectrumRet is properly wired into the scattering and
extinction computations for both dipole and plane-wave excitations.
"""

import sys
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Bypass the broken top-level mnpbem.__init__ so subpackages can be
# imported in isolation.
# ---------------------------------------------------------------------------
if "mnpbem" not in sys.modules:
    _stub = types.ModuleType("mnpbem")
    _stub.__path__ = ["mnpbem"]
    sys.modules["mnpbem"] = _stub

from mnpbem.materials.eps_const import EpsConst
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.greenfun import CompStruct
from mnpbem.simulation.dipole_ret import DipoleRet
from mnpbem.simulation.planewave_ret import PlaneWaveRet
from mnpbem.spectrum import SpectrumRet


# ============================================================================
# Helpers
# ============================================================================

class MockComPoint(object):
    """
    Minimal mock for a ComPoint-like dipole position container.

    Provides the attributes DipoleRet needs: pos, n, eps, index, inout, eps1.
    """

    def __init__(self, pos, eps_list, medium = 1):
        self.pos = np.atleast_2d(pos)
        self.n = self.pos.shape[0]
        self.eps = eps_list
        self.index = np.arange(self.n)
        self.inout = np.full(self.n, medium)

    def eps1(self, enei):
        eps_val, _ = self.eps[0](enei)
        return np.full(self.n, eps_val)


def _make_particle(diameter = 20.0):
    """Create a simple metallic nanosphere ComParticle."""
    eps_out = EpsConst(1.0)
    eps_in = EpsConst(-10.0 + 0.5j)
    p = trisphere(32, diameter)
    cp = ComParticle([eps_out, eps_in], [p], [[2, 1]])
    return cp, eps_out, eps_in


def _make_sig(cp, enei, seed = 42):
    """Create a synthetic CompStruct solution (surface charges/currents)."""
    nfaces = cp.nfaces
    rng = np.random.RandomState(seed)
    sig1 = rng.randn(nfaces) + 1j * rng.randn(nfaces)
    sig2 = rng.randn(nfaces) + 1j * rng.randn(nfaces)
    h1 = rng.randn(nfaces, 3) + 1j * rng.randn(nfaces, 3)
    h2 = rng.randn(nfaces, 3) + 1j * rng.randn(nfaces, 3)
    sig = CompStruct(cp, enei, sig1 = sig1, sig2 = sig2, h1 = h1, h2 = h2)
    return sig


# ============================================================================
# DipoleRet scattering tests
# ============================================================================

class TestDipoleRetScattering(object):

    def test_scattering_returns_positive_single_dipole(self):
        """DipoleRet.scattering with a single z-dipole returns positive value."""
        cp, eps_out, eps_in = _make_particle()
        pt = MockComPoint(
            pos = np.array([[0.0, 0.0, 0.0]]),
            eps_list = [eps_out, eps_in],
            medium = 1,
        )
        dip = np.array([0.0, 0.0, 1.0])
        exc = DipoleRet(pt, dip)

        sig = _make_sig(cp, 500.0)
        sca, dsca = exc.scattering(sig)
        sca_val = np.asarray(sca)
        assert np.all(sca_val > 0), "Scattering cross section must be positive"

    def test_scattering_returns_positive_three_dipoles(self):
        """DipoleRet.scattering with default 3 orthogonal dipoles returns positive."""
        cp, eps_out, eps_in = _make_particle()
        pt = MockComPoint(
            pos = np.array([[0.0, 0.0, 0.0]]),
            eps_list = [eps_out, eps_in],
            medium = 1,
        )
        exc = DipoleRet(pt)  # Default: eye(3) dipoles

        sig = _make_sig(cp, 500.0)
        sca, dsca = exc.scattering(sig)
        sca_val = np.asarray(sca)
        # Shape should be (npt=1, ndip=3)
        assert sca_val.shape == (1, 3), (
            "Expected shape (1, 3), got {}".format(sca_val.shape)
        )
        assert np.all(sca_val > 0), "All scattering values must be positive"

    def test_scattering_shape_matches_dipole_dimensions(self):
        """Scattering output shape should be (npt, ndip)."""
        cp, eps_out, eps_in = _make_particle()
        pt = MockComPoint(
            pos = np.array([[0.0, 0.0, 0.0]]),
            eps_list = [eps_out, eps_in],
            medium = 1,
        )
        # Single dipole: shape should be scalar-like
        dip = np.array([1.0, 0.0, 0.0])
        exc = DipoleRet(pt, dip)
        sig = _make_sig(cp, 500.0)
        sca, dsca = exc.scattering(sig)
        sca_val = np.asarray(sca)
        assert sca_val.shape == (1, 1), (
            "Expected shape (1, 1), got {}".format(sca_val.shape)
        )

    def test_scattering_dsca_is_compstruct(self):
        """Differential scattering should be returned as CompStruct."""
        cp, eps_out, eps_in = _make_particle()
        pt = MockComPoint(
            pos = np.array([[0.0, 0.0, 0.0]]),
            eps_list = [eps_out, eps_in],
            medium = 1,
        )
        exc = DipoleRet(pt, np.array([0.0, 0.0, 1.0]))
        sig = _make_sig(cp, 500.0)
        sca, dsca = exc.scattering(sig)
        assert isinstance(dsca, CompStruct), (
            "dsca should be a CompStruct, got {}".format(type(dsca))
        )
        assert hasattr(dsca, 'dsca'), "dsca CompStruct should have 'dsca' field"

    def test_scattering_different_wavelengths(self):
        """DipoleRet.scattering should give different values at different wavelengths."""
        cp, eps_out, eps_in = _make_particle()
        pt = MockComPoint(
            pos = np.array([[0.0, 0.0, 0.0]]),
            eps_list = [eps_out, eps_in],
            medium = 1,
        )
        dip = np.array([0.0, 0.0, 1.0])
        exc = DipoleRet(pt, dip)

        sig_400 = _make_sig(cp, 400.0, seed = 42)
        sig_600 = _make_sig(cp, 600.0, seed = 42)
        sca_400, _ = exc.scattering(sig_400)
        sca_600, _ = exc.scattering(sig_600)

        # Different wavelengths should give different scattering values
        # (because the wavenumber k differs even with the same charges)
        assert not np.allclose(sca_400, sca_600), (
            "Scattering at 400nm and 600nm should differ"
        )

    def test_spec_initialized(self):
        """DipoleRet.spec should be a SpectrumRet, not None."""
        cp, eps_out, eps_in = _make_particle()
        pt = MockComPoint(
            pos = np.array([[0.0, 0.0, 0.0]]),
            eps_list = [eps_out, eps_in],
            medium = 1,
        )
        exc = DipoleRet(pt)
        assert exc.spec is not None, "DipoleRet.spec should not be None"
        assert isinstance(exc.spec, SpectrumRet), (
            "DipoleRet.spec should be SpectrumRet, got {}".format(type(exc.spec))
        )


# ============================================================================
# PlaneWaveRet scattering tests
# ============================================================================

class TestPlaneWaveRetScattering(object):

    def test_scattering_returns_positive(self):
        """PlaneWaveRet.scattering should return positive cross section."""
        cp, eps_out, eps_in = _make_particle()
        pol = np.array([1.0, 0.0, 0.0])
        dir_vec = np.array([0.0, 0.0, 1.0])
        exc = PlaneWaveRet(pol, dir_vec, medium = 1)

        sig = _make_sig(cp, 500.0)
        sca, dsca = exc.scattering(sig)
        assert sca > 0, "Scattering cross section must be positive, got {}".format(sca)

    def test_scattering_dsca_is_compstruct(self):
        """Differential scattering should be a CompStruct with 'dsca' field."""
        cp, eps_out, eps_in = _make_particle()
        pol = np.array([1.0, 0.0, 0.0])
        dir_vec = np.array([0.0, 0.0, 1.0])
        exc = PlaneWaveRet(pol, dir_vec, medium = 1)

        sig = _make_sig(cp, 500.0)
        sca, dsca = exc.scattering(sig)
        assert isinstance(dsca, CompStruct), (
            "dsca should be CompStruct, got {}".format(type(dsca))
        )

    def test_scattering_different_wavelengths(self):
        """Scattering should differ at different wavelengths."""
        cp, eps_out, eps_in = _make_particle()
        pol = np.array([1.0, 0.0, 0.0])
        dir_vec = np.array([0.0, 0.0, 1.0])
        exc = PlaneWaveRet(pol, dir_vec, medium = 1)

        sig_400 = _make_sig(cp, 400.0, seed = 42)
        sig_600 = _make_sig(cp, 600.0, seed = 42)
        sca_400, _ = exc.scattering(sig_400)
        sca_600, _ = exc.scattering(sig_600)

        assert not np.isclose(sca_400, sca_600), (
            "Scattering at 400nm and 600nm should differ"
        )

    def test_spec_initialized(self):
        """PlaneWaveRet.spec should be a SpectrumRet, not None."""
        pol = np.array([1.0, 0.0, 0.0])
        dir_vec = np.array([0.0, 0.0, 1.0])
        exc = PlaneWaveRet(pol, dir_vec)
        assert exc.spec is not None, "PlaneWaveRet.spec should not be None"
        assert isinstance(exc.spec, SpectrumRet), (
            "PlaneWaveRet.spec should be SpectrumRet, got {}".format(
                type(exc.spec))
        )


# ============================================================================
# PlaneWaveRet extinction tests
# ============================================================================

class TestPlaneWaveRetExtinction(object):

    def test_extinction_returns_scalar(self):
        """Extinction should return a scalar for single polarization."""
        cp, eps_out, eps_in = _make_particle()
        pol = np.array([1.0, 0.0, 0.0])
        dir_vec = np.array([0.0, 0.0, 1.0])
        exc = PlaneWaveRet(pol, dir_vec, medium = 1)

        sig = _make_sig(cp, 500.0)
        ext = exc.extinction(sig)
        assert np.isscalar(ext) or (isinstance(ext, np.ndarray) and ext.ndim == 0), (
            "Extinction should be scalar, got type={} shape={}".format(
                type(ext), getattr(ext, 'shape', 'N/A'))
        )

    def test_extinction_is_finite(self):
        """Extinction should be a finite number."""
        cp, eps_out, eps_in = _make_particle()
        pol = np.array([1.0, 0.0, 0.0])
        dir_vec = np.array([0.0, 0.0, 1.0])
        exc = PlaneWaveRet(pol, dir_vec, medium = 1)

        sig = _make_sig(cp, 500.0)
        ext = exc.extinction(sig)
        assert np.isfinite(ext), "Extinction should be finite, got {}".format(ext)

    def test_extinction_different_wavelengths(self):
        """Extinction should differ at different wavelengths."""
        cp, eps_out, eps_in = _make_particle()
        pol = np.array([1.0, 0.0, 0.0])
        dir_vec = np.array([0.0, 0.0, 1.0])
        exc = PlaneWaveRet(pol, dir_vec, medium = 1)

        sig_400 = _make_sig(cp, 400.0, seed = 42)
        sig_600 = _make_sig(cp, 600.0, seed = 42)
        ext_400 = exc.extinction(sig_400)
        ext_600 = exc.extinction(sig_600)

        assert not np.isclose(ext_400, ext_600), (
            "Extinction at 400nm and 600nm should differ"
        )


# ============================================================================
# PlaneWaveRet absorption = extinction - scattering
# ============================================================================

class TestPlaneWaveRetAbsorption(object):

    def test_absorption_equals_ext_minus_sca(self):
        """Absorption should equal extinction minus scattering."""
        cp, eps_out, eps_in = _make_particle()
        pol = np.array([1.0, 0.0, 0.0])
        dir_vec = np.array([0.0, 0.0, 1.0])
        exc = PlaneWaveRet(pol, dir_vec, medium = 1)

        sig = _make_sig(cp, 500.0)
        ext = exc.extinction(sig)
        sca, _ = exc.scattering(sig)
        abs_val = exc.absorption(sig)

        assert abs_val == pytest.approx(ext - sca, rel = 1e-10), (
            "abs ({}) should equal ext ({}) - sca ({})".format(abs_val, ext, sca)
        )

    def test_absorption_identity_different_wavelengths(self):
        """ext = sca + abs should hold at multiple wavelengths."""
        cp, eps_out, eps_in = _make_particle()
        pol = np.array([1.0, 0.0, 0.0])
        dir_vec = np.array([0.0, 0.0, 1.0])
        exc = PlaneWaveRet(pol, dir_vec, medium = 1)

        for enei in [400.0, 500.0, 600.0, 700.0]:
            sig = _make_sig(cp, enei, seed = 42)
            ext = exc.extinction(sig)
            sca, _ = exc.scattering(sig)
            abs_val = exc.absorption(sig)

            assert abs_val == pytest.approx(ext - sca, rel = 1e-10), (
                "At {}nm: abs ({}) != ext ({}) - sca ({})".format(
                    enei, abs_val, ext, sca)
            )

    def test_absorption_identity_different_polarizations(self):
        """ext = sca + abs should hold for different polarizations."""
        cp, eps_out, eps_in = _make_particle()
        sig = _make_sig(cp, 500.0)

        # x-polarized
        exc_x = PlaneWaveRet(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            medium = 1,
        )
        ext_x = exc_x.extinction(sig)
        sca_x, _ = exc_x.scattering(sig)
        abs_x = exc_x.absorption(sig)
        assert abs_x == pytest.approx(ext_x - sca_x, rel = 1e-10)

        # y-polarized
        exc_y = PlaneWaveRet(
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            medium = 1,
        )
        ext_y = exc_y.extinction(sig)
        sca_y, _ = exc_y.scattering(sig)
        abs_y = exc_y.absorption(sig)
        assert abs_y == pytest.approx(ext_y - sca_y, rel = 1e-10)


# ============================================================================
# CompStruct addition (needed by DipoleRet.scattering)
# ============================================================================

class TestCompStructAddition(object):

    def test_add_matching_fields(self):
        """Adding two CompStructs with matching fields sums them element-wise."""

        class FakeP(object):
            pass

        p = FakeP()
        a = CompStruct(p, 500.0, e = np.array([1.0, 2.0]), h = np.array([3.0, 4.0]))
        b = CompStruct(p, 500.0, e = np.array([5.0, 6.0]), h = np.array([7.0, 8.0]))
        c = a + b
        np.testing.assert_array_equal(c.e, [6.0, 8.0])
        np.testing.assert_array_equal(c.h, [10.0, 12.0])

    def test_add_preserves_particle_and_enei(self):
        """Addition should preserve p and enei from left operand."""

        class FakeP(object):
            pass

        p = FakeP()
        a = CompStruct(p, 500.0, e = np.array([1.0]))
        b = CompStruct(p, 500.0, e = np.array([2.0]))
        c = a + b
        assert c.p is p
        assert c.enei == 500.0

    def test_radd_with_zero(self):
        """0 + CompStruct should return the CompStruct (for sum() support)."""

        class FakeP(object):
            pass

        p = FakeP()
        a = CompStruct(p, 500.0, e = np.array([1.0, 2.0]))
        result = 0 + a
        np.testing.assert_array_equal(result.e, [1.0, 2.0])
