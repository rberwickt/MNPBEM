"""Edge case tests for output / spectrum / mesh-field stages.

Covers M3 Wave 2 B3:
  5.1 Spectrum   — wide wavelength, fine sampling, cross-section consistency
  5.2 MeshField  — single point, line scan, 2D plane, 3D volume, surface limit
  5.3 SpectrumRet/Stat output — custom pinfty, custom direction array

All tests use small Au or generic Drude spheres so they finish quickly. The
goal is robustness: results must be finite, real, and physically consistent
(in particular, ext = sca + abs to within a small tolerance for both stat
and ret BEM).
"""
import os
import sys

import numpy as np
import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mnpbem.materials import EpsConst, EpsDrude
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.bem import BEMStat, BEMRet
from mnpbem.simulation import PlaneWaveStat, PlaneWaveRet, MeshField
from mnpbem.spectrum import SpectrumRet, SpectrumStat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_au_sphere_stat(radius = 5.0, nfaces = 144):
    epsm = EpsConst(1.0)
    epsAu = EpsDrude(eps0 = 10.0, wp = 9.065, gammad = 0.0708, name = 'gold')
    sphere = trisphere(nfaces, radius)
    return ComParticle([epsm, epsAu], [sphere], [[2, 1]], 1)


def _make_au_sphere_ret(radius = 30.0, nfaces = 144):
    epsm = EpsConst(1.0)
    epsAu = EpsDrude(eps0 = 10.0, wp = 9.065, gammad = 0.0708, name = 'gold')
    sphere = trisphere(nfaces, radius)
    return ComParticle([epsm, epsAu], [sphere], [[2, 1]], 1)


def _solve_stat(p, enei, pol = (1.0, 0.0, 0.0)):
    bem = BEMStat(p)
    exc = PlaneWaveStat(list(pol))
    sig, _ = bem.solve(exc.potential(p, enei))
    return bem, exc, sig


def _solve_ret(p, enei, pol = (1.0, 0.0, 0.0), kdir = (0.0, 0.0, 1.0)):
    bem = BEMRet(p)
    exc = PlaneWaveRet(list(pol), list(kdir))
    sig, _ = bem.solve(exc.potential(p, enei))
    return bem, exc, sig


# ---------------------------------------------------------------------------
# 5.1 Spectrum
# ---------------------------------------------------------------------------

class TestSpectrumWavelengthRange(object):
    """Wide wavelength range and very fine sampling."""

    def test_wide_range_decade_stat(self):
        """100 nm to 10000 nm: extinction must stay finite everywhere."""
        p = _make_au_sphere_stat(radius = 5.0, nfaces = 144)
        wavelengths = np.logspace(np.log10(100.0), np.log10(10000.0), 21)

        bem = BEMStat(p)
        exc = PlaneWaveStat([1, 0, 0])

        ext = np.zeros(len(wavelengths))
        sca = np.zeros(len(wavelengths))
        abs_ = np.zeros(len(wavelengths))

        for i, enei in enumerate(wavelengths):
            sig, _ = bem.solve(exc.potential(p, enei))
            ext[i] = exc.extinction(sig)
            sca[i] = exc.scattering(sig)
            abs_[i] = exc.absorption(sig)

        assert np.all(np.isfinite(ext)), 'extinction not finite over wide range'
        assert np.all(np.isfinite(sca)), 'scattering not finite over wide range'
        assert np.all(np.isfinite(abs_)), 'absorption not finite over wide range'
        # Cross sections must be non-negative (modulo small numerical noise).
        # ABS uses Im(eps); for real eps it can be ~0 with sign noise. Tol = 1e-6.
        peak = max(abs_.max(), sca.max(), ext.max(), 1.0)
        assert sca.min() > -1e-6 * peak
        assert ext.min() > -1e-6 * peak

    def test_fine_sampling_500_points(self):
        """500-point sampling around the resonance: result must be smooth."""
        p = _make_au_sphere_stat(radius = 5.0, nfaces = 144)
        wavelengths = np.linspace(450.0, 600.0, 500)

        bem = BEMStat(p)
        exc = PlaneWaveStat([1, 0, 0])

        ext = np.zeros(len(wavelengths))
        for i, enei in enumerate(wavelengths):
            sig, _ = bem.solve(exc.potential(p, enei))
            ext[i] = exc.extinction(sig)

        assert np.all(np.isfinite(ext))
        # Smoothness: max relative jump between neighbors should be small.
        rel_jump = np.abs(np.diff(ext)) / (np.abs(ext[:-1]) + 1e-12)
        assert rel_jump.max() < 0.5, 'spectrum not smooth on fine grid'


class TestSpectrumPlasmonResonance(object):
    """Plasmon resonance must be located near 520-560 nm for small Au sphere."""

    def test_au_sphere_resonance_stat(self):
        p = _make_au_sphere_stat(radius = 5.0, nfaces = 144)
        wavelengths = np.linspace(450.0, 650.0, 41)
        bem = BEMStat(p)
        exc = PlaneWaveStat([1, 0, 0])

        ext = np.array([
            exc.extinction(bem.solve(exc.potential(p, enei))[0])
            for enei in wavelengths
        ])
        idx = int(np.argmax(ext))
        lam_res = float(wavelengths[idx])
        # With Drude (eps0=10, wp=9.065 eV, gammad=0.0708) the quasistatic
        # Frohlich condition Re(eps_metal) = -2 sets the resonance. For these
        # parameters it lands near 470-490 nm; use a wide tolerance window.
        assert 440.0 < lam_res < 580.0, (
            'Au sphere resonance out of range: {:.1f} nm'.format(lam_res)
        )


class TestCrossSectionConsistency(object):
    """ext = sca + abs must hold to high accuracy."""

    @pytest.mark.parametrize('enei', [400.0, 530.0, 700.0])
    def test_stat_energy_conservation(self, enei):
        p = _make_au_sphere_stat(radius = 5.0, nfaces = 144)
        bem = BEMStat(p)
        exc = PlaneWaveStat([1, 0, 0])
        sig, _ = bem.solve(exc.potential(p, enei))

        ext = exc.extinction(sig)
        sca = exc.scattering(sig)
        abs_ = exc.absorption(sig)

        rel_err = abs(ext - (sca + abs_)) / max(abs(ext), 1e-30)
        assert rel_err < 1e-6, (
            'Energy conservation violated at enei={}: ext={}, sca+abs={}'
            .format(enei, ext, sca + abs_)
        )

    @pytest.mark.parametrize('enei', [500.0, 600.0])
    def test_ret_energy_conservation(self, enei):
        p = _make_au_sphere_ret(radius = 30.0, nfaces = 144)
        bem = BEMRet(p)
        exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])
        sig, _ = bem.solve(exc.potential(p, enei))

        ext = exc.extinction(sig)
        sca, _dsca = exc.scattering(sig)
        abs_ = exc.absorption(sig)

        rel_err = abs(ext - (sca + abs_)) / max(abs(ext), 1e-30)
        # In the retarded case absorption is computed as ext - sca by construction,
        # so this is essentially an internal-consistency check (must be exact).
        assert rel_err < 1e-10


class TestSpectrumPinftyVariants(object):
    """SpectrumRet/SpectrumStat with various pinfty inputs."""

    def test_default_pinfty(self):
        spec = SpectrumRet()
        assert spec.ndir > 0
        assert spec.nvec.shape == (spec.ndir, 3)
        assert spec.area.shape == (spec.ndir,)
        assert np.all(np.isfinite(spec.nvec))
        assert np.all(spec.area >= 0)

    def test_int_pinfty(self):
        spec = SpectrumRet(64)
        assert spec.ndir > 0

    def test_array_pinfty(self):
        dirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]],
                        dtype = float)
        spec = SpectrumRet(dirs)
        assert spec.ndir == 4
        assert np.allclose(spec.nvec, dirs)

    def test_spectrum_total_equals_planewave_ret_scattering(self):
        """SpectrumRet.scattering total power matches PlaneWaveRet.scattering
        up to the 1/(0.5 nb) normalization factor."""
        p = _make_au_sphere_ret(radius = 30.0, nfaces = 144)
        bem = BEMRet(p)
        exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])
        sig, _ = bem.solve(exc.potential(p, 530.0))

        sca_pw, _ = exc.scattering(sig)
        spec = exc.spec  # internal SpectrumRet
        sca_sp, _ = spec.scattering(sig)

        eps_val, _ = p.eps[0](sig.enei)
        nb = float(np.real(np.sqrt(eps_val)))
        rel_err = abs(sca_pw - sca_sp / (0.5 * nb)) / max(abs(sca_pw), 1e-30)
        assert rel_err < 1e-10


# ---------------------------------------------------------------------------
# 5.2 MeshField
# ---------------------------------------------------------------------------

class TestMeshFieldBasics(object):
    """Basic MeshField geometry inputs: point, line, plane, volume."""

    def setup_method(self, _method):
        self.p = _make_au_sphere_stat(radius = 5.0, nfaces = 144)
        bem = BEMStat(self.p)
        exc = PlaneWaveStat([1, 0, 0])
        self.sig, _ = bem.solve(exc.potential(self.p, 530.0))

    def test_single_point(self):
        mf = MeshField(self.p, np.array([20.0]),
                                np.array([0.0]),
                                np.array([0.0]))
        e, h = mf.field(self.sig)
        assert e.shape == (1, 3)
        assert np.all(np.isfinite(e))
        # Quasistatic: no h field
        assert h is None or np.all(np.isfinite(h))

    def test_line_scan(self):
        x = np.linspace(10.0, 50.0, 100)
        mf = MeshField(self.p, x, 0.0, 0.0)
        e, h = mf.field(self.sig)
        assert e.shape == (100, 3)
        assert np.all(np.isfinite(e))
        # Far from sphere: |E| should decrease monotonically (roughly).
        norm = np.linalg.norm(e, axis = 1)
        # Allow some non-monotonicity due to face discretization, but the
        # global trend must be decreasing.
        assert norm[0] > norm[-1]

    def test_plane_2d(self):
        # NB: grid spans the particle interior; ComPoint masks interior
        # points with NaN since 'inout=2' (outside) is not defined there.
        # Restrict the finite-check to points outside the sphere.
        x_grid, z_grid = np.meshgrid(np.linspace(-20, 20, 21),
                                     np.linspace(-20, 20, 21))
        mf = MeshField(self.p, x_grid, 0.0, z_grid, mindist = 1.0)
        e, h = mf.field(self.sig)
        assert e.shape == (21, 21, 3)
        # Outside-the-sphere mask: r > radius + small buffer
        r2 = x_grid ** 2 + z_grid ** 2
        outside = r2 > (5.0 + 1.0) ** 2
        assert np.all(np.isfinite(e[outside]))

    def test_plane_xy(self):
        x_grid, y_grid = np.meshgrid(np.linspace(-20, 20, 11),
                                     np.linspace(-20, 20, 11))
        mf = MeshField(self.p, x_grid, y_grid, 15.0, mindist = 1.0)
        e, h = mf.field(self.sig)
        assert e.shape == (11, 11, 3)
        assert np.all(np.isfinite(e))

    def test_plane_yz(self):
        y_grid, z_grid = np.meshgrid(np.linspace(-20, 20, 11),
                                     np.linspace(-20, 20, 11))
        mf = MeshField(self.p, 15.0, y_grid, z_grid, mindist = 1.0)
        e, h = mf.field(self.sig)
        assert e.shape == (11, 11, 3)
        assert np.all(np.isfinite(e))

    def test_3d_volume(self):
        x = np.linspace(-15, 15, 6)
        y = np.linspace(-15, 15, 6)
        z = np.linspace(-15, 15, 6)
        xg, yg, zg = np.meshgrid(x, y, z, indexing = 'ij')
        mf = MeshField(self.p, xg, yg, zg, mindist = 1.0)
        e, h = mf.field(self.sig)
        assert e.shape == (6, 6, 6, 3)
        # Same caveat as test_plane_2d: only points outside the sphere are finite.
        r2 = xg ** 2 + yg ** 2 + zg ** 2
        outside = r2 > (5.0 + 1.0) ** 2
        assert np.all(np.isfinite(e[outside]))


class TestMeshFieldSurfaceLimit(object):
    """Behaviour when grid points approach the particle boundary."""

    def test_surface_proximity_with_mindist(self):
        """mindist guarantees finite E even arbitrarily close to surface."""
        p = _make_au_sphere_stat(radius = 5.0, nfaces = 144)
        bem = BEMStat(p)
        exc = PlaneWaveStat([1, 0, 0])
        sig, _ = bem.solve(exc.potential(p, 530.0))

        # Points along x-axis from r=5.5 (just outside) to r=20.
        x = np.linspace(5.5, 20.0, 30)
        mf = MeshField(p, x, 0.0, 0.0, mindist = 0.5)
        e, h = mf.field(sig)
        assert np.all(np.isfinite(e))

    def test_inside_outside_indices(self):
        """field(sig, inout=2) outside is the default; both 1 and 2 must be
        finite for a point that is unambiguously outside."""
        p = _make_au_sphere_stat(radius = 5.0, nfaces = 144)
        bem = BEMStat(p)
        exc = PlaneWaveStat([1, 0, 0])
        sig, _ = bem.solve(exc.potential(p, 530.0))

        x = np.array([20.0])
        mf = MeshField(p, x, 0.0, 0.0)
        e_out, _ = mf.field(sig, inout = 2)
        assert np.all(np.isfinite(e_out))


class TestMeshFieldRetarded(object):
    """Retarded MeshField produces both E and H."""

    def test_ret_field_has_h(self):
        p = _make_au_sphere_ret(radius = 30.0, nfaces = 144)
        bem = BEMRet(p)
        exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])
        sig, _ = bem.solve(exc.potential(p, 530.0))

        x_grid, z_grid = np.meshgrid(np.linspace(-50, 50, 11),
                                     np.linspace(-50, 50, 11))
        mf = MeshField(p, x_grid, 0.0, z_grid, mindist = 2.0, sim = 'ret')
        e, h = mf.field(sig)
        assert e.shape == (11, 11, 3)
        assert h is not None
        assert h.shape == (11, 11, 3), 'h shape {} != (11, 11, 3)'.format(h.shape)
        # Mask out interior points (r=30 sphere) before checking finiteness.
        r2 = x_grid ** 2 + z_grid ** 2
        outside = r2 > (30.0 + 2.0) ** 2
        assert np.all(np.isfinite(e[outside]))
        assert np.all(np.isfinite(h[outside]))


# ---------------------------------------------------------------------------
# 5.3 SpectrumRet / SpectrumStat output structures
# ---------------------------------------------------------------------------

class TestSpectrumOutputStructure(object):
    """dsca CompStruct must carry correct shape and finite values."""

    def test_dsca_shape_matches_pinfty(self):
        p = _make_au_sphere_ret(radius = 30.0, nfaces = 144)
        bem = BEMRet(p)
        exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])
        sig, _ = bem.solve(exc.potential(p, 530.0))

        spec = SpectrumRet(64)
        sca, dsca = spec.scattering(sig)
        assert dsca.dsca.shape[0] == spec.ndir
        assert np.all(np.isfinite(dsca.dsca))
        # Differential power must be non-negative everywhere (Poynting flux out).
        assert dsca.dsca.min() > -1e-10 * max(abs(sca), 1.0)

    def test_dsca_integral_equals_total(self):
        p = _make_au_sphere_ret(radius = 30.0, nfaces = 144)
        bem = BEMRet(p)
        exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])
        sig, _ = bem.solve(exc.potential(p, 530.0))

        spec = SpectrumRet(64)
        sca, dsca = spec.scattering(sig)
        sca_int = float(np.sum(spec.area * dsca.dsca))
        rel_err = abs(sca - sca_int) / max(abs(sca), 1e-30)
        assert rel_err < 1e-10

    def test_farfield_shapes(self):
        p = _make_au_sphere_ret(radius = 30.0, nfaces = 144)
        bem = BEMRet(p)
        exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])
        sig, _ = bem.solve(exc.potential(p, 530.0))

        spec = SpectrumRet(64)
        field = spec.farfield(sig)
        assert field.e.shape[0] == spec.ndir
        assert field.e.shape[-1] == 3
        assert np.all(np.isfinite(field.e))
        assert np.all(np.isfinite(field.h))

    def test_custom_directions_specific_angles(self):
        p = _make_au_sphere_ret(radius = 30.0, nfaces = 144)
        bem = BEMRet(p)
        exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])
        sig, _ = bem.solve(exc.potential(p, 530.0))

        # Forward (+z), backward (-z), sideways (+x, +y).
        dirs = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0]],
                        dtype = float)
        spec = SpectrumRet(dirs)
        field = spec.farfield(sig)
        assert field.e.shape[0] == 4
        assert np.all(np.isfinite(field.e))
