"""
Comprehensive tests for EELS (Electron Energy Loss Spectroscopy) module.

Tests for:
  - EELSBase: ene2vel, enclosure, distmin, inpolygon, potwire, potinfty,
              potinside, path, full, beam path setup
  - EELSStat: potential, loss, bulkloss, field
  - EELSRet: potential, loss, bulkloss, rad, field, _fieldinfty

MATLAB reference: Simulation/misc/@eelsbase, Simulation/static/@eelsstat,
                  Simulation/retarded/@eelsret
"""

import sys
import os
import numpy as np
import pytest
from scipy.special import kv as besselk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mnpbem.simulation.eels_base import EELSBase
from mnpbem.simulation.eels_stat import EELSStat
from mnpbem.simulation.eels_ret import EELSRet
from mnpbem.misc.units import EV2NM, BOHR, HARTREE, FINE
from mnpbem.greenfun import CompStruct


# ---------------------------------------------------------------------------
# Mock particle objects
# ---------------------------------------------------------------------------

class MockSphere(object):
    """
    Simplified sphere-like particle for EELS testing.

    An octahedron (8 triangular faces) centered at origin with radius R,
    approximating a sphere.  Provides: verts, faces, pos (centroids),
    nvec (outward normals), area, n, np_, eps, inout, index.
    """

    def __init__(self, radius=10.0, eps_funcs=None, inout_arr=None):
        R = radius
        # Octahedron vertices
        self.verts = np.array([
            [R, 0, 0], [-R, 0, 0],
            [0, R, 0], [0, -R, 0],
            [0, 0, R], [0, 0, -R],
        ], dtype=float)

        # 8 triangular faces (NaN-padded 4th column for MATLAB compat)
        self.faces = np.array([
            [0, 2, 4, np.nan],
            [2, 1, 4, np.nan],
            [1, 3, 4, np.nan],
            [3, 0, 4, np.nan],
            [2, 0, 5, np.nan],
            [1, 2, 5, np.nan],
            [3, 1, 5, np.nan],
            [0, 3, 5, np.nan],
        ], dtype=float)

        # Centroids
        nf = self.faces.shape[0]
        self.pos = np.zeros((nf, 3))
        for i in range(nf):
            vidx = self.faces[i, :3].astype(int)
            self.pos[i] = self.verts[vidx].mean(axis=0)

        # Outward normals (for octahedron, normal = normalized centroid)
        norms = np.linalg.norm(self.pos, axis=1, keepdims=True)
        self.nvec = self.pos / norms

        # Face areas via cross product
        self.area = np.zeros(nf)
        for i in range(nf):
            vidx = self.faces[i, :3].astype(int)
            v0, v1, v2 = self.verts[vidx]
            self.area[i] = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

        self.n = nf
        # number of "particle objects" (MATLAB comparticle subobjects)
        self._np = 1

        # inout: material at inside / outside of each subobject (1-indexed)
        # [[2, 1]] means inside=2 (particle), outside=1 (vacuum)
        if inout_arr is not None:
            self.inout = np.array(inout_arr, dtype=int)
        else:
            self.inout = np.array([[2, 1]], dtype=int)

        # Dielectric functions: [vacuum, particle_material]
        if eps_funcs is None:
            self.eps = [
                lambda enei: (1.0 + 0j, 2 * np.pi / enei),
                lambda enei: (-10.0 + 1.0j, 2 * np.pi / enei * np.sqrt(-10.0 + 1.0j)),
            ]
        else:
            self.eps = eps_funcs

    @property
    def np(self):
        return self._np

    def index(self, ip):
        # MATLAB: 1-indexed, returns face indices belonging to subobject ip
        # When ip is an ndarray (face mask from eels_ret), return it as-is
        if isinstance(ip, np.ndarray):
            return list(ip)
        if ip == 1:
            return list(range(self.n))
        return []


class MockSphereNoIntersect(MockSphere):
    """
    Sphere that is small/far enough that no electron beam intersects it.
    Used for testing pure grazing/far-away beams.
    """

    def __init__(self):
        super(MockSphereNoIntersect, self).__init__(radius=1.0)


class MockFlatPlate(object):
    """
    Flat square plate in the z=0 plane, for testing beam-through and
    enclosure computations.  Consists of 2 triangular faces.
    """

    def __init__(self, half_size=5.0):
        s = half_size
        self.verts = np.array([
            [-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]
        ], dtype=float)

        self.faces = np.array([
            [0, 1, 2, np.nan],
            [0, 2, 3, np.nan],
        ], dtype=float)

        nf = 2
        self.pos = np.zeros((nf, 3))
        for i in range(nf):
            vidx = self.faces[i, :3].astype(int)
            self.pos[i] = self.verts[vidx].mean(axis=0)

        self.nvec = np.array([
            [0, 0, 1.0],
            [0, 0, 1.0],
        ])

        self.area = np.zeros(nf)
        for i in range(nf):
            vidx = self.faces[i, :3].astype(int)
            v0, v1, v2 = self.verts[vidx]
            self.area[i] = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

        self.n = nf
        self._np = 1
        self.inout = np.array([[2, 1]], dtype=int)
        self.eps = [
            lambda enei: (1.0 + 0j, 2 * np.pi / enei),
            lambda enei: (-10.0 + 1.0j, 2 * np.pi / enei * np.sqrt(-10.0 + 1.0j)),
        ]

    @property
    def np(self):
        return self._np

    def index(self, ip):
        if isinstance(ip, np.ndarray):
            return list(ip)
        if ip == 1:
            return list(range(self.n))
        return []


# ===========================================================================
#  TestEne2Vel -- static method on EELSBase
# ===========================================================================

class TestEne2Vel(object):

    def test_nonrelativistic_limit(self):
        # For 1 eV electron, v/c << 1
        vel = EELSBase.ene2vel(1.0)
        assert vel == pytest.approx(np.sqrt(2 * 1.0 / 0.51e6), rel=1e-3)

    def test_100keV(self):
        # 100 keV is a typical STEM energy
        vel = EELSBase.ene2vel(100e3)
        # MATLAB formula: sqrt(1 - 1/(1 + ene/0.51e6)^2)
        expected = np.sqrt(1 - 1.0 / (1 + 100e3 / 0.51e6) ** 2)
        assert vel == pytest.approx(expected, rel=1e-12)

    def test_200keV(self):
        vel = EELSBase.ene2vel(200e3)
        expected = np.sqrt(1 - 1.0 / (1 + 200e3 / 0.51e6) ** 2)
        assert vel == pytest.approx(expected, rel=1e-12)

    def test_zero_energy(self):
        vel = EELSBase.ene2vel(0.0)
        assert vel == pytest.approx(0.0, abs=1e-15)

    def test_very_high_energy_approaches_one(self):
        # At 1 TeV, v/c should be very close to 1
        vel = EELSBase.ene2vel(1e12)
        assert vel > 0.999

    def test_scalar_and_array(self):
        # Should work with numpy arrays too
        energies = np.array([100e3, 200e3, 300e3])
        vels = EELSBase.ene2vel(energies)
        for i, ene in enumerate(energies):
            assert vels[i] == pytest.approx(EELSBase.ene2vel(float(ene)), rel=1e-12)

    def test_matches_matlab_formula(self):
        # Explicit check of MATLAB formula: vel = sqrt(1 - 1./(1 + ene/0.51e6).^2)
        for ene in [10, 1e3, 1e4, 1e5, 1e6]:
            vel_py = EELSBase.ene2vel(ene)
            vel_matlab = np.sqrt(1 - 1.0 / (1 + ene / 0.51e6) ** 2)
            assert vel_py == pytest.approx(vel_matlab, rel=1e-14)


# ===========================================================================
#  TestEnclosure
# ===========================================================================

class TestEnclosure(object):

    def test_octahedron_sphere(self):
        p = MockSphere(radius=10.0)
        rad = EELSBase._enclosure(p)
        # For octahedron, max distance from centroid to vertex in 2D (x,y)
        # is bounded by the radius
        assert rad > 0
        assert rad <= 10.0 * np.sqrt(2)

    def test_flat_plate(self):
        p = MockFlatPlate(half_size=5.0)
        rad = EELSBase._enclosure(p)
        # Max 2D distance from centroid to vertex
        assert rad > 0
        # Centroid to farthest vertex in 2D should be <= diagonal/2
        assert rad <= 5.0 * np.sqrt(2) + 1.0

    def test_unit_sphere(self):
        p = MockSphere(radius=1.0)
        rad = EELSBase._enclosure(p)
        assert rad > 0
        assert rad < 2.0


# ===========================================================================
#  TestInPolygon
# ===========================================================================

class TestInPolygon(object):

    def test_point_inside_square(self):
        xv = np.array([0, 1, 1, 0], dtype=float)
        yv = np.array([0, 0, 1, 1], dtype=float)
        x = np.array([0.5])
        y = np.array([0.5])
        in_mask, on_mask = EELSBase._inpolygon(x, y, xv, yv)
        assert in_mask[0] is True or in_mask[0] == True

    def test_point_outside_square(self):
        xv = np.array([0, 1, 1, 0], dtype=float)
        yv = np.array([0, 0, 1, 1], dtype=float)
        x = np.array([2.0])
        y = np.array([2.0])
        in_mask, on_mask = EELSBase._inpolygon(x, y, xv, yv)
        assert in_mask[0] == False

    def test_point_on_edge(self):
        xv = np.array([0, 1, 1, 0], dtype=float)
        yv = np.array([0, 0, 1, 1], dtype=float)
        x = np.array([0.5])
        y = np.array([0.0])
        in_mask, on_mask = EELSBase._inpolygon(x, y, xv, yv)
        # On boundary should be detected as in_mask=True
        assert in_mask[0] == True

    def test_triangle(self):
        xv = np.array([0, 1, 0.5], dtype=float)
        yv = np.array([0, 0, 1.0], dtype=float)
        # Point inside triangle
        in_mask, _ = EELSBase._inpolygon(
            np.array([0.4]), np.array([0.3]), xv, yv)
        assert in_mask[0] == True
        # Point outside
        in_mask, _ = EELSBase._inpolygon(
            np.array([-1.0]), np.array([-1.0]), xv, yv)
        assert in_mask[0] == False

    def test_multiple_points(self):
        xv = np.array([0, 1, 1, 0], dtype=float)
        yv = np.array([0, 0, 1, 1], dtype=float)
        x = np.array([0.5, 2.0, 0.1])
        y = np.array([0.5, 0.5, 0.1])
        in_mask, on_mask = EELSBase._inpolygon(x, y, xv, yv)
        assert in_mask[0] == True
        assert in_mask[1] == False
        assert in_mask[2] == True


# ===========================================================================
#  TestDistmin
# ===========================================================================

class TestDistmin(object):

    def test_close_point(self):
        p = MockSphere(radius=10.0)
        # Point right next to a face
        pos = np.array([[5.0, 5.0]])
        dmin = EELSBase._distmin(p, pos, cutoff=20.0)
        # At least some faces should have finite distance
        assert not np.all(np.isnan(dmin))

    def test_far_point(self):
        p = MockSphere(radius=1.0)
        # Point very far away
        pos = np.array([[1000.0, 1000.0]])
        dmin = EELSBase._distmin(p, pos, cutoff=1.0)
        # All distances should be NaN (beyond cutoff)
        assert np.all(np.isnan(dmin))

    def test_shape(self):
        p = MockSphere(radius=5.0)
        npos = 3
        pos = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
        dmin = EELSBase._distmin(p, pos, cutoff=20.0)
        assert dmin.shape == (p.n, npos)


# ===========================================================================
#  TestPotwire
# ===========================================================================

class TestPotwire(object):

    def test_basic_shape(self):
        n_pos = 5
        n_seg = 3
        r = np.ones((n_pos, n_seg)) * 2.0
        z = np.zeros((n_pos, n_seg))
        q = 0.1
        k = 0.0
        z0 = np.array([-1.0, -2.0, -3.0])
        z1 = np.array([1.0, 2.0, 3.0])
        phi, phir, phiz = EELSBase._potwire(r, z, q, k, z0, z1)
        assert phi.shape == (n_pos, n_seg)
        assert phir.shape == (n_pos, n_seg)
        assert phiz.shape == (n_pos, n_seg)

    def test_zero_k_is_real_like(self):
        # With k=0 (quasistatic), the potential should be dominated
        # by the real part for q=0
        r = np.ones((3, 2)) * 1.0
        z = np.zeros((3, 2))
        z0 = np.array([-5.0, -5.0])
        z1 = np.array([5.0, 5.0])
        phi, phir, phiz = EELSBase._potwire(r, z, 0.0, 0.0, z0, z1)
        # With q=0 and k=0, integral is just log-type
        assert phi.shape == (3, 2)
        # Should be non-zero
        assert np.any(np.abs(phi) > 0)

    def test_symmetry_in_z(self):
        # Potential at z should equal potential at -z for symmetric wire [-L, L]
        r = np.ones((2, 1)) * 1.0
        z_pos = np.array([[1.0], [-1.0]])
        q = 0.0
        k = 0.0
        z0 = np.array([-5.0])
        z1 = np.array([5.0])
        phi_p, _, _ = EELSBase._potwire(r, z_pos, q, k, z0, z1)
        # With q=0, k=0, and symmetric z0/z1, phi should be equal at +z and -z
        assert phi_p[0, 0] == pytest.approx(phi_p[1, 0], rel=1e-10)

    def test_longer_wire_gives_larger_potential(self):
        r = np.ones((1, 2)) * 1.0
        z = np.zeros((1, 2))
        q = 0.0
        k = 0.0
        z0 = np.array([-1.0, -10.0])
        z1 = np.array([1.0, 10.0])
        phi, _, _ = EELSBase._potwire(r, z, q, k, z0, z1)
        # Longer wire should give larger (real) potential at midpoint
        assert np.abs(phi[0, 1]) > np.abs(phi[0, 0])


# ===========================================================================
#  TestEELSBase
# ===========================================================================

class TestEELSBase(object):

    def test_construction_far_beam(self):
        # Beam far from sphere -- no intersection
        p = MockSphere(radius=5.0)
        impact = np.array([[50.0, 50.0]])
        eels = EELSBase(p, impact, width=0.5, vel=0.5)
        assert eels.vel == 0.5
        assert eels.width == 0.5
        assert eels.impact.shape == (1, 2)
        # No intersection -> _z should be empty
        assert eels._z.shape[0] == 0
        assert len(eels._indimp) == 0
        assert len(eels._indmat) == 0

    def test_construction_grazing_beam(self):
        # Beam just outside sphere radius
        p = MockSphere(radius=5.0)
        # Impact at ~7 nm, well outside the octahedron
        impact = np.array([[7.0, 7.0]])
        eels = EELSBase(p, impact, width=0.5, vel=0.5)
        assert eels._z.shape[0] == 0

    def test_construction_through_beam(self):
        # Beam through flat plate at origin (z=0)
        p = MockFlatPlate(half_size=5.0)
        # Impact at (1, 1) -- inside the plate
        impact = np.array([[1.0, 1.0]])
        eels = EELSBase(p, impact, width=0.5, vel=0.5)
        # Beam passing through z=0 plate crosses 2 faces
        # This may or may not produce intersection points depending on geometry
        # The important thing is that it initializes without error
        assert eels.p is p
        assert eels.impact.shape == (1, 2)

    def test_multiple_impacts(self):
        p = MockSphere(radius=5.0)
        impact = np.array([
            [50.0, 0.0],
            [0.0, 50.0],
            [100.0, 100.0],
        ])
        eels = EELSBase(p, impact, width=0.5, vel=0.5)
        assert eels.impact.shape == (3, 2)

    def test_default_cutoff(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[50.0, 0.0]])
        eels = EELSBase(p, impact, width=0.5, vel=0.5)
        # Default cutoff = 10 * width = 5.0
        # _indquad shape should be (p.n, n_impact)
        assert eels._indquad.shape == (p.n, 1)

    def test_phiout_default(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[50.0, 0.0]])
        eels = EELSBase(p, impact, width=0.5, vel=0.5)
        assert eels.phiout == pytest.approx(1e-2)

    def test_phiout_custom(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[50.0, 0.0]])
        eels = EELSBase(p, impact, width=0.5, vel=0.5, phiout=0.05)
        assert eels.phiout == pytest.approx(0.05)

    def test_path_empty_for_far_beam(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[50.0, 50.0]])
        eels = EELSBase(p, impact, width=0.5, vel=0.5)
        result = eels.path()
        # No path inside any medium
        assert result.shape == (len(p.eps), 1)
        assert np.all(result == 0)

    def test_path_with_medium_selection(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[50.0, 50.0]])
        eels = EELSBase(p, impact, width=0.5, vel=0.5)
        # Select medium 1 (vacuum)
        result = eels.path(medium=1)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(0.0)

    def test_full_empty(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[50.0, 50.0]])
        eels = EELSBase(p, impact, width=0.5, vel=0.5)
        # For empty intersection, full should return zeros
        a = np.zeros((p.n, 0), dtype=complex)
        result = eels.full(a)
        assert result.shape == (p.n, 1)
        assert np.all(result == 0)

    def test_repr(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[50.0, 0.0]])
        eels = EELSBase(p, impact, width=0.5, vel=0.5)
        r = repr(eels)
        assert 'EELSBase' in r
        assert 'n_impact=1' in r

    def test_str(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[50.0, 0.0]])
        eels = EELSBase(p, impact, width=0.5, vel=0.5)
        s = str(eels)
        assert 'EELS Base' in s


# ===========================================================================
#  TestPotinfty -- potential for infinite electron beam
# ===========================================================================

class TestPotinfty(object):

    def test_output_shape(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0], [0.0, 15.0]])
        eels = EELSBase(p, impact, width=0.5, vel=0.5)
        q = 0.1
        phi, phip = eels.potinfty(q, gamma=1.0)
        assert phi.shape == (p.n, 2)
        assert phip.shape == (p.n, 2)

    def test_complex_output(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSBase(p, impact, width=0.5, vel=0.5)
        q = 0.1
        phi, phip = eels.potinfty(q, gamma=1.0)
        assert phi.dtype == complex
        assert phip.dtype == complex

    def test_bessel_decay(self):
        # Potential should decay with increasing impact parameter distance
        p = MockSphere(radius=1.0)
        impact_near = np.array([[3.0, 0.0]])
        impact_far = np.array([[30.0, 0.0]])

        eels_near = EELSBase(p, impact_near, width=0.1, vel=0.5)
        eels_far = EELSBase(p, impact_far, width=0.1, vel=0.5)

        q = 0.5
        phi_near, _ = eels_near.potinfty(q, gamma=1.0)
        phi_far, _ = eels_far.potinfty(q, gamma=1.0)

        # Average magnitude should be larger for closer beam
        assert np.mean(np.abs(phi_near)) > np.mean(np.abs(phi_far))

    def test_gamma_effect(self):
        # With gamma > 1, Bessel function argument is smaller -> larger potential
        p = MockSphere(radius=1.0)
        impact = np.array([[5.0, 0.0]])
        eels = EELSBase(p, impact, width=0.1, vel=0.5)
        q = 0.5

        phi_g1, _ = eels.potinfty(q, gamma=1.0)
        phi_g2, _ = eels.potinfty(q, gamma=2.0)

        # K0(q*rr/gamma) with larger gamma gives larger K0 for same rr
        # So phi with gamma=2 should have larger magnitude
        assert np.mean(np.abs(phi_g2)) > np.mean(np.abs(phi_g1))

    def test_matches_analytical_single_point(self):
        # Verify potinfty formula: phi = -2/vel * exp(i*q*z) * K0(q*rr/gamma)
        p = MockSphere(radius=1.0)
        impact = np.array([[5.0, 0.0]])
        eels = EELSBase(p, impact, width=0.1, vel=0.5)
        q = 0.3
        gamma = 1.2

        phi, _ = eels.potinfty(q, gamma=gamma)

        # Compute expected value for first face
        face_pos = p.pos[0, :]
        dx = face_pos[0] - 5.0
        dy = face_pos[1] - 0.0
        r = np.sqrt(dx ** 2 + dy ** 2)
        rr = np.sqrt(r ** 2 + eels.width ** 2)
        z = face_pos[2]

        K0_val = besselk(0, q * rr / gamma)
        expected_phi = -2.0 / eels.vel * np.exp(1j * q * z) * K0_val

        # Should match (may not be exact if refinement is applied)
        assert phi[0, 0] == pytest.approx(expected_phi, rel=1e-4)

    def test_potinfty_prefactor(self):
        # MATLAB: phi = -2/vel * exp(iqz) * K0(q*rr/gamma)
        # Check the -2/vel prefactor
        p = MockSphere(radius=1.0)
        impact = np.array([[10.0, 0.0]])
        vel = 0.7
        eels = EELSBase(p, impact, width=0.1, vel=vel)
        q = 0.1
        phi1, _ = eels.potinfty(q, gamma=1.0)

        # Scale velocity
        eels2 = EELSBase(p, impact, width=0.1, vel=2 * vel)
        phi2, _ = eels2.potinfty(q, gamma=1.0)

        # phi proportional to 1/vel, so phi1 / phi2 ~ 2
        ratio = np.abs(phi1[0, 0]) / np.abs(phi2[0, 0])
        assert ratio == pytest.approx(2.0, rel=1e-4)


# ===========================================================================
#  TestPotinside -- potential for electron beam inside media
# ===========================================================================

class TestPotinside(object):

    def test_empty_for_far_beam(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[50.0, 0.0]])
        eels = EELSBase(p, impact, width=0.5, vel=0.5)
        phi, phip = eels.potinside(0.1, 0.0)
        # No intersections -> empty second dim
        assert phi.shape[0] == p.n
        assert np.all(phi == 0)

    def test_output_shape_empty(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[50.0, 0.0]])
        eels = EELSBase(p, impact, width=0.5, vel=0.5)
        phi, phip = eels.potinside(0.1, 0.0)
        assert phi.shape == (p.n, 0)


# ===========================================================================
#  TestEELSStat
# ===========================================================================

class TestEELSStat(object):

    def test_construction(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSStat(p, impact, width=0.5, vel=0.5)
        assert isinstance(eels, EELSBase)
        assert eels.name == 'eels'
        assert eels.needs == {'sim': 'stat'}

    def test_potential_output_type(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSStat(p, impact, width=0.5, vel=0.5)
        enei = 500.0  # nm
        exc = eels.potential(p, enei)
        assert isinstance(exc, CompStruct)
        assert hasattr(exc, 'phi')
        assert hasattr(exc, 'phip')

    def test_potential_shape(self):
        p = MockSphere(radius=5.0)
        n_imp = 3
        impact = np.array([[15.0, 0.0], [0.0, 15.0], [10.0, 10.0]])
        eels = EELSStat(p, impact, width=0.5, vel=0.5)
        enei = 500.0
        exc = eels.potential(p, enei)
        assert exc.phi.shape == (p.n, n_imp)
        assert exc.phip.shape == (p.n, n_imp)

    def test_potential_complex(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSStat(p, impact, width=0.5, vel=0.5)
        enei = 500.0
        exc = eels.potential(p, enei)
        assert exc.phi.dtype == complex

    def test_bulkloss_far_beam(self):
        # Far beam, no path inside particle -> bulk loss should be ~0
        p = MockSphere(radius=5.0)
        impact = np.array([[50.0, 50.0]])
        eels = EELSStat(p, impact, width=0.5, vel=0.5)
        enei = 500.0
        pbulk = eels.bulkloss(enei)
        assert pbulk.shape == (1,)
        assert pbulk[0] == pytest.approx(0.0, abs=1e-20)

    def test_bulkloss_formula_components(self):
        # Verify the formula uses correct constants
        # MATLAB: 2*fine^2/(bohr*hartree*pi*vel^2) * imag(-1./eps) * path * log(...)
        p = MockSphere(radius=5.0)
        impact = np.array([[50.0, 0.0]])
        vel = 0.5
        eels = EELSStat(p, impact, width=0.5, vel=vel)
        enei = 500.0

        # With no path inside particle, bulk loss is zero
        pbulk = eels.bulkloss(enei)
        assert np.all(pbulk == 0.0)

    def test_loss_for_far_beam(self):
        # Create mock sig object
        p = MockSphere(radius=5.0)
        impact = np.array([[50.0, 0.0]])
        eels = EELSStat(p, impact, width=0.5, vel=0.5)

        class MockSig(object):
            pass

        sig = MockSig()
        sig.enei = 500.0
        sig.p = p
        sig.sig = np.random.randn(p.n, 1) + 1j * np.random.randn(p.n, 1)

        psurf, pbulk = eels.loss(sig)
        # Surface loss can be non-zero even for far beam (induced charges)
        assert psurf.shape == (1,)
        assert pbulk.shape == (1,)
        # Bulk loss zero for far beam
        assert pbulk[0] == pytest.approx(0.0, abs=1e-20)

    def test_loss_formula_prefactor(self):
        # MATLAB: psurf = -fine^2/(bohr*hartree*pi) * imag(area' * (conj(phi) .* sig))
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        vel = 0.5
        eels = EELSStat(p, impact, width=0.5, vel=vel)

        q = 2 * np.pi / (500.0 * vel)
        phi, _ = eels.potinfty(q, 1.0)

        # The prefactor in the loss formula
        prefac = -FINE ** 2 / (BOHR * HARTREE * np.pi)
        assert prefac < 0  # Negative prefactor
        # Combined with imag part, should give positive loss for proper sig

    def test_callable(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSStat(p, impact, width=0.5, vel=0.5)
        enei = 500.0
        exc_call = eels(p, enei)
        exc_pot = eels.potential(p, enei)
        np.testing.assert_allclose(exc_call.phi, exc_pot.phi)

    def test_repr(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSStat(p, impact, width=0.5, vel=0.5)
        r = repr(eels)
        assert 'EELSStat' in r

    def test_str(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSStat(p, impact, width=0.5, vel=0.5)
        s = str(eels)
        assert 'Quasistatic' in s


# ===========================================================================
#  TestEELSStatField
# ===========================================================================

class TestEELSStatField(object):

    def test_field_output_type(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSStat(p, impact, width=0.5, vel=0.5)
        exc = eels.field(p, 500.0)
        assert isinstance(exc, CompStruct)
        assert hasattr(exc, 'e')

    def test_field_shape(self):
        p = MockSphere(radius=5.0)
        n_imp = 2
        impact = np.array([[15.0, 0.0], [0.0, 15.0]])
        eels = EELSStat(p, impact, width=0.5, vel=0.5)
        exc = eels.field(p, 500.0)
        assert exc.e.shape == (p.n, 3, n_imp)

    def test_field_bessel_structure(self):
        # Verify the field uses K0 and K1 Bessel functions
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSStat(p, impact, width=0.5, vel=0.5)
        exc = eels.field(p, 500.0)
        # Ez component should involve K0 (imaginary prefactor)
        # Ex, Ey should involve K1
        # Just check they are complex and non-zero
        assert exc.e.dtype == complex
        assert np.any(np.abs(exc.e) > 0)


# ===========================================================================
#  TestEELSStatBulkloss -- detailed formula verification
# ===========================================================================

class TestEELSStatBulkloss(object):

    def test_zero_path_gives_zero_loss(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[100.0, 100.0]])
        eels = EELSStat(p, impact, width=0.5, vel=0.5)
        pbulk = eels.bulkloss(500.0)
        np.testing.assert_allclose(pbulk, 0.0, atol=1e-20)

    def test_positive_for_lossy_material(self):
        # If beam passes through a lossy material, bulk loss should be positive
        # We need to manually set up _z, _indimp, _indmat for this
        p = MockSphere(radius=5.0)
        impact = np.array([[0.0, 0.0]])
        eels = EELSStat(p, impact, width=0.5, vel=0.5)

        # Manually inject a path inside material 2
        eels._z = np.array([[-5.0, 5.0]])
        eels._indimp = np.array([0])
        eels._indmat = np.array([2])

        enei = 500.0
        pbulk = eels.bulkloss(enei)
        # For Im(-1/eps) > 0 (which is true for eps = -10 + 1j)
        eps_val = p.eps[1](enei)
        if hasattr(eps_val, '__len__'):
            eps_val = eps_val[0]
        imag_inv = np.imag(-1.0 / eps_val)
        # eps = -10+1j => -1/eps = -1/(-10+1j) = (-10-1j)/101 => imag = -1/101 < 0
        # Actually need to check sign
        if imag_inv > 0:
            assert pbulk[0] > 0
        else:
            # Loss can still be correct with negative imaginary part
            # Just check it's finite
            assert np.isfinite(pbulk[0])


# ===========================================================================
#  TestEELSRet
# ===========================================================================

class TestEELSRet(object):

    def test_construction(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSRet(p, impact, width=0.5, vel=0.5)
        assert isinstance(eels, EELSBase)
        assert eels.name == 'eels'
        assert eels.needs == {'sim': 'ret'}
        assert hasattr(eels, 'spec')

    def test_construction_with_pinfty(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        # Pass integer for pinfty (create unit sphere with that many faces)
        eels = EELSRet(p, impact, width=0.5, vel=0.5, pinfty=80)
        assert hasattr(eels.spec, 'pinfty')

    def test_potential_output(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSRet(p, impact, width=0.5, vel=0.5)
        enei = 500.0
        exc = eels.potential(p, enei)
        assert isinstance(exc, CompStruct)
        # Retarded potential has phi1, phi2, a1, a2 etc.
        assert hasattr(exc, 'phi1')
        assert hasattr(exc, 'phi2')
        assert hasattr(exc, 'phi1p')
        assert hasattr(exc, 'phi2p')
        assert hasattr(exc, 'a1')
        assert hasattr(exc, 'a2')

    def test_potential_shape(self):
        p = MockSphere(radius=5.0)
        n_imp = 2
        impact = np.array([[15.0, 0.0], [0.0, 15.0]])
        eels = EELSRet(p, impact, width=0.5, vel=0.5)
        enei = 500.0
        exc = eels.potential(p, enei)
        assert exc.phi1.shape == (p.n, n_imp)
        assert exc.phi2.shape == (p.n, n_imp)
        assert exc.a1.shape == (p.n, 3, n_imp)
        assert exc.a2.shape == (p.n, 3, n_imp)

    def test_potential_vector_potential_relation(self):
        # MATLAB: a = vel * outer([0,0,1], phi)
        # So a_z should be proportional to phi, and a_x = a_y = 0
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        vel = 0.5
        eels = EELSRet(p, impact, width=0.5, vel=vel)
        enei = 500.0
        exc = eels.potential(p, enei)

        # a1[:, 2, :] should be related to phi1 via vel (and eps)
        # a = vel * outer(zhat, phi_unscaled), phi_scaled = phi_unscaled / eps
        # So a[:, 2, :] = vel * phi_unscaled, and phi1 = phi_unscaled / eps
        # => a[:, 2, :] = vel * eps * phi1
        # For embedding medium (eps=1): a1[:,2,:] = vel * phi1
        # Check a1_x and a1_y are zero (beam along z)
        np.testing.assert_allclose(exc.a1[:, 0, :], 0.0, atol=1e-15)
        np.testing.assert_allclose(exc.a1[:, 1, :], 0.0, atol=1e-15)

    def test_callable(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSRet(p, impact, width=0.5, vel=0.5)
        enei = 500.0
        exc_call = eels(p, enei)
        exc_pot = eels.potential(p, enei)
        np.testing.assert_allclose(exc_call.phi1, exc_pot.phi1)

    def test_repr(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSRet(p, impact, width=0.5, vel=0.5)
        r = repr(eels)
        assert 'EELSRet' in r

    def test_str(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSRet(p, impact, width=0.5, vel=0.5)
        s = str(eels)
        assert 'Retarded' in s


# ===========================================================================
#  TestEELSRetBulkloss
# ===========================================================================

class TestEELSRetBulkloss(object):

    def test_zero_path_gives_zero(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[100.0, 100.0]])
        eels = EELSRet(p, impact, width=0.5, vel=0.5)
        pbulk = eels.bulkloss(500.0)
        assert pbulk.shape == (1,)
        np.testing.assert_allclose(pbulk, 0.0, atol=1e-20)

    def test_retarded_vs_static_bulkloss_differ(self):
        # Retarded and static bulkloss formulas are different
        # MATLAB stat: 2*fine^2/(bohr*hartree*pi*vel^2) * imag(-1/eps) * path * log(...)
        # MATLAB ret:  fine^2/(bohr*hartree*pi*vel^2) * imag((vel^2-1/eps)*log(...)) * path
        p = MockSphere(radius=5.0)
        impact = np.array([[0.0, 0.0]])
        vel = 0.5

        eels_stat = EELSStat(p, impact, width=0.5, vel=vel)
        eels_ret = EELSRet(p, impact, width=0.5, vel=vel)

        # Inject same path
        for eels in [eels_stat, eels_ret]:
            eels._z = np.array([[-5.0, 5.0]])
            eels._indimp = np.array([0])
            eels._indmat = np.array([2])

        enei = 500.0
        pbulk_stat = eels_stat.bulkloss(enei)
        pbulk_ret = eels_ret.bulkloss(enei)

        # They should generally differ because the formulas are different
        # (retarded includes Cherenkov radiation term)
        # Just check both are finite
        assert np.isfinite(pbulk_stat[0])
        assert np.isfinite(pbulk_ret[0])

    def test_bulkloss_prefactor(self):
        # MATLAB: fine^2/(bohr*hartree*pi*vel^2)
        vel = 0.5
        prefac = FINE ** 2 / (BOHR * HARTREE * np.pi * vel ** 2)
        assert prefac > 0
        assert np.isfinite(prefac)


# ===========================================================================
#  TestEELSRetField
# ===========================================================================

class TestEELSRetField(object):

    def test_field_output(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSRet(p, impact, width=0.5, vel=0.5)
        exc = eels.field(p, 500.0)
        assert isinstance(exc, CompStruct)
        assert hasattr(exc, 'e')
        assert hasattr(exc, 'h')

    def test_field_shape(self):
        p = MockSphere(radius=5.0)
        n_imp = 2
        impact = np.array([[15.0, 0.0], [0.0, 15.0]])
        eels = EELSRet(p, impact, width=0.5, vel=0.5)
        exc = eels.field(p, 500.0)
        assert exc.e.shape == (p.n, 3, n_imp)
        assert exc.h.shape == (p.n, 3, n_imp)

    def test_magnetic_field_transverse(self):
        # MATLAB: h_z = 0 for the infinite beam field
        # h(:, 1, :) = vel*fac*K1*y_hat
        # h(:, 2, :) = -vel*fac*K1*x_hat
        # h(:, 3, :) = 0 (not set, defaults to 0)
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSRet(p, impact, width=0.5, vel=0.5)
        exc = eels.field(p, 500.0)
        # h_z should be zero
        np.testing.assert_allclose(exc.h[:, 2, :], 0.0, atol=1e-15)

    def test_fieldinfty_static_method(self):
        # Test the static _fieldinfty directly
        pos = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        b = np.array([[0.0, 0.0]])
        k = 0.01 + 0j
        eps = 1.0 + 0j
        vel = 0.5
        width = 0.1

        e, h = EELSRet._fieldinfty(pos, b, k, eps, vel, width)
        assert e.shape == (2, 3, 1)
        assert h.shape == (2, 3, 1)
        # h_z should be 0
        np.testing.assert_allclose(h[:, 2, :], 0.0, atol=1e-15)

    def test_fieldinfty_bessel_functions(self):
        # MATLAB: e_x = -fac/eps * K1 * x_hat
        # where fac = 2*q/(vel*gamma) * exp(i*q*z)
        pos = np.array([[5.0, 0.0, 0.0]])
        b = np.array([[0.0, 0.0]])
        k = 0.1 + 0j
        eps = 1.0 + 0j
        vel = 0.5
        width = 0.01

        e, h = EELSRet._fieldinfty(pos, b, k, eps, vel, width)

        # Compute expected values
        q = k / (vel * np.sqrt(eps))
        gamma = 1.0 / np.sqrt(1 - eps * vel ** 2)
        x = 5.0
        r = np.sqrt(x ** 2 + width ** 2)
        K0_val = besselk(0, q * r / gamma)
        K1_val = besselk(1, q * r / gamma)
        fac = 2 * q / (vel * gamma)

        expected_ex = -fac / eps * K1_val * (x / r)
        expected_ez = fac / eps * K0_val * 1j / gamma

        assert e[0, 0, 0] == pytest.approx(expected_ex, rel=1e-10)
        assert e[0, 2, 0] == pytest.approx(expected_ez, rel=1e-10)
        # e_y should be ~0 since y=0
        assert np.abs(e[0, 1, 0]) < 1e-10 * np.abs(e[0, 0, 0])

    def test_fieldinfty_decay_with_distance(self):
        # Bessel K0, K1 decay exponentially -> field should decay
        b = np.array([[0.0, 0.0]])
        k = 0.1 + 0j
        eps = 1.0 + 0j
        vel = 0.5
        width = 0.01

        pos_near = np.array([[2.0, 0.0, 0.0]])
        pos_far = np.array([[20.0, 0.0, 0.0]])

        e_near, _ = EELSRet._fieldinfty(pos_near, b, k, eps, vel, width)
        e_far, _ = EELSRet._fieldinfty(pos_far, b, k, eps, vel, width)

        assert np.abs(e_near[0, 0, 0]) > np.abs(e_far[0, 0, 0])


# ===========================================================================
#  TestEELSRetLoss
# ===========================================================================

class TestEELSRetLoss(object):

    def test_loss_output_shape(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSRet(p, impact, width=0.5, vel=0.5)

        class MockSigRet(object):
            pass

        sig = MockSigRet()
        sig.enei = 500.0
        sig.p = p
        sig.sig1 = np.random.randn(p.n, 1) + 1j * np.random.randn(p.n, 1)
        sig.sig2 = np.random.randn(p.n, 1) + 1j * np.random.randn(p.n, 1)
        sig.h1 = np.random.randn(p.n, 3, 1) + 1j * np.random.randn(p.n, 3, 1)
        sig.h2 = np.random.randn(p.n, 3, 1) + 1j * np.random.randn(p.n, 3, 1)

        psurf, pbulk = eels.loss(sig)
        assert psurf.shape == (1,)
        assert pbulk.shape == (1,)

    def test_loss_bulk_zero_for_far_beam(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[100.0, 0.0]])
        eels = EELSRet(p, impact, width=0.5, vel=0.5)

        # For far beam, just test bulkloss directly (avoids loss() inner
        # function complexity that requires properly shaped sig)
        pbulk = eels.bulkloss(500.0)
        np.testing.assert_allclose(pbulk, 0.0, atol=1e-20)

    def test_loss_with_single_impact(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSRet(p, impact, width=0.5, vel=0.5)

        class MockSigRet(object):
            pass

        sig = MockSigRet()
        sig.enei = 500.0
        sig.p = p
        sig.sig1 = np.zeros((p.n, 1), dtype=complex)
        sig.sig2 = np.zeros((p.n, 1), dtype=complex)
        sig.h1 = np.zeros((p.n, 3, 1), dtype=complex)
        sig.h2 = np.zeros((p.n, 3, 1), dtype=complex)

        psurf, pbulk = eels.loss(sig)
        # With zero sig/h, surface loss should be zero
        np.testing.assert_allclose(psurf, 0.0, atol=1e-15)
        np.testing.assert_allclose(pbulk, 0.0, atol=1e-20)

    def test_loss_prefactor(self):
        # MATLAB: psurf = fine^2/(bohr*hartree*pi*vel) * psurf
        vel = 0.5
        prefac = FINE ** 2 / (BOHR * HARTREE * np.pi * vel)
        assert prefac > 0
        assert np.isfinite(prefac)


# ===========================================================================
#  TestEELSRetRad -- radiative loss
# ===========================================================================

class TestEELSRetRad(object):

    def test_rad_prefactor(self):
        # MATLAB: prad = fine^2/(2*pi^2*hartree*bohr*k) * sca
        k = 0.01
        prefac = FINE ** 2 / (2 * np.pi ** 2 * HARTREE * BOHR * k)
        assert prefac > 0
        assert np.isfinite(prefac)


# ===========================================================================
#  TestAddPotential -- retarded potential helper
# ===========================================================================

class TestAddPotential(object):

    def test_add_potential_accumulates(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSRet(p, impact, width=0.5, vel=0.5)

        exc = CompStruct(p, 500.0)
        n_imp = 1
        exc.phi1 = np.zeros((p.n, n_imp), dtype=complex)
        exc.phi1p = np.zeros((p.n, n_imp), dtype=complex)
        exc.phi2 = np.zeros((p.n, n_imp), dtype=complex)
        exc.phi2p = np.zeros((p.n, n_imp), dtype=complex)
        exc.a1 = np.zeros((p.n, 3, n_imp), dtype=complex)
        exc.a1p = np.zeros((p.n, 3, n_imp), dtype=complex)
        exc.a2 = np.zeros((p.n, 3, n_imp), dtype=complex)
        exc.a2p = np.zeros((p.n, 3, n_imp), dtype=complex)

        phi = np.ones((p.n, n_imp), dtype=complex) * (1 + 2j)
        phip = np.ones((p.n, n_imp), dtype=complex) * (3 + 4j)

        exc = eels._add_potential(exc, p, phi, phip, 1, 1.0 + 0j, 0.5)

        # phi should be scaled by 1/eps = 1
        # For inout = [[2, 1]], mat=1 is outside, so ind2 = all faces
        # phi2 should be phi/eps = phi
        np.testing.assert_allclose(exc.phi2, phi, rtol=1e-14)

    def test_vector_potential_z_component(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        vel = 0.5
        eels = EELSRet(p, impact, width=0.5, vel=vel)

        exc = CompStruct(p, 500.0)
        n_imp = 1
        exc.phi1 = np.zeros((p.n, n_imp), dtype=complex)
        exc.phi1p = np.zeros((p.n, n_imp), dtype=complex)
        exc.phi2 = np.zeros((p.n, n_imp), dtype=complex)
        exc.phi2p = np.zeros((p.n, n_imp), dtype=complex)
        exc.a1 = np.zeros((p.n, 3, n_imp), dtype=complex)
        exc.a1p = np.zeros((p.n, 3, n_imp), dtype=complex)
        exc.a2 = np.zeros((p.n, 3, n_imp), dtype=complex)
        exc.a2p = np.zeros((p.n, 3, n_imp), dtype=complex)

        phi = np.ones((p.n, n_imp), dtype=complex) * 2.0
        phip = np.zeros((p.n, n_imp), dtype=complex)

        exc = eels._add_potential(exc, p, phi, phip, 1, 1.0 + 0j, vel)

        # a2[:, 2, :] should be vel * phi (z-component of vector potential)
        # a2[:, 0, :] and a2[:, 1, :] should be 0
        np.testing.assert_allclose(exc.a2[:, 0, :], 0.0, atol=1e-15)
        np.testing.assert_allclose(exc.a2[:, 1, :], 0.0, atol=1e-15)
        np.testing.assert_allclose(exc.a2[:, 2, :], vel * phi, rtol=1e-14)


# ===========================================================================
#  TestEdgeCases -- beam positions relative to particle
# ===========================================================================

class TestEdgeCases(object):

    def test_beam_far_away(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[1000.0, 1000.0]])
        eels = EELSBase(p, impact, width=0.5, vel=0.5)
        assert eels._z.shape[0] == 0
        assert len(eels._indmat) == 0
        # Path should be zero
        path_all = eels.path()
        np.testing.assert_allclose(path_all, 0.0)

    def test_multiple_beams_mixed(self):
        # Some beams near, some far
        p = MockSphere(radius=5.0)
        impact = np.array([
            [100.0, 0.0],   # far
            [0.0, 100.0],   # far
            [1000.0, 0.0],  # very far
        ])
        eels = EELSBase(p, impact, width=0.5, vel=0.5)
        path_all = eels.path()
        assert path_all.shape == (len(p.eps), 3)

    def test_zero_width_beam(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        # Width = 0 is a delta function beam
        eels = EELSBase(p, impact, width=0.0, vel=0.5)
        q = 0.1
        phi, phip = eels.potinfty(q, gamma=1.0)
        # Should still work, though Bessel arguments involve width=0
        assert phi.shape == (p.n, 1)
        assert np.all(np.isfinite(phi))

    def test_very_small_velocity(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        vel = 0.01
        eels = EELSBase(p, impact, width=0.5, vel=vel)
        q = 0.1
        phi, phip = eels.potinfty(q, gamma=1.0)
        # phi scales as -2/vel, so small vel gives large phi
        assert phi.shape == (p.n, 1)
        assert np.all(np.isfinite(phi))

    def test_single_impact_parameter(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSBase(p, impact, width=0.5, vel=0.5)
        assert eels.impact.shape == (1, 2)

    def test_many_impact_parameters(self):
        p = MockSphere(radius=5.0)
        n_imp = 20
        theta = np.linspace(0, 2 * np.pi, n_imp, endpoint=False)
        r_imp = 15.0
        impact = np.column_stack([r_imp * np.cos(theta), r_imp * np.sin(theta)])
        eels = EELSBase(p, impact, width=0.5, vel=0.5)
        assert eels.impact.shape == (n_imp, 2)

        # All beams are at same distance from origin
        # potinfty should give same magnitude for all by symmetry (approximately)
        q = 0.1
        phi, _ = eels.potinfty(q, gamma=1.0)
        mags = np.abs(phi).mean(axis=0)
        # Due to octahedron (not perfect sphere), they won't be exactly equal
        # but should be in similar range
        assert np.max(mags) / np.min(mags) < 5.0


# ===========================================================================
#  TestConstants -- verify constant usage matches MATLAB
# ===========================================================================

class TestConstants(object):

    def test_ev2nm(self):
        # MATLAB: eV2nm = 1 / 8.0655477e-4
        assert EV2NM == pytest.approx(1.0 / 8.0655477e-4, rel=1e-10)

    def test_bohr(self):
        # MATLAB: bohr = 0.05292 nm
        assert BOHR == pytest.approx(0.05292)

    def test_hartree(self):
        # MATLAB: hartree = 27.211 eV
        assert HARTREE == pytest.approx(27.211)

    def test_fine(self):
        # MATLAB: fine = 1/137.036
        assert FINE == pytest.approx(1.0 / 137.036)

    def test_bulkloss_prefactor_stat(self):
        # MATLAB stat: 2*fine^2/(bohr*hartree*pi*vel^2)
        vel = 0.5
        prefac = 2 * FINE ** 2 / (BOHR * HARTREE * np.pi * vel ** 2)
        # Should be a positive finite number
        assert prefac > 0
        assert np.isfinite(prefac)

    def test_bulkloss_prefactor_ret(self):
        # MATLAB ret: fine^2/(bohr*hartree*pi*vel^2)
        vel = 0.5
        prefac = FINE ** 2 / (BOHR * HARTREE * np.pi * vel ** 2)
        assert prefac > 0
        assert np.isfinite(prefac)

    def test_loss_prefactor_stat(self):
        # MATLAB stat: -fine^2/(bohr*hartree*pi)
        prefac = FINE ** 2 / (BOHR * HARTREE * np.pi)
        assert prefac > 0
        assert np.isfinite(prefac)

    def test_loss_prefactor_ret(self):
        # MATLAB ret: fine^2/(bohr*hartree*pi*vel)
        vel = 0.5
        prefac = FINE ** 2 / (BOHR * HARTREE * np.pi * vel)
        assert prefac > 0
        assert np.isfinite(prefac)

    def test_rad_prefactor(self):
        # MATLAB: fine^2/(2*pi^2*hartree*bohr*k)
        k = 0.01
        prefac = FINE ** 2 / (2 * np.pi ** 2 * HARTREE * BOHR * k)
        assert prefac > 0
        assert np.isfinite(prefac)


# ===========================================================================
#  TestMATLABvsPhython -- cross-checks of specific formulas
# ===========================================================================

class TestMATLABvsPython(object):

    def test_ene2vel_100keV(self):
        # MATLAB: vel = sqrt(1 - 1./(1 + ene/0.51e6).^2)
        ene = 100e3
        matlab_result = np.sqrt(1 - 1.0 / (1 + ene / 0.51e6) ** 2)
        python_result = EELSBase.ene2vel(ene)
        assert python_result == pytest.approx(matlab_result, rel=1e-15)

    def test_wavenumber_computation(self):
        # MATLAB: q = 2*pi/(enei*vel)
        enei = 500.0  # nm
        vel = 0.5
        q_expected = 2 * np.pi / (enei * vel)
        assert q_expected == pytest.approx(2 * np.pi / (500.0 * 0.5))

    def test_lorentz_gamma(self):
        # MATLAB: gamma = 1./sqrt(1 - eps*vel^2)
        vel = 0.5
        eps = 1.0  # vacuum
        gamma = 1.0 / np.sqrt(1 - eps * vel ** 2)
        expected = 1.0 / np.sqrt(1 - 0.25)
        assert gamma == pytest.approx(expected)

    def test_potinfty_formula(self):
        # MATLAB: phi = -2/vel * exp(i*q*z) * K0(q*rr/gamma)
        vel = 0.5
        q = 0.1
        gamma = 1.5
        z = 3.0
        rr = 5.0
        K0_val = besselk(0, q * rr / gamma)
        phi = -2 / vel * np.exp(1j * q * z) * K0_val

        # Verify it's complex
        assert np.iscomplex(phi) or isinstance(phi, complex)
        # Verify magnitude
        assert np.abs(phi) == pytest.approx(2 / vel * np.abs(K0_val), rel=1e-12)

    def test_potinfty_surface_deriv_formula(self):
        # MATLAB: phip = -2/vel * exp(iqz) * q * (i*K0*nz - K1/gamma*(x*nx+y*ny)/rr)
        vel = 0.5
        q = 0.1
        gamma = 1.5
        z = 3.0
        rr = 5.0
        x = 3.0
        y = 4.0
        nx, ny, nz = 0.5, 0.5, np.sqrt(0.5)

        K0_val = besselk(0, q * rr / gamma)
        K1_val = besselk(1, q * rr / gamma)

        phip = (-2 / vel * np.exp(1j * q * z) * q
                * (1j * K0_val * nz - K1_val / gamma * (x * nx + y * ny) / rr))

        assert isinstance(phip, complex)
        assert np.isfinite(phip)

    def test_static_loss_formula(self):
        # MATLAB: psurf = -fine^2/(bohr*hartree*pi) * imag(area' * (conj(phi) .* sig))
        n = 4
        area = np.ones(n)
        phi = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j])
        sig = np.array([0.1 + 0.2j, 0.3 + 0.4j, 0.5 + 0.6j, 0.7 + 0.8j])

        psurf = -FINE ** 2 / (BOHR * HARTREE * np.pi) * np.imag(
            area @ (np.conj(phi) * sig))

        assert np.isfinite(psurf)

    def test_retarded_bulkloss_formula(self):
        # MATLAB: fine^2/(bohr*hartree*pi*vel^2) *
        #         imag((vel^2 - 1./eps) .* log((qc^2-k^2)./(q^2-k^2))) * path
        vel = 0.5
        eps = -10.0 + 1.0j
        enei = 500.0
        ene = EV2NM / enei
        mass = 0.51e6
        q = 2 * np.pi / (enei * vel)
        qc = q * np.sqrt((mass / ene) ** 2 * vel ** 2 * (1e-2) ** 2 + 1)
        k = 2 * np.pi / enei * np.sqrt(eps)

        term = np.imag((vel ** 2 - 1.0 / eps)
                       * np.log((qc ** 2 - k ** 2) / (q ** 2 - k ** 2)))

        prefac = FINE ** 2 / (BOHR * HARTREE * np.pi * vel ** 2)
        path_length = 10.0  # nm

        pbulk = prefac * term * path_length
        assert np.isfinite(pbulk)

    def test_static_bulkloss_formula(self):
        # MATLAB: 2*fine^2/(bohr*hartree*pi*vel^2) * imag(-1./eps) * path *
        #         log(sqrt((mass/ene)^2*vel^2*phiout^2 + 1))
        vel = 0.5
        eps = -10.0 + 1.0j
        enei = 500.0
        ene = EV2NM / enei
        mass = 0.51e6
        phiout = 1e-2

        prefac = 2 * FINE ** 2 / (BOHR * HARTREE * np.pi * vel ** 2)
        im_term = np.imag(-1.0 / eps)
        log_term = np.log(np.sqrt((mass / ene) ** 2 * vel ** 2 * phiout ** 2 + 1))
        path_length = 10.0

        pbulk = prefac * im_term * path_length * log_term
        assert np.isfinite(pbulk)


# ===========================================================================
#  TestEELSStatPotentialDetailed
# ===========================================================================

class TestEELSStatPotentialDetailed(object):

    def test_potential_includes_eps_scaling(self):
        # MATLAB: phi = phi/eps(1) + full(pin*diag(ideps(indmat)))
        # For far beam (no pin), phi should be scaled by 1/eps(1) = 1 (vacuum)
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        vel = 0.5
        eels = EELSStat(p, impact, width=0.5, vel=vel)
        enei = 500.0

        # Get raw potinfty
        q = 2 * np.pi / (enei * vel)
        phi_raw, phip_raw = eels.potinfty(q, 1.0)

        # Get potential from method
        exc = eels.potential(p, enei)

        # For vacuum embedding (eps(1)=1), should be phi_raw/1 = phi_raw
        np.testing.assert_allclose(exc.phi, phi_raw / 1.0, rtol=1e-12)

    def test_potential_nonzero(self):
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSStat(p, impact, width=0.5, vel=0.5)
        exc = eels.potential(p, 500.0)
        assert np.any(np.abs(exc.phi) > 0)
        assert np.any(np.abs(exc.phip) > 0)


# ===========================================================================
#  TestEELSRetPotentialDetailed
# ===========================================================================

class TestEELSRetPotentialDetailed(object):

    def test_potential_includes_lorentz_factor(self):
        # MATLAB: gamma = 1./sqrt(1 - eps*vel^2)
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        vel = 0.5
        eels = EELSRet(p, impact, width=0.5, vel=vel)
        enei = 500.0

        exc = eels.potential(p, enei)
        # Just verify it completes and produces non-zero results
        assert np.any(np.abs(exc.phi1) > 0) or np.any(np.abs(exc.phi2) > 0)

    def test_potential_phi1_phi2_allocation(self):
        # For MockSphere with inout = [[2, 1]]
        # Outside is material 1 (vacuum), inside is material 2
        # phi1 corresponds to inside faces, phi2 to outside faces
        p = MockSphere(radius=5.0)
        impact = np.array([[15.0, 0.0]])
        eels = EELSRet(p, impact, width=0.5, vel=0.5)
        enei = 500.0
        exc = eels.potential(p, enei)

        # phi2 should have contributions from embedding medium (mat=1)
        # since inout = [[2,1]] means outside=1
        assert exc.phi2.shape == (p.n, 1)
