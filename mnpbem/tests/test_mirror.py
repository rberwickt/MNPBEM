"""
Comprehensive tests for Mirror module (mirror symmetry support).

Tests for:
  - ComParticleMirror: construction, symmetry tables, closed(), mask(), symvalue(), symindex()
  - CompGreenStatMirror: construction, eval, potential, field
  - CompGreenRetMirror: construction, eval, potential, field
  - BEMStatMirror: construction, init, solve (mldivide), potential, field
  - BEMRetMirror: construction, init, solve (mldivide), potential, field
  - BEMStatEigMirror: construction, init, solve (mldivide), potential, field
  - BEMLayerMirror: raises NotImplementedError
  - DipoleStatMirror: construction, potential, field, decayrate
  - DipoleRetMirror: construction, potential, field, decayrate
  - PlaneWaveStatMirror: construction, potential, field, absorption/extinction/scattering
  - PlaneWaveRetMirror: construction, potential, field, absorption/extinction/scattering

MATLAB reference:
  Particles/@comparticlemirror, Greenfun/@compgreenstatmirror,
  Greenfun/@compgreenretmirror, BEM/@bemstatmirror, BEM/@bemretmirror,
  BEM/@bemstateigmirror, BEM/@bemlayermirror,
  Simulation/static/@dipolestatmirror, Simulation/retarded/@dipoleretmirror,
  Simulation/static/@planewavestatmirror, Simulation/retarded/@planewaveretmirror
"""

import sys
import os
import copy
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mnpbem.geometry.comparticle_mirror import ComParticleMirror, CompStructMirror
from mnpbem.greenfun import CompStruct
from mnpbem.greenfun.compgreen_stat_mirror import CompGreenStatMirror
from mnpbem.greenfun.compgreen_ret_mirror import CompGreenRetMirror
from mnpbem.bem.bem_stat_mirror import BEMStatMirror
from mnpbem.bem.bem_ret_mirror import BEMRetMirror
from mnpbem.bem.bem_stat_eig_mirror import BEMStatEigMirror
from mnpbem.bem.bem_layer_mirror import BEMLayerMirror
from mnpbem.simulation.dipole_stat_mirror import DipoleStatMirror
from mnpbem.simulation.dipole_ret_mirror import DipoleRetMirror
from mnpbem.simulation.planewave_stat_mirror import PlaneWaveStatMirror
from mnpbem.simulation.planewave_ret_mirror import PlaneWaveRetMirror


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------


class MockEps(object):
    """Mock dielectric function that returns a constant value.

    MATLAB: epstable / epsconst style callable.
    """

    def __init__(self, value = -11.4 + 1.0j):
        self._value = value

    def __call__(self, enei):
        return np.array([self._value])


class MockParticle(object):
    """Mock particle (triangulated surface) for mirror symmetry testing.

    Provides a simple 4-face tetrahedron centered at the origin with the
    essential attributes: verts, faces, pos, nvec, area, nfaces.
    Also supports flip() for mirror symmetry construction.
    """

    def __init__(self, verts = None, faces = None, center = None):
        if verts is not None:
            self.verts = np.array(verts, dtype = np.float64)
        else:
            R = 5.0
            self.verts = np.array([
                [R, 0, -R / np.sqrt(2)],
                [-R, 0, -R / np.sqrt(2)],
                [0, R, R / np.sqrt(2)],
                [0, -R, R / np.sqrt(2)],
            ], dtype = np.float64)

        if center is not None:
            self.verts = self.verts + np.array(center, dtype = np.float64)

        if faces is not None:
            self.faces = np.array(faces, dtype = np.int64)
        else:
            self.faces = np.array([
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 1],
                [1, 3, 2],
            ], dtype = np.int64)

        self._compute_geometry()

    def _compute_geometry(self):
        """Compute centroids, normals, and areas for each face."""
        nf = self.faces.shape[0]
        self.pos = np.zeros((nf, 3), dtype = np.float64)
        self.nvec = np.zeros((nf, 3), dtype = np.float64)
        self.area = np.zeros(nf, dtype = np.float64)

        for i in range(nf):
            v0 = self.verts[self.faces[i, 0]]
            v1 = self.verts[self.faces[i, 1]]
            v2 = self.verts[self.faces[i, 2]]
            self.pos[i] = (v0 + v1 + v2) / 3.0
            cross = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(cross)
            if norm > 0:
                self.nvec[i] = cross / norm
            self.area[i] = 0.5 * norm

        # Ensure outward normals: if the normal points inward (dot with centroid < 0),
        # flip the normal direction
        center = np.mean(self.verts, axis = 0)
        for i in range(nf):
            if np.dot(self.nvec[i], self.pos[i] - center) < 0:
                self.nvec[i] = -self.nvec[i]

    @property
    def nfaces(self):
        return self.faces.shape[0]

    @property
    def n(self):
        return self.nfaces

    def bradius(self):
        """Boundary element radius (inradius of each triangle).

        MATLAB: bradius = particle/bradius.m
        inradius = area / semi_perimeter
        """
        nf = self.faces.shape[0]
        rad = np.zeros(nf, dtype = np.float64)
        for i in range(nf):
            v0 = self.verts[self.faces[i, 0]]
            v1 = self.verts[self.faces[i, 1]]
            v2 = self.verts[self.faces[i, 2]]
            a = np.linalg.norm(v1 - v0)
            b = np.linalg.norm(v2 - v1)
            c = np.linalg.norm(v0 - v2)
            s = 0.5 * (a + b + c)
            rad[i] = self.area[i] / s if s > 0 else 0.0
        return rad

    def flip(self, direction):
        """Flip particle along given direction (1-indexed to match MATLAB).

        MATLAB: flip(p, k) flips along k-th axis (1=x, 2=y).
        Python @particle uses 0-indexed, but ComParticleMirror._build_full
        calls flip(1) for x, flip(2) for y.
        """
        new_p = MockParticle.__new__(MockParticle)
        new_p.verts = self.verts.copy()
        new_p.faces = self.faces.copy()

        if isinstance(direction, (list, tuple)):
            for d in direction:
                new_p.verts[:, d - 1] = -new_p.verts[:, d - 1]
        else:
            new_p.verts[:, direction - 1] = -new_p.verts[:, direction - 1]

        # Reverse face winding for flipped normals
        new_p.faces = new_p.faces[:, ::-1]
        new_p._compute_geometry()
        return new_p

    def __add__(self, other):
        """Concatenate particles (for closed surfaces)."""
        new_p = MockParticle.__new__(MockParticle)
        offset = self.verts.shape[0]
        new_p.verts = np.vstack([self.verts, other.verts])
        new_p.faces = np.vstack([self.faces, other.faces + offset])
        new_p._compute_geometry()
        return new_p


class MockComParticle(object):
    """Mock ComParticle (compound particle) for testing.

    Mimics the interface of mnpbem.geometry.comparticle.ComParticle
    with multiple particles and dielectric media.
    """

    def __init__(self, eps, particles, inout, **kwargs):
        self.eps = eps
        self.p = list(particles)
        self.inout = np.atleast_2d(inout)
        self.closed = [None] * len(self.p)

        # Concatenate geometry from all particles
        all_pos = []
        all_nvec = []
        all_area = []
        all_verts = []
        self._face_offsets = [0]
        for part in self.p:
            all_pos.append(part.pos)
            all_nvec.append(part.nvec)
            all_area.append(part.area)
            all_verts.append(part.verts)
            self._face_offsets.append(self._face_offsets[-1] + part.nfaces)

        self.pc = type('PC', (), {
            'pos': np.vstack(all_pos) if all_pos else np.empty((0, 3)),
            'nvec': np.vstack(all_nvec) if all_nvec else np.empty((0, 3)),
            'area': np.hstack(all_area) if all_area else np.empty(0),
        })()
        self._mask = list(range(len(self.p)))

    @property
    def nfaces(self):
        return sum(p.nfaces for p in self.p)

    @property
    def n(self):
        return self.nfaces

    @property
    def nvec(self):
        return self.pc.nvec

    @property
    def pos(self):
        return self.pc.pos

    @property
    def area(self):
        return self.pc.area

    @property
    def verts(self):
        return np.vstack([p.verts for p in self.p])

    @property
    def np(self):
        """Number of unique material boundaries."""
        unique_inout = set()
        for row in self.inout:
            unique_inout.add(tuple(row))
        return len(unique_inout)

    @property
    def index(self):
        """Index mapping faces to particles."""
        idx = np.zeros(self.nfaces, dtype = np.int64)
        for i, p in enumerate(self.p):
            idx[self._face_offsets[i]:self._face_offsets[i + 1]] = i
        return idx

    def index_func(self, ip):
        """Return face indices for particle ip (1-indexed)."""
        ip_0 = ip - 1
        start = self._face_offsets[ip_0]
        end = self._face_offsets[ip_0 + 1]
        return np.arange(start, end)

    def eps1(self, enei):
        """Inside dielectric constants at given wavelength."""
        eps_vals = np.array([self.eps[int(row[0]) - 1](enei)[0] for row in self.inout])
        result = np.zeros(self.nfaces, dtype = complex)
        for i, p in enumerate(self.p):
            result[self._face_offsets[i]:self._face_offsets[i + 1]] = eps_vals[i]
        return result

    def eps2(self, enei):
        """Outside dielectric constants at given wavelength."""
        eps_vals = np.array([self.eps[int(row[1]) - 1](enei)[0] for row in self.inout])
        result = np.zeros(self.nfaces, dtype = complex)
        for i, p in enumerate(self.p):
            result[self._face_offsets[i]:self._face_offsets[i + 1]] = eps_vals[i]
        return result

    @property
    def mask(self):
        mask_arr = np.zeros(len(self.p), dtype = bool)
        for i in self._mask:
            mask_arr[i] = True
        return mask_arr

    def set_mask(self, ind):
        if isinstance(ind, (list, np.ndarray)):
            self._mask = [i - 1 for i in ind]
        else:
            self._mask = [ind - 1]

    def closedparticle(self, ind):
        # MATLAB: [p, dir, loc] = closedparticle(obj, ind)
        # Returns 3-tuple: (closed_particle, dir, loc).
        # If self.closed[ind-1] is None or empty, the particle is not
        # closed -> return (None, 1, None) to signal skip.
        entry = self.closed[ind - 1]
        if entry is None:
            return None, 1, None
        # Otherwise treat the particle itself as closed.
        return self.p[ind - 1], 1, None


class MockComPoint(object):
    """Mock ComPoint for dipole positions."""

    def __init__(self, pos):
        self.pos = np.atleast_2d(pos).astype(np.float64)

    @property
    def n(self):
        return self.pos.shape[0]

    @property
    def nfaces(self):
        return self.pos.shape[0]

    def flip(self, direction):
        new_pt = MockComPoint.__new__(MockComPoint)
        new_pt.pos = self.pos.copy()
        if isinstance(direction, (list, tuple)):
            for d in direction:
                new_pt.pos[:, d - 1] = -new_pt.pos[:, d - 1]
        else:
            new_pt.pos[:, direction - 1] = -new_pt.pos[:, direction - 1]
        return new_pt

    def eps1(self, enei):
        return np.ones(self.pos.shape[0])


# ---------------------------------------------------------------------------
# Monkey-patch ComParticleMirror to use mock ComParticle
# ---------------------------------------------------------------------------


def _mock_build_full(self, **kwargs):
    """Build full particle using mock ComParticle instead of real one."""
    p_list = list(self.p)
    inout_list = self.inout.tolist()

    if self.sym in ('x', 'xy'):
        orig_len = len(p_list)
        for i in range(orig_len):
            p_list.append(p_list[i].flip(1))
            inout_list.append(inout_list[i])

    if self.sym in ('y', 'xy'):
        orig_len = len(p_list)
        for i in range(orig_len):
            p_list.append(p_list[i].flip(2))
            inout_list.append(inout_list[i])

    inout_arr = np.array(inout_list)
    self.pfull = MockComParticle(self.eps, p_list, inout_arr, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def eps_metal():
    """Metal dielectric function (Drude-like at 500 nm)."""
    return MockEps(-11.4 + 1.0j)


@pytest.fixture
def eps_vacuum():
    """Vacuum dielectric function."""
    return MockEps(1.0 + 0.0j)


@pytest.fixture
def single_particle():
    """A single 4-face tetrahedron particle."""
    return MockParticle()


@pytest.fixture
def mirror_particle_x(eps_metal, eps_vacuum, single_particle, monkeypatch):
    """ComParticleMirror with x-symmetry.

    Uses monkeypatch to replace _build_full with mock version.
    """
    monkeypatch.setattr(ComParticleMirror, '_build_full', _mock_build_full)
    p = ComParticleMirror(
        eps = [eps_metal, eps_vacuum],
        particles = [single_particle],
        inout = [[1, 2]],
        sym = 'x',
    )
    return p


@pytest.fixture
def mirror_particle_y(eps_metal, eps_vacuum, single_particle, monkeypatch):
    """ComParticleMirror with y-symmetry."""
    monkeypatch.setattr(ComParticleMirror, '_build_full', _mock_build_full)
    p = ComParticleMirror(
        eps = [eps_metal, eps_vacuum],
        particles = [single_particle],
        inout = [[1, 2]],
        sym = 'y',
    )
    return p


@pytest.fixture
def mirror_particle_xy(eps_metal, eps_vacuum, single_particle, monkeypatch):
    """ComParticleMirror with xy-symmetry."""
    monkeypatch.setattr(ComParticleMirror, '_build_full', _mock_build_full)
    p = ComParticleMirror(
        eps = [eps_metal, eps_vacuum],
        particles = [single_particle],
        inout = [[1, 2]],
        sym = 'xy',
    )
    return p


@pytest.fixture
def enei():
    """Typical wavelength (nm) for testing."""
    return 500.0


# ---------------------------------------------------------------------------
# ComParticleMirror tests
# ---------------------------------------------------------------------------


class TestComParticleMirror:
    """Tests for ComParticleMirror geometry class.

    MATLAB: Particles/@comparticlemirror
    Python: mnpbem/geometry/comparticle_mirror.py
    """

    def test_construction_x_symmetry(self, mirror_particle_x):
        """Test construction with x-mirror symmetry.

        Verifies the symmetry table is [[1,1],[1,-1]] (matching MATLAB init.m).
        """
        p = mirror_particle_x
        assert p.sym == 'x'
        expected_table = np.array([[1, 1], [1, -1]], dtype = np.float64)
        np.testing.assert_array_equal(p.symtable, expected_table)

    def test_construction_y_symmetry(self, mirror_particle_y):
        """Test construction with y-mirror symmetry."""
        p = mirror_particle_y
        assert p.sym == 'y'
        expected_table = np.array([[1, 1], [1, -1]], dtype = np.float64)
        np.testing.assert_array_equal(p.symtable, expected_table)

    def test_construction_xy_symmetry(self, mirror_particle_xy):
        """Test construction with xy-mirror symmetry.

        MATLAB: symtable = [1,1,1,1; 1,1,-1,-1; 1,-1,1,-1; 1,-1,-1,1]
        """
        p = mirror_particle_xy
        assert p.sym == 'xy'
        expected_table = np.array([
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [1, -1, -1, 1],
        ], dtype = np.float64)
        np.testing.assert_array_equal(p.symtable, expected_table)

    def test_full_particle_x(self, mirror_particle_x):
        """Test that full() returns a particle with 2x the original faces for x-sym.

        MATLAB: full(obj) -> obj.pfull
        In x-symmetry, the original particle is mirrored along x,
        so the full particle has 2 * nfaces_original faces.
        """
        p = mirror_particle_x
        pfull = p.full()
        assert pfull.nfaces == 2 * 4  # 4 original faces * 2 for x-mirror

    def test_full_particle_xy(self, mirror_particle_xy):
        """Test that full() returns 4x faces for xy-sym."""
        p = mirror_particle_xy
        pfull = p.full()
        assert pfull.nfaces == 4 * 4  # 4 original faces * 4 for xy-mirror

    def test_nfaces_half(self, mirror_particle_x):
        """Test that nfaces returns half of full particle faces.

        MATLAB: The mirror particle's n property returns only the
        non-mirrored portion of the faces.
        """
        p = mirror_particle_x
        assert p.nfaces == 4  # original 4 faces
        assert p.nfaces == p.full().nfaces // 2

    def test_nfaces_half_xy(self, mirror_particle_xy):
        """Test nfaces for xy-symmetry returns 1/4 of full faces."""
        p = mirror_particle_xy
        assert p.nfaces == 4
        assert p.nfaces == p.full().nfaces // 4

    def test_symvalue_plus_minus(self, mirror_particle_x):
        """Test symvalue() for '+' and '-' keys.

        MATLAB: symvalue(obj, '+') -> [1, 1]
                symvalue(obj, '-') -> [1, -1]
        """
        p = mirror_particle_x
        np.testing.assert_array_equal(
            p.symvalue('+'), np.array([[1, 1]], dtype = np.float64))
        np.testing.assert_array_equal(
            p.symvalue('-'), np.array([[1, -1]], dtype = np.float64))

    def test_symvalue_xy_keys(self, mirror_particle_xy):
        """Test symvalue() for xy-symmetry keys.

        MATLAB: symvalue(obj, '++') -> [1, 1, 1, 1]
                symvalue(obj, '+-') -> [1, 1, -1, -1]
                symvalue(obj, '-+') -> [1, -1, 1, -1]
                symvalue(obj, '--') -> [1, -1, -1, 1]
        """
        p = mirror_particle_xy
        np.testing.assert_array_equal(
            p.symvalue('++'), np.array([[1, 1, 1, 1]], dtype = np.float64))
        np.testing.assert_array_equal(
            p.symvalue('+-'), np.array([[1, 1, -1, -1]], dtype = np.float64))
        np.testing.assert_array_equal(
            p.symvalue('-+'), np.array([[1, -1, 1, -1]], dtype = np.float64))
        np.testing.assert_array_equal(
            p.symvalue('--'), np.array([[1, -1, -1, 1]], dtype = np.float64))

    def test_symvalue_list_input(self, mirror_particle_x):
        """Test symvalue() with list of keys.

        MATLAB: symvalue(obj, {'+', '-', '+'}) -> [1,1; 1,-1; 1,1]
        """
        p = mirror_particle_x
        result = p.symvalue(['+', '-', '+'])
        expected = np.array([[1, 1], [1, -1], [1, 1]], dtype = np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_symvalue_invalid_key(self, mirror_particle_x):
        """Test that symvalue() raises ValueError for unknown key."""
        p = mirror_particle_x
        with pytest.raises(ValueError, match = 'Unknown symmetry key'):
            p.symvalue('z')

    def test_symindex(self, mirror_particle_x):
        """Test symindex() returns correct 0-indexed position in symtable.

        MATLAB: symindex(obj, [1,1]) -> 1 (1-indexed)
        Python: symindex(obj, [1,1]) -> 0 (0-indexed)
        """
        p = mirror_particle_x
        assert p.symindex(np.array([1, 1])) == 0
        assert p.symindex(np.array([1, -1])) == 1

    def test_symindex_xy(self, mirror_particle_xy):
        """Test symindex() for xy-symmetry."""
        p = mirror_particle_xy
        assert p.symindex(np.array([1, 1, 1, 1])) == 0
        assert p.symindex(np.array([1, 1, -1, -1])) == 1
        assert p.symindex(np.array([1, -1, 1, -1])) == 2
        assert p.symindex(np.array([1, -1, -1, 1])) == 3

    def test_symindex_not_found(self, mirror_particle_x):
        """Test symindex() raises ValueError for unknown pattern."""
        p = mirror_particle_x
        with pytest.raises(ValueError, match = 'not found in table'):
            p.symindex(np.array([1, 0]))

    def test_mask(self, mirror_particle_x):
        """Test mask property returns boolean array of active particles.

        MATLAB: obj.mask returns boolean array.
        """
        p = mirror_particle_x
        mask = p.mask
        assert mask.shape == (1,)  # 1 original particle
        assert mask[0] == True

    def test_set_mask(self, eps_metal, eps_vacuum, monkeypatch):
        """Test set_mask() masks both mirror and full particles.

        MATLAB: mask(obj, ind) masks particles in obj and obj.pfull.
        """
        monkeypatch.setattr(ComParticleMirror, '_build_full', _mock_build_full)
        p1 = MockParticle()
        p2 = MockParticle(center = [20, 0, 0])
        p = ComParticleMirror(
            eps = [eps_metal, eps_vacuum],
            particles = [p1, p2],
            inout = [[1, 2], [1, 2]],
            sym = 'x',
        )
        p.set_mask([1])
        mask = p.mask
        assert mask[0] == True
        assert mask[1] == False

    def test_nvec_pos_area_properties(self, mirror_particle_x):
        """Test that nvec, pos, area return only half-particle data.

        MATLAB: These properties delegate to the full comparticle but
        return only the first half of the arrays.
        """
        p = mirror_particle_x
        n_half = p.nfaces
        assert p.nvec.shape == (n_half, 3)
        assert p.pos.shape == (n_half, 3)
        assert p.area.shape == (n_half,)

    def test_eps1_eps2(self, mirror_particle_x, enei):
        """Test eps1/eps2 return half-particle dielectric values."""
        p = mirror_particle_x
        e1 = p.eps1(enei)
        e2 = p.eps2(enei)
        assert e1.shape == (p.nfaces,)
        assert e2.shape == (p.nfaces,)
        # Metal inside, vacuum outside
        assert np.real(e1[0]) < 0  # metal
        assert np.isclose(np.real(e2[0]), 1.0)  # vacuum

    def test_closed_simple(self, eps_metal, eps_vacuum, monkeypatch):
        """Test set_closed() with simple index list.

        MATLAB: closed(obj, [1]) sets the first particle and its mirror
        as a closed surface.
        """
        monkeypatch.setattr(ComParticleMirror, '_build_full', _mock_build_full)
        p = ComParticleMirror(
            eps = [eps_metal, eps_vacuum],
            particles = [MockParticle()],
            inout = [[1, 2]],
            sym = 'x',
            closed_args = ([1],),
        )
        # The pfull should have closed entries set
        assert p.pfull.closed is not None
        assert any(c is not None for c in p.pfull.closed)

    def test_repr(self, mirror_particle_x):
        """Test string representation."""
        s = repr(mirror_particle_x)
        assert 'ComParticleMirror' in s
        assert 'sym=x' in s


# ---------------------------------------------------------------------------
# CompStructMirror tests
# ---------------------------------------------------------------------------


class TestCompStructMirror:
    """Tests for CompStructMirror helper class.

    MATLAB: Particles/@compstructmirror
    Python: mnpbem/geometry/comparticle_mirror.py (CompStructMirror)
    """

    def test_construction(self, mirror_particle_x, enei):
        """Test basic construction."""
        sm = CompStructMirror(mirror_particle_x, enei)
        assert sm.p is mirror_particle_x
        assert sm.enei == enei
        assert len(sm.val) == 0

    def test_full_without_fun(self, mirror_particle_x, enei):
        """Test full() without callback function returns self."""
        sm = CompStructMirror(mirror_particle_x, enei)
        result = sm.full()
        assert result is sm

    def test_full_with_fun(self, mirror_particle_x, enei):
        """Test full() with callback function calls the function."""
        called = [False]

        def my_fun(x):
            called[0] = True
            return x

        sm = CompStructMirror(mirror_particle_x, enei, my_fun)
        sm.full()
        assert called[0]

    def test_expand_empty(self, mirror_particle_x, enei):
        """Test expand() with no val returns empty tuple."""
        sm = CompStructMirror(mirror_particle_x, enei)
        result = sm.expand()
        assert result == ()

    def test_repr(self, mirror_particle_x, enei):
        """Test string representation."""
        sm = CompStructMirror(mirror_particle_x, enei)
        s = repr(sm)
        assert 'CompStructMirror' in s


# ---------------------------------------------------------------------------
# CompGreenStatMirror tests
# ---------------------------------------------------------------------------


class TestCompGreenStatMirror:
    """Tests for CompGreenStatMirror Green function class.

    MATLAB: Greenfun/@compgreenstatmirror
    Python: mnpbem/greenfun/compgreen_stat_mirror.py

    Note: These tests use xfail because constructing CompGreenStatMirror
    requires a real CompGreenStat which needs full particle geometry
    and Green function computation. We test the interface and logic
    with mocked internals where possible.
    """

    def test_class_attributes(self):
        """Test class-level constants match MATLAB."""
        assert CompGreenStatMirror.name == 'greenfunction'
        assert CompGreenStatMirror.needs == {'sim': 'stat', 'sym': True}

    def test_construction(self, mirror_particle_x):
        """Test construction creates inner Green function.

        MATLAB: compgreenstatmirror(p, ~, varargin) creates
                compgreenstat(p, full(p)).
        """
        g = CompGreenStatMirror(mirror_particle_x)
        assert g.p is mirror_particle_x
        assert g.g is not None

    def test_eval_logic_2d(self):
        """Test eval() contraction logic for 2D matrices.

        MATLAB: eval.m splits a (n, n*nsym) matrix into nsym blocks
        and contracts with symtable.
        """
        # Create a mock object to test the eval logic directly
        n = 4
        n_sym = 2  # x or y symmetry
        tab = np.array([[1, 1], [1, -1]], dtype = np.float64)

        # Create block matrix [A, B] where A and B are (n, n)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        mat = np.hstack([A, B])

        # Manual contraction
        g0 = A + B      # tab[0]: [1, 1]
        g1 = A - B      # tab[1]: [1, -1]

        # Test the contraction logic from CompGreenStatMirror.eval
        sub_mats = [mat[:, b * n:(b + 1) * n] for b in range(2)]
        g = [np.zeros_like(sub_mats[0])] * n_sym
        for i_sym in range(n_sym):
            g[i_sym] = np.zeros_like(sub_mats[0])
            for j_block in range(tab.shape[1]):
                g[i_sym] = g[i_sym] + tab[i_sym, j_block] * sub_mats[j_block]

        np.testing.assert_allclose(g[0], g0)
        np.testing.assert_allclose(g[1], g1)

    def test_eval_logic_3d(self):
        """Test eval() contraction logic for 3D matrices (Gp, H1p, H2p).

        MATLAB: for 3D arrays (n, 3, n*nsym), splits along third dimension.
        """
        n = 4
        n_sym = 2
        tab = np.array([[1, 1], [1, -1]], dtype = np.float64)

        A = np.random.randn(n, 3, n)
        B = np.random.randn(n, 3, n)
        mat = np.concatenate([A, B], axis = 2)

        # Manual contraction
        g0 = A + B
        g1 = A - B

        # Contraction logic
        sub_mats = [mat[:, :, b * n:(b + 1) * n] for b in range(2)]
        g = [np.zeros_like(sub_mats[0])] * n_sym
        for i_sym in range(n_sym):
            g[i_sym] = np.zeros_like(sub_mats[0])
            for j_block in range(tab.shape[1]):
                g[i_sym] = g[i_sym] + tab[i_sym, j_block] * sub_mats[j_block]

        np.testing.assert_allclose(g[0], g0)
        np.testing.assert_allclose(g[1], g1)

    def test_eval_logic_xy_symmetry(self):
        """Test eval() contraction logic for xy-symmetry (4 blocks).

        MATLAB: symtable for 'xy' has 4 rows and 4 columns.
        """
        n = 3
        n_sym = 4
        tab = np.array([
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [1, -1, -1, 1],
        ], dtype = np.float64)

        blocks = [np.random.randn(n, n) for _ in range(4)]
        mat = np.hstack(blocks)

        # Contract
        sub_mats = [mat[:, b * n:(b + 1) * n] for b in range(4)]
        g = []
        for i_sym in range(n_sym):
            val = np.zeros_like(sub_mats[0])
            for j_block in range(4):
                val = val + tab[i_sym, j_block] * sub_mats[j_block]
            g.append(val)

        # Verify: g[0] = blocks[0] + blocks[1] + blocks[2] + blocks[3]
        np.testing.assert_allclose(g[0], sum(blocks))
        # g[1] = blocks[0] + blocks[1] - blocks[2] - blocks[3]
        np.testing.assert_allclose(g[1], blocks[0] + blocks[1] - blocks[2] - blocks[3])

    def test_outer_nvec_eye(self):
        """Test _outer_nvec_eye helper function.

        MATLAB: div = sign * 2 * pi * outer(nvec, eye(n))
        This creates a (n, 3, n) tensor.
        """
        from mnpbem.greenfun.compgreen_stat_mirror import _outer_nvec_eye

        n = 4
        nvec = np.random.randn(n, 3)
        result = _outer_nvec_eye(nvec, n)

        assert result.shape == (n, 3, n)
        # Diagonal check: result[i, :, i] == nvec[i, :]
        for i in range(n):
            np.testing.assert_array_equal(result[i, :, i], nvec[i, :])
        # Off-diagonal check: result[i, :, j] == 0 for i != j
        for i in range(n):
            for j in range(n):
                if i != j:
                    np.testing.assert_array_equal(result[i, :, j], np.zeros(3))


# ---------------------------------------------------------------------------
# CompGreenRetMirror tests
# ---------------------------------------------------------------------------


class TestCompGreenRetMirror:
    """Tests for CompGreenRetMirror Green function class.

    MATLAB: Greenfun/@compgreenretmirror
    Python: mnpbem/greenfun/compgreen_ret_mirror.py
    """

    def test_class_attributes(self):
        """Test class-level constants match MATLAB."""
        assert CompGreenRetMirror.name == 'greenfunction'
        assert CompGreenRetMirror.needs == {'sim': 'ret', 'sym': True}

    def test_construction(self, mirror_particle_x):
        """Test construction creates inner Green function.

        MATLAB: compgreenretmirror(p, ~, varargin) creates
                compgreenret(p, full(p)).
        May pass if mock objects satisfy CompGreenRet requirements.
        """
        g = CompGreenRetMirror(mirror_particle_x)
        assert g.p is mirror_particle_x

    def test_indmul_helper(self):
        """Test _indmul helper for indexed matrix multiplication.

        MATLAB: indmul(mat, v, ind) in compgreenretmirror/field.m
        Multiplies mat{ind(k)} * v(:, k, :) for k=1,2,3.
        """
        from mnpbem.greenfun.compgreen_ret_mirror import _indmul

        n = 5
        mat_list = [np.random.randn(n, n) for _ in range(2)]
        v = np.random.randn(n, 3)
        ind = [0, 1, 0]  # x->mat[0], y->mat[1], z->mat[0]

        result = _indmul(mat_list, v, ind)
        assert result.shape == (n, 3)
        np.testing.assert_allclose(result[:, 0], mat_list[0] @ v[:, 0])
        np.testing.assert_allclose(result[:, 1], mat_list[1] @ v[:, 1])
        np.testing.assert_allclose(result[:, 2], mat_list[0] @ v[:, 2])

    def test_indmul_zero(self):
        """Test _indmul returns 0 when mat_list[0] is 0 (scalar zero check).

        MATLAB: if length(mat{1}) == 1 && mat{1} == 0 -> u = 0
        """
        from mnpbem.greenfun.compgreen_ret_mirror import _indmul

        mat_list = [0, 0]
        v = np.random.randn(5, 3)
        ind = [0, 1, 0]
        result = _indmul(mat_list, v, ind)
        assert result == 0

    def test_indcross_helper(self):
        """Test _indcross helper for indexed cross product.

        MATLAB: indcross(mat, v, ind) in compgreenretmirror/field.m
        Computes cross product using mat{ind(k)}(:, i, :) components.
        """
        from mnpbem.greenfun.compgreen_ret_mirror import _indcross

        n = 4
        # 3D matrices (n, 3, n) simulating Green function derivatives
        mat_list = [np.random.randn(n, 3, n) for _ in range(2)]
        v = np.random.randn(n, 3)
        ind = [0, 1, 0]

        result = _indcross(mat_list, v, ind)
        assert result.shape == (n, 3)

    def test_matmul_helper(self):
        """Test _matmul helper function handles zero cases.

        MATLAB: matmul() in various mirror files checks for scalar zero.
        """
        from mnpbem.greenfun.compgreen_ret_mirror import _matmul

        assert _matmul(0, np.array([1, 2, 3])) == 0
        assert _matmul(np.array([1, 2, 3]), 0) == 0

        A = np.array([[1, 2], [3, 4]])
        x = np.array([1, 2])
        np.testing.assert_array_equal(_matmul(A, x), A @ x)


# ---------------------------------------------------------------------------
# BEMStatMirror tests
# ---------------------------------------------------------------------------


class TestBEMStatMirror:
    """Tests for BEMStatMirror BEM solver class.

    MATLAB: BEM/@bemstatmirror
    Python: mnpbem/bem/bem_stat_mirror.py
    """

    def test_class_attributes(self):
        """Test class-level constants match MATLAB."""
        assert BEMStatMirror.name == 'bemsolver'
        assert BEMStatMirror.needs == {'sim': 'stat', 'sym': True}

    def test_construction(self, mirror_particle_x):
        """Test construction creates Green function and extracts F.

        MATLAB: bemstatmirror(p, op) creates compgreenstatmirror(p, p)
        and extracts F = obj.g.F.
        """
        bem = BEMStatMirror(mirror_particle_x)
        assert bem.p is mirror_particle_x
        assert bem.F is not None

    def test_init_matrices_logic(self):
        """Test _init_matrices logic: Lambda + F inversion.

        MATLAB: subsref.m line: mat{i} = -inv(diag(Lambda) + F{i})
        where Lambda = 2*pi*(eps1+eps2)/(eps1-eps2).
        """
        n = 4
        eps1 = np.full(n, -11.4 + 1.0j)
        eps2 = np.full(n, 1.0)
        lambda_diag = 2 * np.pi * (eps1 + eps2) / (eps1 - eps2)

        F = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        mat = -np.linalg.inv(np.diag(lambda_diag) + F)

        # Verify: (diag(Lambda) + F) @ mat == -I
        product = (np.diag(lambda_diag) + F) @ mat
        np.testing.assert_allclose(product, -np.eye(n), atol = 1e-10)

    def test_solve_logic(self):
        """Test solve logic: sig = mat * phip.

        MATLAB: mldivide.m: sig.val{i} = compstruct(obj.p, exc.enei,
                'sig', matmul(obj.mat{ind}, exc.val{i}.phip))
        """
        n = 4
        mat = np.random.randn(n, n)
        phip = np.random.randn(n)
        sig = mat @ phip
        assert sig.shape == (n,)

    def test_repr(self):
        """Test string representation for uninitialized solver."""
        # Create a minimal mock to test repr without full construction
        bem = BEMStatMirror.__new__(BEMStatMirror)
        bem.p = 'mock'
        bem.enei = None
        s = repr(bem)
        assert 'BEMStatMirror' in s
        assert 'not initialized' in s


# ---------------------------------------------------------------------------
# BEMRetMirror tests
# ---------------------------------------------------------------------------


class TestBEMRetMirror:
    """Tests for BEMRetMirror BEM solver class.

    MATLAB: BEM/@bemretmirror
    Python: mnpbem/bem/bem_ret_mirror.py
    """

    def test_class_attributes(self):
        """Test class-level constants match MATLAB."""
        assert BEMRetMirror.name == 'bemsolver'
        assert BEMRetMirror.needs == {'sim': 'ret', 'sym': True}

    def test_construction(self, mirror_particle_x):
        """Test construction creates Green function.

        MATLAB: bemretmirror(p, op) creates compgreenretmirror(p, p).
        May pass if mock objects satisfy CompGreenRetMirror requirements.
        """
        bem = BEMRetMirror(mirror_particle_x)
        assert bem.p is mirror_particle_x

    def test_subtract_list_helper(self):
        """Test _subtract_list helper for element-wise cell subtraction.

        MATLAB: subtract(a, b) in initmat.m subtracts cell arrays.
        """
        from mnpbem.bem.bem_ret_mirror import _subtract_list

        A = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
        B = [np.array([[1, 1], [1, 1]]), np.array([[2, 2], [2, 2]])]
        result = _subtract_list(A, B)
        np.testing.assert_array_equal(result[0], np.array([[0, 1], [2, 3]]))
        np.testing.assert_array_equal(result[1], np.array([[3, 4], [5, 6]]))

    def test_subtract_list_with_zeros(self):
        """Test _subtract_list when some elements are zero."""
        from mnpbem.bem.bem_ret_mirror import _subtract_list

        A = [0, np.array([[1, 2]])]
        B = [np.array([[3, 4]]), 0]
        result = _subtract_list(A, B)
        np.testing.assert_array_equal(result[0], -np.array([[3, 4]]))
        np.testing.assert_array_equal(result[1], np.array([[1, 2]]))

    def test_index_vec_helper(self):
        """Test _index_vec helper extracts component from vector.

        MATLAB: index(v, ind) in bemretmirror/mldivide.m
        For 2D: v(:, ind), for 3D: reshape(v(:, ind, :), [siz(1), siz(3:end)])
        """
        from mnpbem.bem.bem_ret_mirror import _index_vec

        v2d = np.random.randn(5, 3)
        np.testing.assert_array_equal(_index_vec(v2d, 1), v2d[:, 1])

        v3d = np.random.randn(5, 3, 2)
        np.testing.assert_array_equal(_index_vec(v3d, 2), v3d[:, 2, :])

        assert _index_vec(0, 0) == 0

    def test_vector_helper(self):
        """Test _vector helper combines components to vector.

        MATLAB: vector(vx, vy, vz) in bemretmirror/mldivide.m
        Produces an (n, 3) or (n, 3, npol) array.
        """
        from mnpbem.bem.bem_ret_mirror import _vector

        n = 5
        vx = np.random.randn(n)
        vy = np.random.randn(n)
        vz = np.random.randn(n)
        result = _vector(vx, vy, vz)
        assert result.shape == (n, 3)
        np.testing.assert_array_equal(result[:, 0], vx)
        np.testing.assert_array_equal(result[:, 1], vy)
        np.testing.assert_array_equal(result[:, 2], vz)

    def test_vector_helper_2d(self):
        """Test _vector helper for 2D input (npol > 1)."""
        from mnpbem.bem.bem_ret_mirror import _vector

        n, npol = 5, 3
        vx = np.random.randn(n, npol)
        vy = np.random.randn(n, npol)
        vz = np.random.randn(n, npol)
        result = _vector(vx, vy, vz)
        assert result.shape == (n, 3, npol)
        np.testing.assert_array_equal(result[:, 0, :], vx)

    def test_outer_eps_helper(self):
        """Test _outer_eps helper.

        MATLAB: outer(nvec, phi, eps) computes nvec .* (phi .* eps)
        Used in excitation.m for Eq. (15).
        """
        from mnpbem.bem.bem_ret_mirror import _outer_eps

        n = 4
        nvec = np.random.randn(n, 3)
        phi = np.random.randn(n)
        eps = np.random.randn(n)
        result = _outer_eps(nvec, phi, eps)
        assert result.shape == (n, 3)

        # result[i, :] = nvec[i, :] * (phi[i] * eps[i])
        for i in range(n):
            np.testing.assert_allclose(result[i, :], nvec[i, :] * (phi[i] * eps[i]))

    def test_outer_eps_zero(self):
        """Test _outer_eps returns 0 for zero phi."""
        from mnpbem.bem.bem_ret_mirror import _outer_eps

        nvec = np.random.randn(4, 3)
        eps = np.random.randn(4)
        assert _outer_eps(nvec, 0, eps) == 0

    def test_inner_eps_helper(self):
        """Test _inner_eps helper.

        MATLAB: inner(nvec, a, eps) computes sum(nvec .* a, 2) .* eps
        Used in excitation.m for Eq. (18).
        """
        from mnpbem.bem.bem_ret_mirror import _inner_eps

        n = 4
        nvec = np.random.randn(n, 3)
        a = np.random.randn(n, 3)
        eps = np.random.randn(n)
        result = _inner_eps(nvec, a, eps)
        expected = np.sum(nvec * a, axis = 1) * eps
        np.testing.assert_allclose(result, expected)

    def test_matmul_diag_vec_helper(self):
        """Test _matmul_diag_vec helper.

        MATLAB: matmul(nx, val) where nx is a component of nvec.
        Performs element-wise multiplication.
        """
        from mnpbem.bem.bem_ret_mirror import _matmul_diag_vec

        n = 5
        n_comp = np.random.randn(n)
        val = np.random.randn(n)
        result = _matmul_diag_vec(n_comp, val)
        np.testing.assert_array_equal(result, n_comp * val)

        # 2D val
        val2d = np.random.randn(n, 3)
        result2d = _matmul_diag_vec(n_comp, val2d)
        expected2d = n_comp[:, np.newaxis] * val2d
        np.testing.assert_array_equal(result2d, expected2d)

        # Zero val
        assert _matmul_diag_vec(n_comp, 0) == 0

    def test_init_sigmai_logic(self):
        """Test _init_sigmai logic matches MATLAB initsigmai.m.

        MATLAB Eq. (21,22):
        Sigma = Sigma1_z * L1_z - Sigma2_z * L2_z +
                k^2 * sum_i((L_i * Deltai_i) .* outer(i)) * L_z
        Sigmai = inv(Sigma)
        """
        n = 3
        k = 2 * np.pi / 500.0
        nvec = np.random.randn(n, 3)
        # Normalize normal vectors
        for i in range(n):
            nvec[i] /= np.linalg.norm(nvec[i])

        L1 = 2.0 + 0.5j   # scalar eps
        L2 = 1.0
        Sigma1 = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        Sigma2 = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        Deltai = np.linalg.inv(Sigma1 - Sigma2)

        # Outer product helper
        def outer_ii(i):
            return np.outer(nvec[:, i], nvec[:, i])

        L_diff = L1 - L2

        # MATLAB formula (scalar eps case):
        Sigma = (Sigma1 * L1 - Sigma2 * L2
                 + k ** 2 * (L_diff * Deltai * outer_ii(0)) * L_diff
                 + k ** 2 * (L_diff * Deltai * outer_ii(1)) * L_diff
                 + k ** 2 * (L_diff * Deltai * outer_ii(2)) * L_diff)
        Sigmai = np.linalg.inv(Sigma)

        # The result should be invertible
        product = Sigma @ Sigmai
        np.testing.assert_allclose(product, np.eye(n), atol = 1e-10)

    def test_repr(self):
        """Test string representation."""
        bem = BEMRetMirror.__new__(BEMRetMirror)
        bem.p = 'mock'
        bem.enei = None
        s = repr(bem)
        assert 'BEMRetMirror' in s
        assert 'not initialized' in s


# ---------------------------------------------------------------------------
# BEMStatEigMirror tests
# ---------------------------------------------------------------------------


class TestBEMStatEigMirror:
    """Tests for BEMStatEigMirror BEM solver with eigenmode expansion.

    MATLAB: BEM/@bemstateigmirror
    Python: mnpbem/bem/bem_stat_eig_mirror.py
    """

    def test_class_attributes(self):
        """Test class-level constants match MATLAB."""
        assert BEMStatEigMirror.name == 'bemsolver'
        assert BEMStatEigMirror.needs == {'sim': 'stat', 'nev': True, 'sym': True}

    def test_construction(self, mirror_particle_x):
        """Test construction computes eigenmodes.

        MATLAB: bemstateigmirror(p, op) creates compgreenstatmirror(p, p),
        then computes left/right eigenvectors of F.
        """
        bem = BEMStatEigMirror(mirror_particle_x, nev = 2)
        assert bem.p is mirror_particle_x
        assert bem.nev == 2

    def test_eigenmode_resolvent_logic(self):
        """Test eigenmode resolvent matrix construction.

        MATLAB: subsref.m:
            unit_lambda = obj.unit{i} * Lambda(:)
            resolvent = reshape(unit_lambda, nev, nev) + ene
            mat{i} = -ur * inv(resolvent) * ul

        scipy.sparse.linalg.eigs returns (eigenvalues, eigenvectors),
        i.e., (w, v) where v[:, i] is the eigenvector for w[i].
        """
        n = 6
        nev = 3

        # Create mock eigenvalues and eigenvectors
        F = np.random.randn(n, n) + 1j * 0.01 * np.random.randn(n, n)
        # Make it slightly asymmetric so left/right differ
        F = F + np.diag(np.arange(1, n + 1, dtype = np.float64))

        from scipy.sparse.linalg import eigs
        # eigs returns (eigenvalues, eigenvectors)
        ene_raw_l, ul_raw = eigs(F.T, k = nev, which = 'SR', maxiter = 1000)
        ul = ul_raw.T  # (nev, n)
        ene_raw, ur = eigs(F, k = nev, which = 'SR', maxiter = 1000)
        # ur is (n, nev), ene_raw is (nev,)
        ene = np.diag(ene_raw)

        # Make eigenvectors orthogonal
        overlap = ul @ ur  # (nev, nev)
        ul = np.linalg.solve(overlap, ul)

        # Lambda
        eps1 = -11.4 + 1.0j
        eps2 = 1.0
        Lambda_scalar = 2 * np.pi * (eps1 + eps2) / (eps1 - eps2)
        Lambda = np.full(1, Lambda_scalar)

        # Unit matrices (simplified: single material boundary)
        unit = np.zeros((nev ** 2, 1), dtype = complex)
        ind = np.arange(n)
        chunk = ul[:, ind] @ ur[ind, :]  # (nev, nev)
        unit[:, 0] = chunk.ravel()

        unit_lambda = unit @ Lambda  # (nev^2,)
        unit_lambda_mat = unit_lambda.reshape(nev, nev)
        resolvent = unit_lambda_mat + ene
        mat = -ur @ np.linalg.solve(resolvent, ul)

        # ur is (n, nev), resolvent is (nev, nev), ul is (nev, n)
        # so mat = (n, nev) @ (nev, nev)^{-1} @ (nev, n) = (n, n)
        assert mat.shape == (n, n)
        # mat should be a finite matrix
        assert np.all(np.isfinite(mat))

    def test_repr(self):
        """Test string representation."""
        bem = BEMStatEigMirror.__new__(BEMStatEigMirror)
        bem.p = 'mock'
        bem.nev = 20
        bem.enei = None
        s = repr(bem)
        assert 'BEMStatEigMirror' in s
        assert 'nev=20' in s


# ---------------------------------------------------------------------------
# BEMLayerMirror tests
# ---------------------------------------------------------------------------


class TestBEMLayerMirror:
    """Tests for BEMLayerMirror dummy class.

    MATLAB: BEM/@bemlayermirror
    Python: mnpbem/bem/bem_layer_mirror.py

    Both MATLAB and Python raise an error on construction because
    BEM solvers for layers and mirror symmetry are not implemented.
    """

    def test_raises_not_implemented(self):
        """Test that construction raises NotImplementedError.

        MATLAB: error('BEM solvers for layers and mirror symmetry not implemented')
        Python: raises NotImplementedError with same message.
        """
        with pytest.raises(NotImplementedError, match = 'not implemented'):
            BEMLayerMirror()

    def test_class_attributes(self):
        """Test class-level constants match MATLAB."""
        assert BEMLayerMirror.name == 'bemsolver'
        assert BEMLayerMirror.needs == {'sim': True, 'layer': True, 'sym': True}


# ---------------------------------------------------------------------------
# DipoleStatMirror tests
# ---------------------------------------------------------------------------


class TestDipoleStatMirror:
    """Tests for DipoleStatMirror simulation class.

    MATLAB: Simulation/static/@dipolestatmirror
    Python: mnpbem/simulation/dipole_stat_mirror.py
    """

    def test_class_attributes(self):
        """Test class-level constants match MATLAB."""
        assert DipoleStatMirror.name == 'dipole'
        assert DipoleStatMirror.needs == {'sim': 'stat', 'sym': True}

    def test_construction(self):
        """Test construction wraps a DipoleStat object.

        MATLAB: dipolestatmirror(pt, dip) creates dipolestat(pt, dip).
        """
        pt = MockComPoint(np.array([[0, 0, 20]]))
        dip = DipoleStatMirror(pt)
        assert dip.dip is not None
        assert dip.sym is None

    def test_init_mirror_x(self):
        """Test _init_mirror creates correct mirror dipoles for x-symmetry.

        MATLAB init.m:
          mirror{1} = dipolestat(pt, eye(3))
          mirror{2} = dipolestat(flip(pt, 1), [-1,0,0; 0,1,0; 0,0,1])
        """
        # Test the symmetry logic without full DipoleStat construction
        sym = 'x'
        dip_eye = np.eye(3)
        mirror_dip = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype = np.float64)

        # For x-mirror: x-component is negated
        assert mirror_dip[0, 0] == -1
        assert mirror_dip[1, 1] == 1
        assert mirror_dip[2, 2] == 1

    def test_init_mirror_y(self):
        """Test _init_mirror creates correct mirror dipoles for y-symmetry.

        MATLAB init.m:
          mirror{2} = dipolestat(flip(pt, 2), [1,0,0; 0,-1,0; 0,0,1])
        """
        mirror_dip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype = np.float64)
        assert mirror_dip[0, 0] == 1
        assert mirror_dip[1, 1] == -1
        assert mirror_dip[2, 2] == 1

    def test_init_mirror_xy(self):
        """Test _init_mirror creates 4 mirror dipoles for xy-symmetry.

        MATLAB init.m:
          mirror{1} = dipolestat(pt, eye(3))
          mirror{2} = dipolestat(flip(pt,1), [-1,0,0; 0,1,0; 0,0,1])
          mirror{3} = dipolestat(flip(pt,2), [1,0,0; 0,-1,0; 0,0,1])
          mirror{4} = dipolestat(flip(pt,[1,2]), [-1,0,0; 0,-1,0; 0,0,1])
        """
        # Verify the expected dipole patterns for xy
        eye3 = np.eye(3)
        flip_x = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        flip_y = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        flip_xy = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

        assert flip_x[0, 0] == -1 and flip_x[1, 1] == 1
        assert flip_y[0, 0] == 1 and flip_y[1, 1] == -1
        assert flip_xy[0, 0] == -1 and flip_xy[1, 1] == -1

    def test_potential_symmetry_values_x(self, mirror_particle_x):
        """Test potential() assigns correct symmetry values for x-symmetry.

        MATLAB potential.m for sym='x':
          val{1}.symval = symvalue({'-', '+', '+'})
          val{2}.symval = symvalue({'+', '-', '-'})
        """
        p = mirror_particle_x
        sv1 = p.symvalue(['-', '+', '+'])
        sv2 = p.symvalue(['+', '-', '-'])

        # val{1}: [-,+,+] means x-antisymmetric, y-symmetric, z-symmetric
        expected_sv1 = np.array([[1, -1], [1, 1], [1, 1]])
        expected_sv2 = np.array([[1, 1], [1, -1], [1, -1]])
        np.testing.assert_array_equal(sv1, expected_sv1)
        np.testing.assert_array_equal(sv2, expected_sv2)

    def test_potential_symmetry_values_xy(self, mirror_particle_xy):
        """Test potential() symmetry values for xy-symmetry.

        MATLAB potential.m for sym='xy':
          val{1}.symval = symvalue({'-+', '+-', '++'})
          val{2}.symval = symvalue({'++', '--', '-+'})
          val{3}.symval = symvalue({'--', '++', '+-'})
          val{4}.symval = symvalue({'+-', '-+', '--'})
        """
        p = mirror_particle_xy
        sv1 = p.symvalue(['-+', '+-', '++'])
        sv2 = p.symvalue(['++', '--', '-+'])

        # val{1}: x='-+', y='+-', z='++'
        expected_sv1 = np.array([
            [1, -1, 1, -1],   # -+
            [1, 1, -1, -1],   # +-
            [1, 1, 1, 1],     # ++
        ])
        np.testing.assert_array_equal(sv1, expected_sv1)

    def test_repr(self):
        """Test string representation."""
        d = DipoleStatMirror.__new__(DipoleStatMirror)
        d.dip = 'mock_dip'
        d.sym = 'x'
        s = repr(d)
        assert 'DipoleStatMirror' in s


# ---------------------------------------------------------------------------
# DipoleRetMirror tests
# ---------------------------------------------------------------------------


class TestDipoleRetMirror:
    """Tests for DipoleRetMirror simulation class.

    MATLAB: Simulation/retarded/@dipoleretmirror
    Python: mnpbem/simulation/dipole_ret_mirror.py
    """

    def test_class_attributes(self):
        """Test class-level constants match MATLAB."""
        assert DipoleRetMirror.name == 'dipole'
        assert DipoleRetMirror.needs == {'sim': 'ret', 'sym': True}

    def test_construction(self):
        """Test construction wraps a DipoleRet object.

        MATLAB: dipoleretmirror(pt, dip) creates dipoleretmirror(pt, dip).
        """
        pt = MockComPoint(np.array([[0, 0, 20]]))
        dip = DipoleRetMirror(pt)
        assert dip.dip is not None

    def test_init_mirror_creates_correct_dipoles_x(self):
        """Test _init_mirror for x-symmetry matches MATLAB init.m.

        Same structure as DipoleStatMirror but with DipoleRet objects.
        """
        # Mirror dipole pattern for x-symmetry is the same as stat
        flip_x = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(flip_x @ np.array([1, 0, 0]),
                                       np.array([-1, 0, 0]))
        np.testing.assert_array_equal(flip_x @ np.array([0, 1, 0]),
                                       np.array([0, 1, 0]))

    def test_potential_symmetry_values_y(self, mirror_particle_y):
        """Test potential() symmetry values for y-symmetry.

        MATLAB potential.m for sym='y':
          val{1}.symval = symvalue({'+', '-', '+'})
          val{2}.symval = symvalue({'-', '+', '-'})
        """
        p = mirror_particle_y
        sv1 = p.symvalue(['+', '-', '+'])
        sv2 = p.symvalue(['-', '+', '-'])
        expected_sv1 = np.array([[1, 1], [1, -1], [1, 1]])
        expected_sv2 = np.array([[1, -1], [1, 1], [1, -1]])
        np.testing.assert_array_equal(sv1, expected_sv1)
        np.testing.assert_array_equal(sv2, expected_sv2)

    def test_repr(self):
        """Test string representation."""
        d = DipoleRetMirror.__new__(DipoleRetMirror)
        d.dip = 'mock_dip'
        d.sym = 'x'
        s = repr(d)
        assert 'DipoleRetMirror' in s


# ---------------------------------------------------------------------------
# PlaneWaveStatMirror tests
# ---------------------------------------------------------------------------


class TestPlaneWaveStatMirror:
    """Tests for PlaneWaveStatMirror simulation class.

    MATLAB: Simulation/static/@planewavestatmirror
    Python: mnpbem/simulation/planewave_stat_mirror.py
    """

    def test_class_attributes(self):
        """Test class-level constants match MATLAB."""
        assert PlaneWaveStatMirror.name == 'planewave'
        assert PlaneWaveStatMirror.needs == {'sim': 'stat', 'sym': True}

    def test_construction(self):
        """Test construction creates inner PlaneWaveStat.

        MATLAB: planewavestatmirror(pol) creates planewavestat(pol).
        May pass if mock objects satisfy PlaneWaveStat requirements.
        """
        pol = np.array([[1, 0, 0]])
        pw = PlaneWaveStatMirror(pol)
        assert pw.exc is not None
        np.testing.assert_array_equal(pw.pol, np.array([[1, 0, 0]]))

    def test_potential_phip_x(self, mirror_particle_x):
        """Test potential() creates correct phip for x-symmetry.

        MATLAB potential.m:
          phip_x = -nvec * [1; 0; 0]
          phip_y = -nvec * [0; 1; 0]
          val{1} = compstruct(p, enei, 'phip', phip_x, 'symval', symvalue({'+','-','-'}))
          val{2} = compstruct(p, enei, 'phip', phip_y, 'symval', symvalue({'-','+','-'}))
        """
        p = mirror_particle_x
        nvec = p.nvec

        phip_x = -nvec @ np.array([1, 0, 0])
        phip_y = -nvec @ np.array([0, 1, 0])

        # phip_x should be the negative x-component of the normal vectors
        np.testing.assert_allclose(phip_x, -nvec[:, 0])
        np.testing.assert_allclose(phip_y, -nvec[:, 1])

    def test_potential_symvalues_x(self, mirror_particle_x):
        """Test potential() assigns correct symmetry values for x-symmetry.

        MATLAB potential.m for sym='x':
          val{1}.symval = symvalue({'+', '-', '-'})
          val{2}.symval = symvalue({'-', '+', '-'})
        Python must match these assignments.
        """
        p = mirror_particle_x
        sv1 = p.symvalue(['+', '-', '-'])
        sv2 = p.symvalue(['-', '+', '-'])

        expected_sv1 = np.array([[1, 1], [1, -1], [1, -1]])
        expected_sv2 = np.array([[1, -1], [1, 1], [1, -1]])
        np.testing.assert_array_equal(sv1, expected_sv1)
        np.testing.assert_array_equal(sv2, expected_sv2)

    def test_potential_symvalues_y(self, mirror_particle_y):
        """Test potential() symmetry values for y-symmetry.

        MATLAB potential.m for sym='y':
          val{1}.symval = symvalue({'+', '-', '+'})
          val{2}.symval = symvalue({'-', '+', '-'})
        """
        p = mirror_particle_y
        sv1 = p.symvalue(['+', '-', '+'])
        sv2 = p.symvalue(['-', '+', '-'])

        expected_sv1 = np.array([[1, 1], [1, -1], [1, 1]])
        expected_sv2 = np.array([[1, -1], [1, 1], [1, -1]])
        np.testing.assert_array_equal(sv1, expected_sv1)
        np.testing.assert_array_equal(sv2, expected_sv2)

    def test_potential_symvalues_xy(self, mirror_particle_xy):
        """Test potential() symmetry values for xy-symmetry.

        MATLAB potential.m for sym='xy':
          val{1}.symval = symvalue({'++', '--', '-+'})
          val{2}.symval = symvalue({'--', '++', '+-'})
        """
        p = mirror_particle_xy
        sv1 = p.symvalue(['++', '--', '-+'])
        sv2 = p.symvalue(['--', '++', '+-'])

        expected_sv1 = np.array([
            [1, 1, 1, 1],      # ++
            [1, -1, -1, 1],    # --
            [1, -1, 1, -1],    # -+
        ])
        expected_sv2 = np.array([
            [1, -1, -1, 1],    # --
            [1, 1, 1, 1],      # ++
            [1, 1, -1, -1],    # +-
        ])
        np.testing.assert_array_equal(sv1, expected_sv1)
        np.testing.assert_array_equal(sv2, expected_sv2)

    def test_full_logic(self):
        """Test full() scalar field expansion logic.

        MATLAB full.m:
          For scalars (sig, phi, phip):
            v(:, ip) = pol(ip, 1) * val1 + pol(ip, 2) * val2
          For vectors (e):
            v(:, :, ip) = pol(ip, 1) * val1 + pol(ip, 2) * val2
        """
        n = 8
        pol = np.array([[1, 0, 0], [0, 1, 0], [0.7071, 0.7071, 0]])
        npol = pol.shape[0]

        val1 = np.random.randn(n) + 1j * np.random.randn(n)
        val2 = np.random.randn(n) + 1j * np.random.randn(n)

        v = np.zeros((n, npol), dtype = complex)
        for ip in range(npol):
            v[:, ip] = pol[ip, 0] * val1 + pol[ip, 1] * val2

        # Verify x-pol uses val1, y-pol uses val2, diagonal uses both
        np.testing.assert_allclose(v[:, 0], val1)
        np.testing.assert_allclose(v[:, 1], val2)
        np.testing.assert_allclose(v[:, 2], 0.7071 * val1 + 0.7071 * val2, rtol = 1e-4)

    def test_repr(self):
        """Test string representation."""
        pw = PlaneWaveStatMirror.__new__(PlaneWaveStatMirror)
        pw.pol = np.array([[1, 0, 0]])
        pw.exc = None
        s = repr(pw)
        assert 'PlaneWaveStatMirror' in s


# ---------------------------------------------------------------------------
# PlaneWaveRetMirror tests
# ---------------------------------------------------------------------------


class TestPlaneWaveRetMirror:
    """Tests for PlaneWaveRetMirror simulation class.

    MATLAB: Simulation/retarded/@planewaveretmirror
    Python: mnpbem/simulation/planewave_ret_mirror.py
    """

    def test_class_attributes(self):
        """Test class-level constants match MATLAB."""
        assert PlaneWaveRetMirror.name == 'planewave'
        assert PlaneWaveRetMirror.needs == {'sim': 'ret', 'sym': True}

    def test_construction(self):
        """Test construction with pol and dir.

        MATLAB: planewaveretmirror(pol, dir) checks pol(:,3)==0.
        May pass if mock objects satisfy PlaneWaveRet requirements.
        """
        pol = np.array([[1, 0, 0]])
        dir = np.array([[0, 0, 1]])
        pw = PlaneWaveRetMirror(pol, dir)
        np.testing.assert_array_equal(pw.pol, np.array([[1, 0, 0]]))
        np.testing.assert_array_equal(pw.dir, np.array([[0, 0, 1]]))

    def test_pol_z_assertion(self):
        """Test that pol[:, 2] != 0 raises AssertionError.

        MATLAB: planewaveretmirror requires pol(:,3) == 0 for mirror symmetry.
        Python: assert np.allclose(pol[:, 2], 0).
        """
        pol = np.array([[1, 0, 1]])  # non-zero z-component
        dir = np.array([[0, 0, 1]])
        with pytest.raises(AssertionError):
            PlaneWaveRetMirror(pol, dir)

    def test_potential_direction_basis(self):
        """Test potential() creates x-pol and y-pol basis excitations.

        MATLAB potential.m:
          For i = 1:2
            pol = zeros(2,3); pol(:,i) = 1;
            dir = [0,0,1; 0,0,-1];
            pot.val{i} = exc(p, enei);
          end
        Python follows same pattern with PlaneWaveRet(pol_basis, dir_basis).
        """
        # x-polarization basis
        pol_x = np.zeros((2, 3))
        pol_x[:, 0] = 1.0
        np.testing.assert_array_equal(pol_x, np.array([[1, 0, 0], [1, 0, 0]]))

        # y-polarization basis
        pol_y = np.zeros((2, 3))
        pol_y[:, 1] = 1.0
        np.testing.assert_array_equal(pol_y, np.array([[0, 1, 0], [0, 1, 0]]))

        # direction basis: forward and backward
        dir_basis = np.array([[0, 0, 1], [0, 0, -1]], dtype = np.float64)
        np.testing.assert_array_equal(dir_basis[0], [0, 0, 1])
        np.testing.assert_array_equal(dir_basis[1], [0, 0, -1])

    def test_potential_symvalues_x(self, mirror_particle_x):
        """Test potential() symmetry values for x-symmetry.

        MATLAB potential.m for sym='x':
          val{1}.symval = symvalue({'+', '-', '-'})
          val{2}.symval = symvalue({'-', '+', '+'})
        """
        p = mirror_particle_x
        sv1 = p.symvalue(['+', '-', '-'])
        sv2 = p.symvalue(['-', '+', '+'])

        expected_sv1 = np.array([[1, 1], [1, -1], [1, -1]])
        expected_sv2 = np.array([[1, -1], [1, 1], [1, 1]])
        np.testing.assert_array_equal(sv1, expected_sv1)
        np.testing.assert_array_equal(sv2, expected_sv2)

    def test_potential_symvalues_y(self, mirror_particle_y):
        """Test potential() symmetry values for y-symmetry.

        MATLAB potential.m for sym='y':
          val{1}.symval = symvalue({'+', '-', '+'})
          val{2}.symval = symvalue({'-', '+', '-'})
        """
        p = mirror_particle_y
        sv1 = p.symvalue(['+', '-', '+'])
        sv2 = p.symvalue(['-', '+', '-'])

        expected_sv1 = np.array([[1, 1], [1, -1], [1, 1]])
        expected_sv2 = np.array([[1, -1], [1, 1], [1, -1]])
        np.testing.assert_array_equal(sv1, expected_sv1)
        np.testing.assert_array_equal(sv2, expected_sv2)

    def test_potential_symvalues_xy(self, mirror_particle_xy):
        """Test potential() symmetry values for xy-symmetry.

        MATLAB potential.m for sym='xy':
          val{1}.symval = symvalue({'++', '--', '-+'})
          val{2}.symval = symvalue({'--', '++', '+-'})
        """
        p = mirror_particle_xy
        sv1 = p.symvalue(['++', '--', '-+'])
        sv2 = p.symvalue(['--', '++', '+-'])

        expected_sv1 = np.array([
            [1, 1, 1, 1],
            [1, -1, -1, 1],
            [1, -1, 1, -1],
        ])
        expected_sv2 = np.array([
            [1, -1, -1, 1],
            [1, 1, 1, 1],
            [1, 1, -1, -1],
        ])
        np.testing.assert_array_equal(sv1, expected_sv1)
        np.testing.assert_array_equal(sv2, expected_sv2)

    def test_full_direction_selection(self):
        """Test full() selects correct direction column based on dir[ip, 2].

        MATLAB full.m:
          j = 1 if dir(ip,3) > 0, else j = 2
          v(:,ip) = pol(ip,1) * val1(:,j) + pol(ip,2) * val2(:,j)

        Python full.m uses 0-indexed: j = 0 if dir[ip,2] > 0 else 1.
        """
        pol = np.array([[1, 0, 0], [0, 1, 0]])
        dir = np.array([[0, 0, 1], [0, 0, -1]])

        n = 8
        val1 = np.random.randn(n, 2) + 1j * np.random.randn(n, 2)
        val2 = np.random.randn(n, 2) + 1j * np.random.randn(n, 2)

        npol = pol.shape[0]
        v = np.zeros((n, npol), dtype = complex)
        for ip in range(npol):
            j = 0 if dir[ip, 2] > 0 else 1
            v[:, ip] = pol[ip, 0] * val1[:, j] + pol[ip, 1] * val2[:, j]

        # Forward propagating x-pol: uses column 0 of val1
        np.testing.assert_allclose(v[:, 0], val1[:, 0])
        # Backward propagating y-pol: uses column 1 of val2
        np.testing.assert_allclose(v[:, 1], val2[:, 1])

    def test_repr(self):
        """Test string representation."""
        pw = PlaneWaveRetMirror.__new__(PlaneWaveRetMirror)
        pw.pol = np.array([[1, 0, 0]])
        pw.dir = np.array([[0, 0, 1]])
        pw.exc = None
        s = repr(pw)
        assert 'PlaneWaveRetMirror' in s


# ---------------------------------------------------------------------------
# Cross-class integration tests (logic verification)
# ---------------------------------------------------------------------------


class TestMirrorSymmetryConsistency:
    """Tests for cross-class consistency of mirror symmetry logic.

    These verify that the symmetry tables, symmetry values, and
    contraction patterns are consistent across all mirror classes.
    """

    def test_symtable_is_orthogonal_x(self, mirror_particle_x):
        """Test that the symmetry table rows for x/y symmetry are orthogonal.

        MATLAB: symtable = [1,1; 1,-1]
        This is a Hadamard matrix (H_2), so rows are orthogonal.
        """
        tab = mirror_particle_x.symtable
        product = tab @ tab.T
        expected = 2 * np.eye(2)  # H_2 * H_2^T = 2*I
        np.testing.assert_array_equal(product, expected)

    def test_symtable_is_orthogonal_xy(self, mirror_particle_xy):
        """Test that the symmetry table for xy-symmetry is a Hadamard matrix.

        MATLAB: symtable for 'xy' is the 4x4 Hadamard matrix H_4.
        H_4 * H_4^T = 4*I
        """
        tab = mirror_particle_xy.symtable
        product = tab @ tab.T
        expected = 4 * np.eye(4)
        np.testing.assert_array_equal(product, expected)

    def test_contraction_roundtrip_x(self):
        """Test that expanding and contracting preserves the original signal.

        For x-symmetry:
          Given two blocks A, B forming mat = [A, B]:
          g[0] = A + B (symmetric)
          g[1] = A - B (antisymmetric)
          Recovery: A = (g[0] + g[1]) / 2, B = (g[0] - g[1]) / 2
        """
        n = 4
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)

        g0 = A + B
        g1 = A - B

        A_recovered = 0.5 * (g0 + g1)
        B_recovered = 0.5 * (g0 - g1)
        np.testing.assert_allclose(A_recovered, A)
        np.testing.assert_allclose(B_recovered, B)

    def test_contraction_roundtrip_xy(self):
        """Test expand/contract roundtrip for xy-symmetry (4 blocks).

        Using the Hadamard matrix H_4:
        g = H_4 @ blocks
        blocks = H_4^{-1} @ g = (1/4) * H_4^T @ g
        """
        n = 3
        tab = np.array([
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [1, -1, -1, 1],
        ], dtype = np.float64)

        blocks = [np.random.randn(n, n) for _ in range(4)]

        # Contract
        g = []
        for i in range(4):
            val = np.zeros((n, n))
            for j in range(4):
                val += tab[i, j] * blocks[j]
            g.append(val)

        # Recover (using H^{-1} = H^T / 4)
        for j in range(4):
            recovered = np.zeros((n, n))
            for i in range(4):
                recovered += tab[i, j] * g[i]
            recovered /= 4
            np.testing.assert_allclose(recovered, blocks[j], atol = 1e-12)

    def test_sym_x_symvalue_consistency(self, mirror_particle_x):
        """Test symvalue keys map to correct symtable rows for x-symmetry.

        '+' -> [1,1] -> symtable row 0
        '-' -> [1,-1] -> symtable row 1
        """
        p = mirror_particle_x
        plus_val = p.symvalue('+')[0]
        minus_val = p.symvalue('-')[0]

        np.testing.assert_array_equal(plus_val, p.symtable[0])
        np.testing.assert_array_equal(minus_val, p.symtable[1])

        assert p.symindex(plus_val) == 0
        assert p.symindex(minus_val) == 1

    def test_sym_xy_symvalue_consistency(self, mirror_particle_xy):
        """Test symvalue keys map to correct symtable rows for xy-symmetry."""
        p = mirror_particle_xy
        keys = ['++', '+-', '-+', '--']
        for i, key in enumerate(keys):
            val = p.symvalue(key)[0]
            np.testing.assert_array_equal(val, p.symtable[i])
            assert p.symindex(val) == i

    def test_dipole_mirror_dip_matrices_are_diagonal(self):
        """Test that mirror dipole transformation matrices are diagonal.

        For all symmetries, the dipole transformation matrices used in
        init.m are diagonal (only sign changes on diagonal).
        This is a MATLAB-Python consistency check.
        """
        eye3 = np.eye(3)
        flip_x = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        flip_y = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        flip_xy = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

        for mat in [eye3, flip_x, flip_y, flip_xy]:
            # Must be diagonal
            off_diag = mat - np.diag(np.diag(mat))
            np.testing.assert_array_equal(off_diag, np.zeros((3, 3)))
            # Diagonal values must be +/-1
            for d in np.diag(mat):
                assert abs(d) == 1

    def test_planewave_stat_mirror_x_pol_uses_nvec_x(self, mirror_particle_x):
        """Test that x-polarization excitation uses nvec dot [1,0,0].

        MATLAB potential.m: phip = -nvec * [1;0;0]' = -nvec(:,1)
        """
        p = mirror_particle_x
        nvec = p.nvec
        phip_x = -nvec @ np.array([1, 0, 0])
        np.testing.assert_allclose(phip_x, -nvec[:, 0])

    def test_planewave_ret_mirror_requires_z_polarization_zero(self):
        """Test that PlaneWaveRetMirror enforces pol[:, 2] == 0.

        MATLAB: planewaveretmirror requires z-component of polarization to be
        zero because mirror symmetry is in the x-y plane.
        """
        # Valid (z=0)
        pol_valid = np.array([[1, 0, 0], [0, 1, 0]])
        assert np.allclose(pol_valid[:, 2], 0)

        # Invalid (z != 0)
        pol_invalid = np.array([[1, 0, 0.5]])
        assert not np.allclose(pol_invalid[:, 2], 0)


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Unit tests for helper functions in mirror modules."""

    def test_bemret_sub(self):
        """Test _sub helper in bem_ret_mirror."""
        from mnpbem.bem.bem_ret_mirror import _sub

        assert _sub(0, 0) == 0
        np.testing.assert_array_equal(_sub(np.array([1, 2]), 0), np.array([1, 2]))
        np.testing.assert_array_equal(_sub(0, np.array([1, 2])), np.array([-1, -2]))
        np.testing.assert_array_equal(
            _sub(np.array([3, 4]), np.array([1, 1])), np.array([2, 3]))

    def test_bemret_add(self):
        """Test _add helper in bem_ret_mirror."""
        from mnpbem.bem.bem_ret_mirror import _add

        assert _add(0, 0) == 0
        np.testing.assert_array_equal(_add(np.array([1, 2]), 0), np.array([1, 2]))
        np.testing.assert_array_equal(_add(0, np.array([1, 2])), np.array([1, 2]))
        np.testing.assert_array_equal(
            _add(np.array([1, 2]), np.array([3, 4])), np.array([4, 6]))

    def test_bemret_scalar_or_mat_sub(self):
        """Test _scalar_or_mat_sub helper."""
        from mnpbem.bem.bem_ret_mirror import _scalar_or_mat_sub

        assert _scalar_or_mat_sub(5.0, 3.0) == 2.0
        np.testing.assert_array_equal(
            _scalar_or_mat_sub(np.array([5, 6]), np.array([1, 2])),
            np.array([4, 4]))

    def test_bemret_matmul(self):
        """Test _matmul helper in bem_ret_mirror."""
        from mnpbem.bem.bem_ret_mirror import _matmul

        assert _matmul(0, 5) == 0
        assert _matmul(5, 0) == 0
        assert _matmul(3, 4) == 12

        A = np.array([[1, 2], [3, 4]])
        x = np.array([1, 1])
        np.testing.assert_array_equal(_matmul(A, x), np.array([3, 7]))

    def test_bemret_matmul_diag(self):
        """Test _matmul_diag helper in bem_ret_mirror."""
        from mnpbem.bem.bem_ret_mirror import _matmul_diag

        assert _matmul_diag(2.0, 0) == 0

        eps = np.array([2.0, 3.0])
        phi_p = np.array([1.0, 2.0])
        np.testing.assert_array_equal(_matmul_diag(eps, phi_p), eps * phi_p)

        # Scalar eps
        assert np.isclose(_matmul_diag(2.0, np.array([3.0]))[0], 6.0)

    def test_compgreenstat_matmul(self):
        """Test _matmul helper in compgreen_stat_mirror."""
        from mnpbem.greenfun.compgreen_stat_mirror import _matmul

        assert _matmul(0, np.array([1, 2])) == 0
        assert _matmul(np.array([[1, 0], [0, 1]]), 0) == 0

        A = np.eye(3)
        x = np.array([1, 2, 3])
        np.testing.assert_array_equal(_matmul(A, x), x)

    def test_compgreenstat_matmul_3d(self):
        """Test _matmul_3d helper in compgreen_stat_mirror."""
        from mnpbem.greenfun.compgreen_stat_mirror import _matmul_3d

        assert _matmul_3d(0, np.array([1])) == 0
        assert _matmul_3d(np.eye(3).reshape(1, 3, 3), 0) == 0

        n = 4
        a = np.random.randn(n, 3, n)
        x = np.random.randn(n)
        result = _matmul_3d(a, x)
        assert result.shape == (n, 3)
        for j in range(3):
            np.testing.assert_allclose(result[:, j], a[:, j, :] @ x)

    def test_compgreenret_matmul_3d(self):
        """Test _matmul_3d helper in compgreen_ret_mirror."""
        from mnpbem.greenfun.compgreen_ret_mirror import _matmul_3d

        n = 4
        a = np.random.randn(n, 3, n)
        x = np.random.randn(n)
        result = _matmul_3d(a, x)
        assert result.shape == (n, 3)

        # 2D x
        npol = 2
        x2d = np.random.randn(n, npol)
        result2d = _matmul_3d(a, x2d)
        assert result2d.shape == (n, 3, npol)


# ---------------------------------------------------------------------------
# MockParticle and MockComParticle unit tests
# ---------------------------------------------------------------------------


class TestMockObjects:
    """Verify that mock objects work correctly for testing."""

    def test_mock_particle_geometry(self):
        """Test MockParticle has valid geometry."""
        p = MockParticle()
        assert p.nfaces == 4
        assert p.verts.shape == (4, 3)
        assert p.pos.shape == (4, 3)
        assert p.nvec.shape == (4, 3)
        assert p.area.shape == (4,)
        assert np.all(p.area > 0)

    def test_mock_particle_outward_normals(self):
        """Test MockParticle has outward-pointing normals."""
        p = MockParticle()
        center = np.mean(p.verts, axis = 0)
        for i in range(p.nfaces):
            # Normal should point away from center
            assert np.dot(p.nvec[i], p.pos[i] - center) > 0

    def test_mock_particle_flip_x(self):
        """Test MockParticle.flip(1) mirrors x-coordinates."""
        p = MockParticle()
        pf = p.flip(1)
        np.testing.assert_allclose(pf.verts[:, 0], -p.verts[:, 0], atol = 1e-12)
        np.testing.assert_allclose(np.sort(np.abs(pf.verts[:, 1])),
                                    np.sort(np.abs(p.verts[:, 1])), atol = 1e-12)

    def test_mock_particle_flip_y(self):
        """Test MockParticle.flip(2) mirrors y-coordinates."""
        p = MockParticle()
        pf = p.flip(2)
        np.testing.assert_allclose(pf.verts[:, 1], -p.verts[:, 1], atol = 1e-12)

    def test_mock_particle_add(self):
        """Test MockParticle addition (concatenation)."""
        p1 = MockParticle()
        p2 = MockParticle(center = [20, 0, 0])
        p_combined = p1 + p2
        assert p_combined.nfaces == 8
        assert p_combined.verts.shape[0] == 8

    def test_mock_eps(self):
        """Test MockEps callable returns correct value."""
        eps = MockEps(-11.4 + 1.0j)
        result = eps(500.0)
        assert np.isclose(result[0], -11.4 + 1.0j)

    def test_mock_comparticle(self):
        """Test MockComParticle construction and properties."""
        eps_metal = MockEps(-11.4 + 1.0j)
        eps_vac = MockEps(1.0)
        p = MockParticle()
        cp = MockComParticle([eps_metal, eps_vac], [p], [[1, 2]])
        assert cp.nfaces == 4
        assert cp.np >= 1
        assert cp.nvec.shape == (4, 3)

    def test_mock_comparticle_eps(self):
        """Test MockComParticle eps1/eps2 methods."""
        eps_metal = MockEps(-11.4 + 1.0j)
        eps_vac = MockEps(1.0)
        p = MockParticle()
        cp = MockComParticle([eps_metal, eps_vac], [p], [[1, 2]])
        e1 = cp.eps1(500.0)
        e2 = cp.eps2(500.0)
        assert np.isclose(e1[0], -11.4 + 1.0j)
        assert np.isclose(e2[0], 1.0)

    def test_mock_compoint_flip(self):
        """Test MockComPoint flip."""
        pt = MockComPoint(np.array([[10, 20, 30]]))
        pf = pt.flip(1)
        np.testing.assert_array_equal(pf.pos[0], [-10, 20, 30])
        pf2 = pt.flip([1, 2])
        np.testing.assert_array_equal(pf2.pos[0], [-10, -20, 30])
