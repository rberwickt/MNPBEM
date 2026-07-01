import os
import sys
import numpy as np
import pytest
from typing import Optional, Tuple, Any, List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mnpbem.greenfun.clustertree import ClusterTree
from mnpbem.greenfun.hmatrix import HMatrix
from mnpbem.greenfun.aca_compgreen_ret import ACACompGreenRet


# ---------------------------------------------------------------------------
# Mock objects that mimic the MNPBEM particle/dielectric interfaces
# ---------------------------------------------------------------------------

class MockParticle(object):

    def __init__(self,
            pos: np.ndarray,
            nvec: Optional[np.ndarray] = None,
            area: Optional[np.ndarray] = None):

        self.pos = np.asarray(pos, dtype = np.float64)
        self.n = self.pos.shape[0]

        if nvec is None:
            # Unit normals pointing radially outward
            norms = np.linalg.norm(self.pos, axis = 1, keepdims = True)
            norms = np.maximum(norms, 1e-10)
            self.nvec = self.pos / norms
        else:
            self.nvec = np.asarray(nvec, dtype = np.float64)

        if area is None:
            # Uniform area per face
            self.area = np.ones(self.n, dtype = np.float64)
        else:
            self.area = np.asarray(area, dtype = np.float64)


class MockEps(object):

    # Mock dielectric function that returns (eps, k) for a given wavelength.
    # Models a simple Drude metal for testing.

    def __init__(self, eps_val: complex = 1.0):

        self._eps_val = eps_val

    def __call__(self, enei: float) -> Tuple[complex, complex]:

        # enei is wavelength in nm
        k = 2.0 * np.pi / enei
        return self._eps_val, k


class MockComParticle(object):

    # Mock composite particle that mimics MNPBEM ComParticle interface.

    def __init__(self,
            particles: List[MockParticle],
            eps_list: Optional[List[MockEps]] = None,
            inout: Optional[np.ndarray] = None):

        self.p = particles

        # Total number of faces
        self.n = sum(part.n for part in particles)

        # Composite positions
        total_pos = np.empty((self.n, 3), dtype = np.float64)
        total_nvec = np.empty((self.n, 3), dtype = np.float64)
        total_area = np.empty(self.n, dtype = np.float64)
        offset = 0
        for part in particles:
            total_pos[offset:offset + part.n] = part.pos
            total_nvec[offset:offset + part.n] = part.nvec
            total_area[offset:offset + part.n] = part.area
            offset += part.n

        self.pos = total_pos
        self.nvec = total_nvec
        self.area = total_area

        # Dielectric functions
        if eps_list is None:
            self.eps = [MockEps(1.0), MockEps(-10.0 + 1.0j)]
        else:
            self.eps = eps_list

        # In/out matrix: (nparticles, 2) telling which material is on each side
        if inout is None:
            self.inout = np.array([[2, 1]] * len(particles), dtype = np.int64)
        else:
            self.inout = np.asarray(inout, dtype = np.int64)

        # Closed surface attribute
        self.closed = [None] * len(particles)

        # Cumulative indices for indexing into subparticles
        self._cum_idx = [0]
        for part in particles:
            self._cum_idx.append(self._cum_idx[-1] + part.n)

    def index_func(self, i: int) -> np.ndarray:

        # i is 1-based (MATLAB convention)
        start = self._cum_idx[i - 1]
        end = self._cum_idx[i]
        return np.arange(start, end, dtype = np.int64)

    def eps1(self, enei: float) -> np.ndarray:

        # Return epsilon values for inside surface for all faces
        vals = np.empty(self.n, dtype = np.complex128)
        for idx, part in enumerate(self.p):
            eps_idx = self.inout[idx, 0] - 1  # 0-based
            eps_val, _ = self.eps[eps_idx](enei)
            start = self._cum_idx[idx]
            end = self._cum_idx[idx + 1]
            vals[start:end] = eps_val
        return vals

    def eps2(self, enei: float) -> np.ndarray:

        # Return epsilon values for outside surface for all faces
        vals = np.empty(self.n, dtype = np.complex128)
        for idx, part in enumerate(self.p):
            eps_idx = self.inout[idx, 1] - 1  # 0-based
            eps_val, _ = self.eps[eps_idx](enei)
            start = self._cum_idx[idx]
            end = self._cum_idx[idx + 1]
            vals[start:end] = eps_val
        return vals

    def closedparticle(self, i: int) -> Tuple[Any, float, Any]:

        return None, 1.0, None


# ---------------------------------------------------------------------------
# Mock CompGreenRet for controlled testing
# ---------------------------------------------------------------------------

class MockCompGreenRet(object):

    # A mock CompGreenRet that uses a simple kernel function
    # so we can verify ACA compression against known dense results.

    def __init__(self,
            p1: MockComParticle,
            p2: MockComParticle,
            **options: Any):

        self.p1 = p1
        self.p2 = p2
        self.con = self._build_con(p1, p2)

    def _build_con(self,
            p1: MockComParticle,
            p2: MockComParticle) -> List[List[np.ndarray]]:

        n1 = p1.inout.shape[1]
        n2 = p2.inout.shape[1]
        con = [[None for _ in range(n2)] for _ in range(n1)]

        for i in range(n1):
            for j in range(n2):
                io1 = p1.inout[:, i]
                io2 = p2.inout[:, j]
                npart1 = len(io1)
                npart2 = len(io2)
                c1 = np.tile(io1.reshape(-1, 1), (1, npart2))
                c2 = np.tile(io2.reshape(1, -1), (npart1, 1))
                con_mat = np.zeros((npart1, npart2), dtype = np.int64)
                mask = (c1 == c2)
                con_mat[mask] = c1[mask]
                con[i][j] = con_mat

        return con

    def eval(self,
            i: int,
            j: int,
            key: str,
            enei: float,
            ind: Any = None) -> np.ndarray:

        # Simple retarded Green function evaluation using 1/r kernel
        # with complex wavenumber dependence.
        pos = self.p1.pos
        nvec = self.p1.nvec
        k = 2.0 * np.pi / enei

        n = self.p1.n
        mat = np.zeros((n, n), dtype = np.complex128)

        # Get connectivity for this region pair
        con = self.con[i][j]

        for i1 in range(con.shape[0]):
            for i2 in range(con.shape[1]):
                if con[i1, i2] > 0:
                    eps_val, k_val = self.p1.eps[con[i1, i2] - 1](enei)

                    idx1 = self.p1.index_func(i1 + 1)
                    idx2 = self.p2.index_func(i2 + 1)

                    pos1 = pos[idx1]
                    pos2 = pos[idx2]
                    nvec2 = nvec[idx2]

                    # Compute pairwise distances
                    n1_local = len(idx1)
                    n2_local = len(idx2)

                    # diff[a, b] = pos1[a] - pos2[b]
                    diff = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]  # (n1, n2, 3)
                    dist = np.linalg.norm(diff, axis = 2)  # (n1, n2)
                    dist = np.maximum(dist, 1e-10)

                    match key:

                        case 'G':
                            # Retarded Green function: exp(ikr) / (4*pi*r)
                            g_block = np.exp(1j * k_val * dist) / (4.0 * np.pi * dist)

                        case 'F':
                            # Surface derivative of G
                            # dG/dn = (ik - 1/r) * exp(ikr) / (4*pi*r) * (r_hat . n)
                            r_hat = diff / dist[:, :, np.newaxis]
                            cos_angle = np.sum(r_hat * nvec2[np.newaxis, :, :], axis = 2)
                            g_block = (1j * k_val - 1.0 / dist) * np.exp(1j * k_val * dist) / (4.0 * np.pi * dist) * cos_angle

                        case 'H1':
                            # F + 2*pi * delta
                            r_hat = diff / dist[:, :, np.newaxis]
                            cos_angle = np.sum(r_hat * nvec2[np.newaxis, :, :], axis = 2)
                            g_block = (1j * k_val - 1.0 / dist) * np.exp(1j * k_val * dist) / (4.0 * np.pi * dist) * cos_angle
                            # Add 2*pi on diagonal
                            if n1_local == n2_local and np.array_equal(idx1, idx2):
                                g_block += 2.0 * np.pi * np.eye(n1_local)

                        case 'H2':
                            # F - 2*pi * delta
                            r_hat = diff / dist[:, :, np.newaxis]
                            cos_angle = np.sum(r_hat * nvec2[np.newaxis, :, :], axis = 2)
                            g_block = (1j * k_val - 1.0 / dist) * np.exp(1j * k_val * dist) / (4.0 * np.pi * dist) * cos_angle
                            # Subtract 2*pi on diagonal
                            if n1_local == n2_local and np.array_equal(idx1, idx2):
                                g_block -= 2.0 * np.pi * np.eye(n1_local)

                        case _:
                            raise ValueError('[error] Unknown key <{}>'.format(key))

                    # Multiply by area for integration weight
                    area2 = self.p1.area[idx2]
                    g_block = g_block * area2[np.newaxis, :]

                    mat[np.ix_(idx1, idx2)] = g_block

        if ind is not None:
            return mat.ravel()[ind]

        return mat


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def sphere_points(n: int, radius: float = 10.0) -> np.ndarray:

    rng = np.random.RandomState(42)
    phi = rng.uniform(0, 2 * np.pi, n)
    cos_theta = rng.uniform(-1, 1, n)
    theta = np.arccos(cos_theta)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    pos = np.empty((n, 3), dtype = np.float64)
    pos[:, 0] = x
    pos[:, 1] = y
    pos[:, 2] = z
    return pos


def make_single_particle(n: int = 100, radius: float = 10.0) -> MockComParticle:

    pos = sphere_points(n, radius)
    part = MockParticle(pos)
    eps_list = [MockEps(1.0), MockEps(-10.0 + 1.0j)]
    cp = MockComParticle([part], eps_list = eps_list)
    return cp


def make_two_particles(n1: int = 80,
        n2: int = 80,
        r1: float = 10.0,
        r2: float = 10.0,
        offset: float = 60.0) -> MockComParticle:

    pos1 = sphere_points(n1, r1)

    rng2 = np.random.RandomState(99)
    phi = rng2.uniform(0, 2 * np.pi, n2)
    cos_theta = rng2.uniform(-1, 1, n2)
    theta = np.arccos(cos_theta)
    pos2 = np.empty((n2, 3), dtype = np.float64)
    pos2[:, 0] = r2 * np.sin(theta) * np.cos(phi) + offset
    pos2[:, 1] = r2 * np.sin(theta) * np.sin(phi)
    pos2[:, 2] = r2 * np.cos(theta)

    part1 = MockParticle(pos1)
    part2 = MockParticle(pos2)

    eps_list = [MockEps(1.0), MockEps(-10.0 + 1.0j)]
    cp = MockComParticle([part1, part2], eps_list = eps_list)
    return cp


# ---------------------------------------------------------------------------
# Monkey-patch ACACompGreenRet to use MockCompGreenRet
# ---------------------------------------------------------------------------

class TestableACACompGreenRet(ACACompGreenRet):

    # Subclass that replaces the internal CompGreenRet with MockCompGreenRet.

    def __init__(self,
            p: Any,
            htol: float = 1e-6,
            kmax: int = 100,
            cleaf: int = 32,
            fadmiss: Any = None,
            eta: float = 2.5,
            **options: Any):

        self.p = p
        self.g = MockCompGreenRet(p, p, **options)
        pos = p.pos
        ipart_arr = self._build_ipart_arr(p)
        self.tree = ClusterTree(pos, cleaf = cleaf, ipart_arr = ipart_arr)
        self.eta = eta
        self._user_fadmiss = fadmiss
        self.hmat_template = HMatrix(
                tree = self.tree, htol = htol, kmax = kmax,
                fadmiss = fadmiss if fadmiss is not None else
                        (lambda r1, r2, d: eta * min(r1, r2) < d))
        self._cache = {}
        self.htol = htol
        self.kmax = kmax
        self.cleaf = cleaf
        self.options = options


# ---------------------------------------------------------------------------
# Tests: ACA Green function matches dense Green function
# ---------------------------------------------------------------------------

class TestACAGreenMatchesDense(object):

    def test_G_single_particle(self) -> None:

        cp = make_single_particle(n = 100)
        enei = 600.0

        aca_green = TestableACACompGreenRet(cp, htol = 1e-6, cleaf = 16)

        # Dense reference
        dense_mat = aca_green.g.eval(0, 0, 'G', enei)

        # ACA H-matrix -> full
        hmat = aca_green.eval(0, 0, 'G', enei)
        aca_full = hmat.full()

        rel_err = np.linalg.norm(aca_full - dense_mat) / np.linalg.norm(dense_mat)
        assert rel_err < 1e-3, 'ACA G relative error: {}'.format(rel_err)

    def test_F_single_particle(self) -> None:

        cp = make_single_particle(n = 100)
        enei = 600.0

        aca_green = TestableACACompGreenRet(cp, htol = 1e-6, cleaf = 16)

        dense_mat = aca_green.g.eval(0, 0, 'F', enei)
        hmat = aca_green.eval(0, 0, 'F', enei)
        aca_full = hmat.full()

        rel_err = np.linalg.norm(aca_full - dense_mat) / np.linalg.norm(dense_mat)
        assert rel_err < 1e-3, 'ACA F relative error: {}'.format(rel_err)

    def test_H1_single_particle(self) -> None:

        cp = make_single_particle(n = 100)
        enei = 600.0

        aca_green = TestableACACompGreenRet(cp, htol = 1e-6, cleaf = 16)

        dense_mat = aca_green.g.eval(0, 0, 'H1', enei)
        hmat = aca_green.eval(0, 0, 'H1', enei)
        aca_full = hmat.full()

        rel_err = np.linalg.norm(aca_full - dense_mat) / np.linalg.norm(dense_mat)
        assert rel_err < 1e-3, 'ACA H1 relative error: {}'.format(rel_err)

    def test_H2_single_particle(self) -> None:

        cp = make_single_particle(n = 100)
        enei = 600.0

        aca_green = TestableACACompGreenRet(cp, htol = 1e-6, cleaf = 16)

        dense_mat = aca_green.g.eval(0, 0, 'H2', enei)
        hmat = aca_green.eval(0, 0, 'H2', enei)
        aca_full = hmat.full()

        rel_err = np.linalg.norm(aca_full - dense_mat) / np.linalg.norm(dense_mat)
        assert rel_err < 1e-3, 'ACA H2 relative error: {}'.format(rel_err)

    def test_G_two_particles(self) -> None:

        cp = make_two_particles(n1 = 80, n2 = 80, offset = 60.0)
        enei = 600.0

        aca_green = TestableACACompGreenRet(cp, htol = 1e-6, cleaf = 16)

        # Test inside-inside region pair
        dense_mat = aca_green.g.eval(0, 0, 'G', enei)
        hmat = aca_green.eval(0, 0, 'G', enei)
        aca_full = hmat.full()

        if np.linalg.norm(dense_mat) > 1e-12:
            rel_err = np.linalg.norm(aca_full - dense_mat) / np.linalg.norm(dense_mat)
            assert rel_err < 1e-3, 'ACA G two-particle relative error: {}'.format(rel_err)

    def test_G_region_pairs(self) -> None:

        cp = make_single_particle(n = 80)
        enei = 600.0

        aca_green = TestableACACompGreenRet(cp, htol = 1e-6, cleaf = 16)

        # Test all four region pairs
        for i in range(2):
            for j in range(2):
                dense_mat = aca_green.g.eval(i, j, 'G', enei)
                if np.isscalar(dense_mat) or np.linalg.norm(dense_mat) < 1e-12:
                    continue

                hmat = aca_green.eval(i, j, 'G', enei)
                aca_full = hmat.full()

                rel_err = np.linalg.norm(aca_full - dense_mat) / np.linalg.norm(dense_mat)
                assert rel_err < 1e-3, (
                    'ACA G region ({}, {}) relative error: {}'.format(i, j, rel_err))

    def test_different_wavelengths(self) -> None:

        cp = make_single_particle(n = 80)
        aca_green = TestableACACompGreenRet(cp, htol = 1e-6, cleaf = 16)

        for enei in [400.0, 600.0, 800.0]:
            dense_mat = aca_green.g.eval(0, 0, 'G', enei)
            if np.isscalar(dense_mat) or np.linalg.norm(dense_mat) < 1e-12:
                continue

            aca_green.clear_cache()
            hmat = aca_green.eval(0, 0, 'G', enei)
            aca_full = hmat.full()

            rel_err = np.linalg.norm(aca_full - dense_mat) / np.linalg.norm(dense_mat)
            assert rel_err < 1e-3, (
                'ACA G at wavelength {} nm relative error: {}'.format(enei, rel_err))


# ---------------------------------------------------------------------------
# Tests: Potential computation matches non-ACA version
# ---------------------------------------------------------------------------

class MockCompStruct(object):

    # Minimal mock for CompStruct to pass to potential().

    def __init__(self,
            p: Any,
            enei: float,
            **kwargs: Any):

        self.p = p
        self.enei = enei
        self.val = {}
        for key, value in kwargs.items():
            self.val[key] = value

    def __getattr__(self, name: str) -> Any:

        if name in ('p', 'enei', 'val'):
            return object.__getattribute__(self, name)
        val = object.__getattribute__(self, 'val')
        if name in val:
            return val[name]
        raise AttributeError('[error] No attribute <{}>'.format(name))


class TestPotentialMatches(object):

    def test_potential_inside_scalar(self) -> None:

        cp = make_single_particle(n = 80)
        enei = 600.0

        aca_green = TestableACACompGreenRet(cp, htol = 1e-6, cleaf = 16)

        n = cp.n
        rng = np.random.RandomState(42)
        sig1 = rng.randn(n) + 1j * rng.randn(n)
        sig2 = rng.randn(n) + 1j * rng.randn(n)
        h1 = rng.randn(n, 3) + 1j * rng.randn(n, 3)
        h2 = rng.randn(n, 3) + 1j * rng.randn(n, 3)

        sig = MockCompStruct(cp, enei, sig1 = sig1, sig2 = sig2, h1 = h1, h2 = h2)

        # ACA potential
        pot_aca = aca_green.potential(sig, inout = 1)

        # Dense reference potential
        G1 = aca_green.g.eval(0, 0, 'G', enei)
        G2 = aca_green.g.eval(0, 1, 'G', enei)
        H1 = aca_green.g.eval(0, 0, 'H1', enei)
        H2 = aca_green.g.eval(0, 1, 'H1', enei)

        # Dense computation
        phi_dense = self._dense_matmul(G1, sig1) + self._dense_matmul(G2, sig2)
        phip_dense = self._dense_matmul(H1, sig1) + self._dense_matmul(H2, sig2)

        phi_aca = pot_aca.val['phi1']
        phip_aca = pot_aca.val['phi1p']

        if np.linalg.norm(phi_dense) > 1e-12:
            rel_err = np.linalg.norm(phi_aca - phi_dense) / np.linalg.norm(phi_dense)
            assert rel_err < 1e-2, 'Potential phi1 relative error: {}'.format(rel_err)

        if np.linalg.norm(phip_dense) > 1e-12:
            rel_err = np.linalg.norm(phip_aca - phip_dense) / np.linalg.norm(phip_dense)
            assert rel_err < 1e-2, 'Potential phi1p relative error: {}'.format(rel_err)

    def test_potential_outside(self) -> None:

        cp = make_single_particle(n = 80)
        enei = 600.0

        aca_green = TestableACACompGreenRet(cp, htol = 1e-6, cleaf = 16)

        n = cp.n
        rng = np.random.RandomState(77)
        sig1 = rng.randn(n) + 1j * rng.randn(n)
        sig2 = rng.randn(n) + 1j * rng.randn(n)
        h1 = rng.randn(n, 3) + 1j * rng.randn(n, 3)
        h2 = rng.randn(n, 3) + 1j * rng.randn(n, 3)

        sig = MockCompStruct(cp, enei, sig1 = sig1, sig2 = sig2, h1 = h1, h2 = h2)

        # ACA potential
        pot_aca = aca_green.potential(sig, inout = 2)

        # Dense reference
        G1 = aca_green.g.eval(1, 0, 'G', enei)
        G2 = aca_green.g.eval(1, 1, 'G', enei)
        H1 = aca_green.g.eval(1, 0, 'H2', enei)
        H2 = aca_green.g.eval(1, 1, 'H2', enei)

        phi_dense = self._dense_matmul(G1, sig1) + self._dense_matmul(G2, sig2)

        phi_aca = pot_aca.val['phi2']

        if np.linalg.norm(phi_dense) > 1e-12:
            rel_err = np.linalg.norm(phi_aca - phi_dense) / np.linalg.norm(phi_dense)
            assert rel_err < 1e-2, 'Potential phi2 relative error: {}'.format(rel_err)

    def test_potential_vector(self) -> None:

        cp = make_single_particle(n = 80)
        enei = 600.0

        aca_green = TestableACACompGreenRet(cp, htol = 1e-6, cleaf = 16)

        n = cp.n
        rng = np.random.RandomState(55)
        sig1 = rng.randn(n) + 1j * rng.randn(n)
        sig2 = rng.randn(n) + 1j * rng.randn(n)
        h1 = rng.randn(n, 3) + 1j * rng.randn(n, 3)
        h2 = rng.randn(n, 3) + 1j * rng.randn(n, 3)

        sig = MockCompStruct(cp, enei, sig1 = sig1, sig2 = sig2, h1 = h1, h2 = h2)

        # ACA potential
        pot_aca = aca_green.potential(sig, inout = 1)

        # Dense reference for vector potential
        G1 = aca_green.g.eval(0, 0, 'G', enei)
        G2 = aca_green.g.eval(0, 1, 'G', enei)

        a_dense = self._dense_matmul(G1, h1) + self._dense_matmul(G2, h2)

        a_aca = pot_aca.val['a1']

        if np.linalg.norm(a_dense) > 1e-12:
            rel_err = np.linalg.norm(a_aca - a_dense) / np.linalg.norm(a_dense)
            assert rel_err < 1e-2, 'Vector potential a1 relative error: {}'.format(rel_err)

    def _dense_matmul(self, a: Any, x: np.ndarray) -> np.ndarray:

        if np.isscalar(a) or (isinstance(a, np.ndarray) and a.size == 1):
            if a == 0:
                return np.zeros_like(x)
            return a * x
        return a @ x


# ---------------------------------------------------------------------------
# Tests: Compression ratio
# ---------------------------------------------------------------------------

class TestCompression(object):

    def test_compression_single_particle(self) -> None:

        # Two well-separated spheres give good compression
        cp = make_two_particles(n1 = 150, n2 = 150, offset = 100.0)
        enei = 600.0

        aca_green = TestableACACompGreenRet(cp, htol = 1e-4, cleaf = 16)
        hmat = aca_green.eval(0, 0, 'G', enei)

        eta = hmat.compression()
        assert eta < 1.0, 'Compression ratio {} should be < 1 for n=300'.format(eta)

    def test_compression_two_particles(self) -> None:

        # Two well-separated spheres should give better compression
        cp = make_two_particles(n1 = 100, n2 = 100, offset = 100.0)
        enei = 600.0

        aca_green = TestableACACompGreenRet(cp, htol = 1e-4, cleaf = 16)
        hmat = aca_green.eval(0, 0, 'G', enei)

        eta = hmat.compression()
        assert eta < 1.0, 'Compression ratio {} should be < 1 for two spheres'.format(eta)

    def test_compression_method(self) -> None:

        cp = make_two_particles(n1 = 150, n2 = 150, offset = 100.0)
        aca_green = TestableACACompGreenRet(cp, htol = 1e-4, cleaf = 16)

        eta = aca_green.compression(i = 0, j = 0, key = 'G', enei = 600.0)
        assert 0 < eta < 1.0, 'compression() method returned: {}'.format(eta)

    def test_tighter_tolerance_less_compression(self) -> None:

        cp = make_two_particles(n1 = 100, n2 = 100, offset = 100.0)
        enei = 600.0

        aca_loose = TestableACACompGreenRet(cp, htol = 1e-2, cleaf = 16)
        aca_tight = TestableACACompGreenRet(cp, htol = 1e-8, cleaf = 16)

        hmat_loose = aca_loose.eval(0, 0, 'G', enei)
        hmat_tight = aca_tight.eval(0, 0, 'G', enei)

        eta_loose = hmat_loose.compression()
        eta_tight = hmat_tight.compression()

        # Tighter tolerance should use more storage (higher compression ratio)
        assert eta_tight >= eta_loose, (
            'Tighter tolerance should have higher compression ratio: '
            'tight={}, loose={}'.format(eta_tight, eta_loose))


# ---------------------------------------------------------------------------
# Tests: H-matrix multiply consistency
# ---------------------------------------------------------------------------

class TestHMatrixMultiplyConsistency(object):

    def test_hmat_vec_matches_dense_vec(self) -> None:

        cp = make_single_particle(n = 100)
        enei = 600.0

        aca_green = TestableACACompGreenRet(cp, htol = 1e-6, cleaf = 16)

        dense_mat = aca_green.g.eval(0, 0, 'G', enei)
        hmat = aca_green.eval(0, 0, 'G', enei)

        rng = np.random.RandomState(123)
        v = rng.randn(cp.n) + 1j * rng.randn(cp.n)

        result_hmat = hmat.mtimes_vec(v)
        result_dense = dense_mat @ v

        rel_err = np.linalg.norm(result_hmat - result_dense) / np.linalg.norm(result_dense)
        assert rel_err < 1e-3, 'H-matrix vec multiply error: {}'.format(rel_err)

    def test_hmat_mat_matches_dense_mat(self) -> None:

        cp = make_single_particle(n = 100)
        enei = 600.0

        aca_green = TestableACACompGreenRet(cp, htol = 1e-6, cleaf = 16)

        dense_mat = aca_green.g.eval(0, 0, 'G', enei)
        hmat = aca_green.eval(0, 0, 'G', enei)

        rng = np.random.RandomState(456)
        V = rng.randn(cp.n, 3) + 1j * rng.randn(cp.n, 3)

        result_hmat = hmat.mtimes_vec(V)
        result_dense = dense_mat @ V

        rel_err = np.linalg.norm(result_hmat - result_dense) / np.linalg.norm(result_dense)
        assert rel_err < 1e-3, 'H-matrix matrix multiply error: {}'.format(rel_err)


# ---------------------------------------------------------------------------
# Tests: Cache behavior
# ---------------------------------------------------------------------------

class TestCache(object):

    def test_eval_caches_result(self) -> None:

        cp = make_single_particle(n = 80)
        enei = 600.0

        aca_green = TestableACACompGreenRet(cp, htol = 1e-6, cleaf = 16)

        hmat1 = aca_green.eval(0, 0, 'G', enei)
        hmat2 = aca_green.eval(0, 0, 'G', enei)

        # Should be the same object (cached)
        assert hmat1 is hmat2

    def test_clear_cache(self) -> None:

        cp = make_single_particle(n = 80)
        enei = 600.0

        aca_green = TestableACACompGreenRet(cp, htol = 1e-6, cleaf = 16)

        aca_green.eval(0, 0, 'G', enei)
        assert len(aca_green._cache) > 0

        aca_green.clear_cache()
        assert len(aca_green._cache) == 0

    def test_different_keys_different_cache(self) -> None:

        cp = make_single_particle(n = 80)
        enei = 600.0

        aca_green = TestableACACompGreenRet(cp, htol = 1e-6, cleaf = 16)

        hmat_G = aca_green.eval(0, 0, 'G', enei)
        hmat_F = aca_green.eval(0, 0, 'F', enei)

        # Should be different objects
        assert hmat_G is not hmat_F


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
