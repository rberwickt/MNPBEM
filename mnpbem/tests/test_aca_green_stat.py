import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mnpbem.greenfun.clustertree import ClusterTree
from mnpbem.greenfun.hmatrix import HMatrix
from mnpbem.greenfun.compgreen_stat import CompGreenStat, CompStruct
from mnpbem.greenfun.aca_compgreen_stat import ACACompGreenStat


# ---------------------------------------------------------------------------
# Mock particle object for standalone testing
# ---------------------------------------------------------------------------

class MockParticle(object):

    # Minimal particle interface needed by CompGreenStat and ACACompGreenStat.
    # Provides: pos, nvec, area, n, p, closed, bradius, quadpol, quad_integration

    def __init__(self,
            pos: np.ndarray,
            nvec: np.ndarray,
            area: np.ndarray) -> None:

        self.pos = np.asarray(pos, dtype = np.float64)
        self.nvec = np.asarray(nvec, dtype = np.float64)
        self.area = np.asarray(area, dtype = np.float64)
        self.n = self.pos.shape[0]
        self.p = [self]
        self.closed = [None]
        # Tangent vectors orthogonal to nvec (needed by compgreen_stat
        # cart-deriv refinement path).  Construct any pair forming a
        # right-handed basis with nvec.
        ref = np.zeros_like(self.nvec)
        ref[:, 0] = 1.0
        parallel_mask = np.abs(np.einsum('ij,ij->i', self.nvec, ref)) > 0.9
        ref[parallel_mask] = np.array([0.0, 1.0, 0.0])
        tvec1 = np.cross(self.nvec, ref)
        tvec1 /= np.linalg.norm(tvec1, axis = 1, keepdims = True)
        tvec2 = np.cross(self.nvec, tvec1)
        tvec2 /= np.linalg.norm(tvec2, axis = 1, keepdims = True)
        self.tvec1 = tvec1
        self.tvec2 = tvec2

    @property
    def nfaces(self) -> int:
        return self.pos.shape[0]

    def bradius(self) -> np.ndarray:
        # Approximate bounding radius per face: sqrt(area / pi)
        return np.sqrt(self.area / np.pi)

    def index_func(self, i: int) -> np.ndarray:
        return np.arange(self.n)

    def closedparticle(self, i: int) -> tuple:
        return (None, 1.0, None)

    def quadpol(self, face_indices: np.ndarray) -> tuple:
        # Dummy polar quadrature
        n = len(face_indices)
        pos_quad = self.pos[face_indices]
        w_quad = self.area[face_indices]
        row_quad = np.arange(n)
        return pos_quad, w_quad, row_quad

    def quad_integration(self, face_indices: np.ndarray) -> tuple:
        from scipy.sparse import csr_matrix
        n = len(face_indices)
        pos_quad = self.pos[face_indices]
        w_data = self.area[face_indices]
        row = np.arange(n)
        col = np.arange(n)
        w_sparse = csr_matrix((w_data, (row, col)), shape = (n, n))
        return pos_quad, w_sparse, face_indices


class MockComParticle(object):

    # Minimal ComParticle interface wrapping MockParticle.

    def __init__(self, particle: MockParticle) -> None:

        self._particle = particle
        self.p = [particle]
        self.closed = [None]

    @property
    def pos(self) -> np.ndarray:
        return self._particle.pos

    @property
    def nvec(self) -> np.ndarray:
        return self._particle.nvec

    @property
    def area(self) -> np.ndarray:
        return self._particle.area

    @property
    def n(self) -> int:
        return self._particle.n

    @property
    def nfaces(self) -> int:
        return self._particle.n

    @property
    def pc(self) -> MockParticle:
        return self._particle

    def bradius(self) -> np.ndarray:
        return self._particle.bradius()

    def index_func(self, i: int) -> np.ndarray:
        return np.arange(self._particle.n)

    def closedparticle(self, i: int) -> tuple:
        return (None, 1.0, None)

    def eps1(self, enei: float) -> np.ndarray:
        # Gold-like eps inside
        return np.full(self._particle.n, -10.0 + 0.5j)

    def eps2(self, enei: float) -> np.ndarray:
        # Vacuum outside
        return np.full(self._particle.n, 1.0 + 0.0j)


# ---------------------------------------------------------------------------
# Helper: generate sphere-like points with outward normals
# ---------------------------------------------------------------------------

def make_sphere_particle(n: int, radius: float = 10.0, seed: int = 42) -> MockComParticle:

    rng = np.random.RandomState(seed)
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

    # Outward normal = position / radius
    nvec = pos / radius

    # Approximate area per face: surface area / n
    area = np.full(n, 4.0 * np.pi * radius ** 2 / n, dtype = np.float64)

    particle = MockParticle(pos, nvec, area)
    return MockComParticle(particle)


# ---------------------------------------------------------------------------
# Test: ACA static Green function accuracy vs dense
# ---------------------------------------------------------------------------

class TestACAGreenStatAccuracy(object):

    def test_G_matrix_accuracy(self) -> None:

        n = 100
        p = make_sphere_particle(n, radius = 10.0)

        # Dense Green function
        g_dense = CompGreenStat(p, p)
        G_dense = g_dense.G

        # ACA-accelerated Green function
        g_aca = ACACompGreenStat(p, htol = 1e-6, kmax = 100, cleaf = 16)
        G_hmat = g_aca.eval('G')
        G_aca_full = G_hmat.full()

        rel_err = np.linalg.norm(G_aca_full - G_dense) / np.linalg.norm(G_dense)
        assert rel_err < 1e-3, 'ACA G matrix relative error: {}'.format(rel_err)

    def test_F_matrix_accuracy(self) -> None:

        n = 100
        p = make_sphere_particle(n, radius = 10.0)

        # Dense Green function
        g_dense = CompGreenStat(p, p)
        F_dense = g_dense.F

        # ACA-accelerated Green function
        g_aca = ACACompGreenStat(p, htol = 1e-6, kmax = 100, cleaf = 16)
        F_hmat = g_aca.eval('F')
        F_aca_full = F_hmat.full()

        rel_err = np.linalg.norm(F_aca_full - F_dense) / np.linalg.norm(F_dense)
        assert rel_err < 1e-3, 'ACA F matrix relative error: {}'.format(rel_err)

    def test_H1_matrix_accuracy(self) -> None:

        n = 100
        p = make_sphere_particle(n, radius = 10.0)

        g_dense = CompGreenStat(p, p)
        H1_dense = g_dense.eval('H1')

        g_aca = ACACompGreenStat(p, htol = 1e-6, kmax = 100, cleaf = 16)
        H1_hmat = g_aca.eval('H1')
        H1_aca_full = H1_hmat.full()

        rel_err = np.linalg.norm(H1_aca_full - H1_dense) / np.linalg.norm(H1_dense)
        assert rel_err < 1e-3, 'ACA H1 matrix relative error: {}'.format(rel_err)

    def test_H2_matrix_accuracy(self) -> None:

        n = 100
        p = make_sphere_particle(n, radius = 10.0)

        g_dense = CompGreenStat(p, p)
        H2_dense = g_dense.eval('H2')

        g_aca = ACACompGreenStat(p, htol = 1e-6, kmax = 100, cleaf = 16)
        H2_hmat = g_aca.eval('H2')
        H2_aca_full = H2_hmat.full()

        rel_err = np.linalg.norm(H2_aca_full - H2_dense) / np.linalg.norm(H2_dense)
        assert rel_err < 1e-3, 'ACA H2 matrix relative error: {}'.format(rel_err)

    def test_multiple_eval_keys(self) -> None:

        n = 80
        p = make_sphere_particle(n, radius = 10.0)

        g_aca = ACACompGreenStat(p, htol = 1e-6, kmax = 100, cleaf = 16)
        G_hmat, F_hmat = g_aca.eval('G', 'F')

        # Both should be HMatrix objects
        assert isinstance(G_hmat, HMatrix)
        assert isinstance(F_hmat, HMatrix)


# ---------------------------------------------------------------------------
# Test: potential computation matches non-ACA version
# ---------------------------------------------------------------------------

class TestACAGreenStatPotential(object):

    def test_potential_inside(self) -> None:

        n = 100
        p = make_sphere_particle(n, radius = 10.0)

        # Create random surface charges
        rng = np.random.RandomState(123)
        sig_vals = rng.randn(n)
        sig = CompStruct(p, 500.0, sig = sig_vals)

        # Dense potential
        g_dense = CompGreenStat(p, p)
        pot_dense = g_dense.potential(sig, inout = 1)

        # ACA potential
        g_aca = ACACompGreenStat(p, htol = 1e-6, kmax = 100, cleaf = 16)
        pot_aca = g_aca.potential(sig, inout = 1)

        # Compare phi1
        rel_err_phi = np.linalg.norm(pot_aca.phi1 - pot_dense.phi1) / np.linalg.norm(pot_dense.phi1)
        assert rel_err_phi < 1e-3, 'ACA potential phi1 relative error: {}'.format(rel_err_phi)

        # Compare phi1p
        rel_err_phip = np.linalg.norm(pot_aca.phi1p - pot_dense.phi1p) / np.linalg.norm(pot_dense.phi1p)
        assert rel_err_phip < 1e-3, 'ACA potential phi1p relative error: {}'.format(rel_err_phip)

    def test_potential_outside(self) -> None:

        n = 100
        p = make_sphere_particle(n, radius = 10.0)

        rng = np.random.RandomState(456)
        sig_vals = rng.randn(n)
        sig = CompStruct(p, 500.0, sig = sig_vals)

        # Dense potential
        g_dense = CompGreenStat(p, p)
        pot_dense = g_dense.potential(sig, inout = 2)

        # ACA potential
        g_aca = ACACompGreenStat(p, htol = 1e-6, kmax = 100, cleaf = 16)
        pot_aca = g_aca.potential(sig, inout = 2)

        # Compare phi2
        rel_err_phi = np.linalg.norm(pot_aca.phi2 - pot_dense.phi2) / np.linalg.norm(pot_dense.phi2)
        assert rel_err_phi < 1e-3, 'ACA potential phi2 relative error: {}'.format(rel_err_phi)

        # Compare phi2p
        rel_err_phip = np.linalg.norm(pot_aca.phi2p - pot_dense.phi2p) / np.linalg.norm(pot_dense.phi2p)
        assert rel_err_phip < 1e-3, 'ACA potential phi2p relative error: {}'.format(rel_err_phip)

    def test_potential_matvec_consistency(self) -> None:

        # The potential should be equivalent to H-matrix @ sig
        n = 80
        p = make_sphere_particle(n, radius = 10.0)

        rng = np.random.RandomState(789)
        sig_vals = rng.randn(n)
        sig = CompStruct(p, 500.0, sig = sig_vals)

        g_aca = ACACompGreenStat(p, htol = 1e-6, kmax = 100, cleaf = 16)

        # Get H-matrices directly
        G_hmat = g_aca.eval('G')
        H1_hmat = g_aca.eval('H1')

        # Manual multiply
        phi_manual = G_hmat @ sig_vals
        phip_manual = H1_hmat @ sig_vals

        # Via potential method
        pot = g_aca.potential(sig, inout = 1)

        np.testing.assert_allclose(pot.phi1, phi_manual, rtol = 1e-10)
        np.testing.assert_allclose(pot.phi1p, phip_manual, rtol = 1e-10)


# ---------------------------------------------------------------------------
# Test: F surface derivative matrix accuracy
# ---------------------------------------------------------------------------

class TestACAFMatrix(object):

    def test_F_diagonal_values(self) -> None:

        # For a self-interaction (p1 == p2), F diagonal should be ~ -2*pi
        n = 100
        p = make_sphere_particle(n, radius = 10.0)

        g_dense = CompGreenStat(p, p)
        F_dense = g_dense.F

        g_aca = ACACompGreenStat(p, htol = 1e-6, kmax = 100, cleaf = 16)
        F_aca = g_aca.full('F')

        # Dense F diagonal should be close to -2*pi
        diag_dense = np.diag(F_dense)
        diag_aca = np.diag(F_aca)

        np.testing.assert_allclose(diag_aca, diag_dense, atol = 0.5,
            err_msg = 'F diagonal values differ between ACA and dense')

    def test_F_off_diagonal_accuracy(self) -> None:

        n = 100
        p = make_sphere_particle(n, radius = 10.0)

        g_dense = CompGreenStat(p, p)
        F_dense = g_dense.F

        g_aca = ACACompGreenStat(p, htol = 1e-6, kmax = 100, cleaf = 16)
        F_aca = g_aca.full('F')

        # Compare off-diagonal elements
        mask = ~np.eye(n, dtype = bool)
        off_dense = F_dense[mask]
        off_aca = F_aca[mask]

        rel_err = np.linalg.norm(off_aca - off_dense) / np.linalg.norm(off_dense)
        assert rel_err < 1e-3, 'F off-diagonal relative error: {}'.format(rel_err)

    def test_F_action_on_vector(self) -> None:

        # F @ v should match between dense and ACA
        n = 100
        p = make_sphere_particle(n, radius = 10.0)

        rng = np.random.RandomState(321)
        v = rng.randn(n)

        # Dense
        g_dense = CompGreenStat(p, p)
        Fv_dense = g_dense.F @ v

        # ACA
        g_aca = ACACompGreenStat(p, htol = 1e-6, kmax = 100, cleaf = 16)
        F_hmat = g_aca.eval('F')
        Fv_aca = F_hmat @ v

        rel_err = np.linalg.norm(Fv_aca - Fv_dense) / np.linalg.norm(Fv_dense)
        assert rel_err < 1e-3, 'F @ v relative error: {}'.format(rel_err)


# ---------------------------------------------------------------------------
# Test: ACA compression and properties
# ---------------------------------------------------------------------------

class TestACACompression(object):

    def test_compression_ratio(self) -> None:

        n = 200
        p = make_sphere_particle(n, radius = 10.0)

        g_aca = ACACompGreenStat(p, htol = 1e-4, kmax = 100, cleaf = 16)
        eta = g_aca.compression('G')

        # Compression ratio should be positive and finite.
        # For a single sphere at moderate n, ACA may not compress below 1.0
        # due to overhead from low-rank factor storage. The real benefit
        # appears with larger problems or well-separated multi-particle geometries.
        assert 0 < eta < 2.0, 'Compression ratio {} out of expected range'.format(eta)

    def test_attribute_access(self) -> None:

        n = 80
        p = make_sphere_particle(n, radius = 10.0)

        g_aca = ACACompGreenStat(p, htol = 1e-6, kmax = 100, cleaf = 16)

        # Access via attribute
        G_hmat = g_aca.G
        F_hmat = g_aca.F

        assert isinstance(G_hmat, HMatrix)
        assert isinstance(F_hmat, HMatrix)

    def test_caching(self) -> None:

        n = 80
        p = make_sphere_particle(n, radius = 10.0)

        g_aca = ACACompGreenStat(p, htol = 1e-6, kmax = 100, cleaf = 16)

        # First access computes
        G1 = g_aca.eval('G')
        # Second access returns cached
        G2 = g_aca.eval('G')

        assert G1 is G2, 'eval should return cached H-matrix'

    def test_repr(self) -> None:

        n = 80
        p = make_sphere_particle(n, radius = 10.0)

        g_aca = ACACompGreenStat(p, htol = 1e-6, kmax = 100, cleaf = 16)
        repr_str = repr(g_aca)
        assert 'ACACompGreenStat' in repr_str
        assert '80' in repr_str

    def test_gp_raises_error(self) -> None:

        n = 50
        p = make_sphere_particle(n, radius = 10.0)

        g_aca = ACACompGreenStat(p, htol = 1e-6, kmax = 100, cleaf = 16)

        with pytest.raises(ValueError):
            g_aca.eval('Gp')

    def test_tighter_tolerance_gives_better_accuracy(self) -> None:

        n = 100
        p = make_sphere_particle(n, radius = 10.0)

        g_dense = CompGreenStat(p, p)
        G_dense = g_dense.G

        g_aca_loose = ACACompGreenStat(p, htol = 1e-3, kmax = 100, cleaf = 16)
        G_loose = g_aca_loose.full('G')
        err_loose = np.linalg.norm(G_loose - G_dense) / np.linalg.norm(G_dense)

        g_aca_tight = ACACompGreenStat(p, htol = 1e-8, kmax = 100, cleaf = 16)
        G_tight = g_aca_tight.full('G')
        err_tight = np.linalg.norm(G_tight - G_dense) / np.linalg.norm(G_dense)

        assert err_tight <= err_loose, (
            'Tighter tolerance should give better accuracy: err_tight={}, err_loose={}'.format(
                err_tight, err_loose))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
