import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mnpbem.greenfun.clustertree import ClusterTree
from mnpbem.greenfun.hmatrix import HMatrix


# ---------------------------------------------------------------------------
# Helper: generate random 3D points on a sphere
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


def two_sphere_points(n1: int, n2: int,
        r1: float = 10.0, r2: float = 10.0,
        offset: float = 50.0) -> tuple:

    pos1 = sphere_points(n1, r1)
    pos2 = sphere_points(n2, r2)
    pos2[:, 0] += offset

    total = n1 + n2
    pos = np.empty((total, 3), dtype = np.float64)
    pos[:n1] = pos1
    pos[n1:] = pos2

    ipart = np.empty(total, dtype = np.int64)
    ipart[:n1] = 0
    ipart[n1:] = 1

    return pos, ipart


def green_kernel(pos: np.ndarray) -> callable:

    # 1/r Green's function kernel
    def fun(row: np.ndarray, col: np.ndarray) -> np.ndarray:
        r1 = pos[row]
        r2 = pos[col]
        dist = np.linalg.norm(r1 - r2, axis = 1)
        # Avoid division by zero on diagonal
        dist = np.maximum(dist, 1e-10)
        return 1.0 / (4.0 * np.pi * dist)

    return fun


# ---------------------------------------------------------------------------
# ClusterTree tests
# ---------------------------------------------------------------------------

class TestClusterTree(object):

    def test_basic_construction(self) -> None:

        pos = sphere_points(100)
        tree = ClusterTree(pos, cleaf = 16)

        # Tree should have correct number of face indices
        assert tree.n == 100
        assert tree.ind.shape == (100, 2)
        assert tree.son.shape[0] > 0
        assert tree.cind.shape[1] == 2

    def test_bisection_leaf_size(self) -> None:

        pos = sphere_points(200)
        cleaf = 20
        tree = ClusterTree(pos, cleaf = cleaf)

        # All leaves should have size <= cleaf (or be forced split due to particles)
        num_nodes = tree.son.shape[0]
        for i in range(num_nodes):
            if tree.son[i, 0] == -1:  # leaf
                sz = tree.cind[i, 1] - tree.cind[i, 0] + 1
                assert sz <= cleaf, 'Leaf node {} has size {} > cleaf {}'.format(i, sz, cleaf)

    def test_index_mapping_bijective(self) -> None:

        pos = sphere_points(100)
        tree = ClusterTree(pos, cleaf = 16)

        # ind[:, 0] should be a permutation of 0..n-1
        assert set(tree.ind[:, 0].tolist()) == set(range(100))
        # ind[:, 1] should be a permutation of 0..n-1
        assert set(tree.ind[:, 1].tolist()) == set(range(100))

    def test_part2cluster_cluster2part_roundtrip(self) -> None:

        pos = sphere_points(80)
        tree = ClusterTree(pos, cleaf = 10)

        rng = np.random.RandomState(123)
        v = rng.randn(80)

        # part2cluster then cluster2part should give back original
        v_cluster = tree.part2cluster(v)
        v_back = tree.cluster2part(v_cluster)
        np.testing.assert_allclose(v_back, v, atol = 1e-14)

    def test_part2cluster_cluster2part_roundtrip_reverse(self) -> None:

        pos = sphere_points(80)
        tree = ClusterTree(pos, cleaf = 10)

        rng = np.random.RandomState(456)
        v = rng.randn(80)

        # cluster2part then part2cluster should also give back original
        v_part = tree.cluster2part(v)
        v_back = tree.part2cluster(v_part)
        np.testing.assert_allclose(v_back, v, atol = 1e-14)

    def test_matsize(self) -> None:

        pos = sphere_points(50)
        tree = ClusterTree(pos, cleaf = 10)
        assert tree.matsize(tree) == (50, 50)

    def test_two_particles(self) -> None:

        pos, ipart = two_sphere_points(50, 50, offset = 100.0)
        tree = ClusterTree(pos, cleaf = 16, ipart_arr = ipart)

        assert tree.n == 100
        # Root should span composite particles
        assert tree.ipart[0] == -1

    def test_bounding_box(self) -> None:

        pos = sphere_points(100)
        tree = ClusterTree(pos, cleaf = 16)

        # Root bounding box should encompass all points
        root_mid = tree.mid[0]
        root_rad = tree.rad[0]
        for i in range(100):
            dist = np.linalg.norm(pos[i] - root_mid)
            assert dist <= root_rad + 1e-10, (
                'Point {} at distance {} exceeds root radius {}'.format(i, dist, root_rad))


# ---------------------------------------------------------------------------
# Admissibility tests
# ---------------------------------------------------------------------------

class TestAdmissibility(object):

    def test_admissibility_returns_entries(self) -> None:

        pos = sphere_points(100)
        tree = ClusterTree(pos, cleaf = 16)
        admiss = tree.admissibility(tree)

        # Should have some entries
        assert len(admiss) > 0
        # All values should be 1 (low-rank) or 2 (dense/leaf)
        for key, val in admiss.items():
            assert val in (1, 2), 'Unexpected admissibility value: {}'.format(val)

    def test_admissibility_covers_all_leaves(self) -> None:

        pos = sphere_points(64)
        tree = ClusterTree(pos, cleaf = 8)
        admiss = tree.admissibility(tree)

        # The admissibility entries should partition the full matrix
        # Check total coverage
        total_elements = 0
        for (i1, i2), val in admiss.items():
            s1 = tree.cind[i1, 1] - tree.cind[i1, 0] + 1
            s2 = tree.cind[i2, 1] - tree.cind[i2, 0] + 1
            total_elements += s1 * s2

        assert total_elements == 64 * 64, (
            'Admissibility covers {} elements, expected {}'.format(total_elements, 64 * 64))

    def test_custom_admissibility_function(self) -> None:

        pos = sphere_points(100)
        tree = ClusterTree(pos, cleaf = 16)

        # Strict admissibility: more low-rank blocks
        strict = lambda rad1, rad2, dist: 1.0 * min(rad1, rad2) < dist
        admiss_strict = tree.admissibility(tree, fadmiss = strict)

        # Loose admissibility: fewer low-rank blocks
        loose = lambda rad1, rad2, dist: 10.0 * min(rad1, rad2) < dist
        admiss_loose = tree.admissibility(tree, fadmiss = loose)

        n_lr_strict = sum(1 for v in admiss_strict.values() if v == 1)
        n_lr_loose = sum(1 for v in admiss_loose.values() if v == 1)

        # Stricter condition should yield more low-rank blocks
        assert n_lr_strict >= n_lr_loose


# ---------------------------------------------------------------------------
# ACA tests
# ---------------------------------------------------------------------------

class TestACA(object):

    def test_aca_approximation_accuracy(self) -> None:

        n = 100
        pos = sphere_points(n)
        tree = ClusterTree(pos, cleaf = 16)
        fun = green_kernel(pos)

        hmat = HMatrix.from_func(tree, fun, htol = 1e-6)

        # Build the full dense matrix for comparison
        row_grid, col_grid = np.meshgrid(np.arange(n), np.arange(n), indexing = 'ij')
        A_dense = fun(row_grid.ravel(), col_grid.ravel()).reshape(n, n)

        # Convert H-matrix to full and compare
        A_hmat = hmat.full()

        rel_err = np.linalg.norm(A_hmat - A_dense) / np.linalg.norm(A_dense)
        assert rel_err < 1e-3, 'ACA relative error {} is too large'.format(rel_err)

    def test_aca_compression(self) -> None:

        # Two well-separated spheres give good compression
        pos, ipart = two_sphere_points(100, 100, offset = 200.0)
        n = 200
        tree = ClusterTree(pos, cleaf = 16, ipart_arr = ipart)
        fun = green_kernel(pos)

        hmat = HMatrix.from_func(tree, fun, htol = 1e-4)

        eta = hmat.compression()
        # H-matrix should use fewer elements than full matrix
        assert eta < 1.0, 'Compression ratio {} should be < 1'.format(eta)


# ---------------------------------------------------------------------------
# H-matrix multiply tests
# ---------------------------------------------------------------------------

class TestHMatrixMultiply(object):

    def test_hmat_vec_multiply(self) -> None:

        n = 100
        pos = sphere_points(n)
        tree = ClusterTree(pos, cleaf = 16)
        fun = green_kernel(pos)

        hmat = HMatrix.from_func(tree, fun, htol = 1e-6)

        # Dense reference
        row_grid, col_grid = np.meshgrid(np.arange(n), np.arange(n), indexing = 'ij')
        A_dense = fun(row_grid.ravel(), col_grid.ravel()).reshape(n, n)

        rng = np.random.RandomState(99)
        v = rng.randn(n)

        result_hmat = hmat @ v
        result_dense = A_dense @ v

        rel_err = np.linalg.norm(result_hmat - result_dense) / np.linalg.norm(result_dense)
        assert rel_err < 1e-3, 'H-matrix * vec relative error: {}'.format(rel_err)

    def test_hmat_mat_multiply(self) -> None:

        n = 80
        pos = sphere_points(n)
        tree = ClusterTree(pos, cleaf = 16)
        fun = green_kernel(pos)

        hmat = HMatrix.from_func(tree, fun, htol = 1e-6)

        # Dense reference
        row_grid, col_grid = np.meshgrid(np.arange(n), np.arange(n), indexing = 'ij')
        A_dense = fun(row_grid.ravel(), col_grid.ravel()).reshape(n, n)

        rng = np.random.RandomState(77)
        V = rng.randn(n, 3)

        result_hmat = hmat @ V
        result_dense = A_dense @ V

        rel_err = np.linalg.norm(result_hmat - result_dense) / np.linalg.norm(result_dense)
        assert rel_err < 1e-3, 'H-matrix * matrix relative error: {}'.format(rel_err)

    def test_scalar_multiply(self) -> None:

        n = 50
        pos = sphere_points(n)
        tree = ClusterTree(pos, cleaf = 10)
        fun = green_kernel(pos)

        hmat = HMatrix.from_func(tree, fun, htol = 1e-6)
        hmat2 = 3.0 * hmat

        rng = np.random.RandomState(55)
        v = rng.randn(n)

        result1 = hmat @ v
        result2 = hmat2 @ v

        np.testing.assert_allclose(result2, 3.0 * result1, rtol = 1e-10)


# ---------------------------------------------------------------------------
# H-matrix solve tests
# ---------------------------------------------------------------------------

class TestHMatrixSolve(object):

    def test_solve_matches_dense(self) -> None:

        n = 60
        pos = sphere_points(n)
        tree = ClusterTree(pos, cleaf = 10)

        # Use a diagonally dominant kernel for stable solve
        def fun(row: np.ndarray, col: np.ndarray) -> np.ndarray:
            r1 = pos[row]
            r2 = pos[col]
            dist = np.linalg.norm(r1 - r2, axis = 1)
            dist = np.maximum(dist, 1e-10)
            vals = 1.0 / (4.0 * np.pi * dist)
            # Add large diagonal for well-conditioning
            vals[row == col] += 10.0
            return vals

        hmat = HMatrix.from_func(tree, fun, htol = 1e-8)

        # Dense reference
        row_grid, col_grid = np.meshgrid(np.arange(n), np.arange(n), indexing = 'ij')
        A_dense = fun(row_grid.ravel(), col_grid.ravel()).reshape(n, n)

        rng = np.random.RandomState(33)
        b = rng.randn(n)

        # Dense solve
        x_dense = np.linalg.solve(A_dense, b)

        # H-matrix solve
        x_hmat = hmat.solve(b)

        rel_err = np.linalg.norm(x_hmat - x_dense) / np.linalg.norm(x_dense)
        assert rel_err < 1e-3, 'H-matrix solve relative error: {}'.format(rel_err)

    def test_lu_solve(self) -> None:

        n = 60
        pos = sphere_points(n)
        tree = ClusterTree(pos, cleaf = 10)

        def fun(row: np.ndarray, col: np.ndarray) -> np.ndarray:
            r1 = pos[row]
            r2 = pos[col]
            dist = np.linalg.norm(r1 - r2, axis = 1)
            dist = np.maximum(dist, 1e-10)
            vals = 1.0 / (4.0 * np.pi * dist)
            vals[row == col] += 10.0
            return vals

        hmat = HMatrix.from_func(tree, fun, htol = 1e-8)

        rng = np.random.RandomState(44)
        b = rng.randn(n)

        # LU solve
        hmat_lu = hmat._copy()
        hmat_lu.lu()
        x_lu = hmat_lu.solve(b)

        # Dense reference
        row_grid, col_grid = np.meshgrid(np.arange(n), np.arange(n), indexing = 'ij')
        A_dense = fun(row_grid.ravel(), col_grid.ravel()).reshape(n, n)
        x_dense = np.linalg.solve(A_dense, b)

        rel_err = np.linalg.norm(x_lu - x_dense) / np.linalg.norm(x_dense)
        assert rel_err < 1e-3, 'LU solve relative error: {}'.format(rel_err)


# ---------------------------------------------------------------------------
# Full reconstruction tests
# ---------------------------------------------------------------------------

class TestFullReconstruction(object):

    def test_full_matches_original(self) -> None:

        n = 80
        pos = sphere_points(n)
        tree = ClusterTree(pos, cleaf = 16)
        fun = green_kernel(pos)

        hmat = HMatrix.from_func(tree, fun, htol = 1e-6)

        # Dense reference
        row_grid, col_grid = np.meshgrid(np.arange(n), np.arange(n), indexing = 'ij')
        A_dense = fun(row_grid.ravel(), col_grid.ravel()).reshape(n, n)

        A_full = hmat.full()

        rel_err = np.linalg.norm(A_full - A_dense) / np.linalg.norm(A_dense)
        assert rel_err < 1e-3, 'full() reconstruction relative error: {}'.format(rel_err)

    def test_full_symmetry(self) -> None:

        n = 50
        pos = sphere_points(n)
        tree = ClusterTree(pos, cleaf = 10)
        fun = green_kernel(pos)

        hmat = HMatrix.from_func(tree, fun, htol = 1e-8)
        A_full = hmat.full()

        # 1/r kernel is symmetric
        asym = np.linalg.norm(A_full - A_full.T) / np.linalg.norm(A_full)
        # ACA may introduce small asymmetry, but should be within tolerance
        assert asym < 1e-2, 'Asymmetry of full() for symmetric kernel: {}'.format(asym)


# ---------------------------------------------------------------------------
# Additional H-matrix operation tests
# ---------------------------------------------------------------------------

class TestHMatrixOperations(object):

    def test_negation(self) -> None:

        n = 50
        pos = sphere_points(n)
        tree = ClusterTree(pos, cleaf = 10)
        fun = green_kernel(pos)

        hmat = HMatrix.from_func(tree, fun, htol = 1e-6)
        neg_hmat = -hmat

        A1 = hmat.full()
        A2 = neg_hmat.full()
        np.testing.assert_allclose(A2, -A1, atol = 1e-12)

    def test_addition(self) -> None:

        n = 50
        pos = sphere_points(n)
        tree = ClusterTree(pos, cleaf = 10)
        fun = green_kernel(pos)

        hmat = HMatrix.from_func(tree, fun, htol = 1e-6)
        hmat_sum = hmat + hmat

        A1 = hmat.full()
        A_sum = hmat_sum.full()

        rel_err = np.linalg.norm(A_sum - 2.0 * A1) / np.linalg.norm(2.0 * A1)
        assert rel_err < 1e-2, 'H-matrix addition relative error: {}'.format(rel_err)

    def test_subtraction(self) -> None:

        n = 50
        pos = sphere_points(n)
        tree = ClusterTree(pos, cleaf = 10)
        fun = green_kernel(pos)

        hmat = HMatrix.from_func(tree, fun, htol = 1e-6)
        hmat_diff = hmat - hmat

        A_diff = hmat_diff.full()
        assert np.linalg.norm(A_diff) < 1e-8, (
            'H-matrix self-subtraction norm: {}'.format(np.linalg.norm(A_diff)))

    def test_diag(self) -> None:

        n = 50
        pos = sphere_points(n)
        tree = ClusterTree(pos, cleaf = 10)
        fun = green_kernel(pos)

        hmat = HMatrix.from_func(tree, fun, htol = 1e-8)
        d = hmat.diag()

        # Compare with dense diagonal
        A = hmat.full()
        d_dense = np.diag(A)

        np.testing.assert_allclose(d, d_dense, atol = 1e-6)

    def test_eye(self) -> None:

        n = 50
        pos = sphere_points(n)
        tree = ClusterTree(pos, cleaf = 10)
        fun = green_kernel(pos)

        hmat = HMatrix.from_func(tree, fun, htol = 1e-6)
        eye_hmat = hmat.eye_hmat()

        A_eye = eye_hmat.full()
        np.testing.assert_allclose(A_eye, np.eye(n), atol = 1e-12)

    def test_pad(self) -> None:

        n = 50
        pos = sphere_points(n)
        tree = ClusterTree(pos, cleaf = 10)

        hmat = HMatrix(tree = tree, htol = 1e-6)
        hmat.pad()

        # All blocks should be zero-filled
        for v in hmat.val:
            assert v is not None
            np.testing.assert_array_equal(v, np.zeros_like(v))

    def test_compression_ratio(self) -> None:

        # Two well-separated spheres give good compression
        pos, ipart = two_sphere_points(100, 100, offset = 200.0)
        n = 200
        tree = ClusterTree(pos, cleaf = 16, ipart_arr = ipart)
        fun = green_kernel(pos)

        hmat = HMatrix.from_func(tree, fun, htol = 1e-4)

        eta = hmat.compression()
        assert 0 < eta < 1.0, 'Compression ratio {} out of expected range'.format(eta)

    def test_truncate_reduces_rank(self) -> None:

        n = 100
        pos = sphere_points(n)
        tree = ClusterTree(pos, cleaf = 16)
        fun = green_kernel(pos)

        hmat = HMatrix.from_func(tree, fun, htol = 1e-8)

        # Count total rank
        def total_rank(h: HMatrix) -> int:
            r = 0
            for l in h.lhs:
                if l is not None:
                    r += l.shape[1]
            return r

        rank_before = total_rank(hmat)

        # Truncate with looser tolerance
        hmat_trunc = hmat._copy()
        hmat_trunc.truncate(htol = 1e-2)
        rank_after = total_rank(hmat_trunc)

        assert rank_after <= rank_before, (
            'Truncation should not increase rank: {} -> {}'.format(rank_before, rank_after))


# ---------------------------------------------------------------------------
# Two-particle cluster tree tests
# ---------------------------------------------------------------------------

class TestTwoParticles(object):

    def test_two_particle_tree(self) -> None:

        pos, ipart = two_sphere_points(50, 50, offset = 100.0)
        tree = ClusterTree(pos, cleaf = 16, ipart_arr = ipart)

        assert tree.n == 100
        # Should have particle split at root level
        admiss = tree.admissibility(tree)
        assert len(admiss) > 0

    def test_two_particle_hmatrix(self) -> None:

        pos, ipart = two_sphere_points(40, 40, offset = 100.0)
        tree = ClusterTree(pos, cleaf = 10, ipart_arr = ipart)

        n = 80
        fun = green_kernel(pos)
        hmat = HMatrix.from_func(tree, fun, htol = 1e-6)

        row_grid, col_grid = np.meshgrid(np.arange(n), np.arange(n), indexing = 'ij')
        A_dense = fun(row_grid.ravel(), col_grid.ravel()).reshape(n, n)

        A_full = hmat.full()
        rel_err = np.linalg.norm(A_full - A_dense) / np.linalg.norm(A_dense)
        assert rel_err < 1e-3, 'Two-particle full() error: {}'.format(rel_err)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
