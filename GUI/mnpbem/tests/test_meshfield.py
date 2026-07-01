"""
Tests for MeshField class.

Tests near-field computation on a mesh grid from BEM solutions.
Validates against MATLAB MNPBEM @meshfield reference values.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mnpbem.materials.eps_table import EpsTable
from mnpbem.materials.eps_const import EpsConst
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.geometry.compoint import ComPoint, Point
from mnpbem.greenfun.compgreen_stat import CompGreenStat, CompStruct
from mnpbem.simulation import PlaneWaveStat, MeshField
from mnpbem.bem import BEMStat


# ============================================================================
# MATLAB reference values
# ============================================================================
# Generated with MATLAB MNPBEM:
#   epstab = {epsconst(1), epstable('gold.dat')};
#   p = comparticle(epstab, {trisphere(144,20)}, [2,1], 1,
#       bemoptions('sim','stat','interp','curv'));
#   bem = bemsolver(p, bemoptions('sim','stat','interp','curv'));
#   exc = planewave([1,0,0],[0,0,1], bemoptions('sim','stat'));
#   sig = bem \ exc(p, 520);
#   [x, z] = meshgrid(linspace(-30,30,5), linspace(-30,30,5));
#   y = 0*x;
#   mf = meshfield(p, x, y, z, bemoptions('sim','stat','interp','curv'));
#   [e, h] = mf.field(sig);

MATLAB_ENORM_CURV = np.array([
    [0.035537, 0.057540, 0.063577, 0.057550, 0.035546],
    [0.083876, 0.284434, 0.506236, 0.284349, 0.083898],
    [0.127140, 1.007148, 1.717132, 1.007596, 0.127158],
    [0.083885, 0.284348, 0.506940, 0.284181, 0.083866],
    [0.035545, 0.057556, 0.063584, 0.057533, 0.035534],
])


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope='module')
def gold_sphere_setup():
    """Set up a 20nm gold sphere BEM solve at 520nm."""
    epstab = [EpsConst(1.0), EpsTable('gold.dat')]
    sphere = trisphere(144, 20)
    p = ComParticle(epstab, [sphere], [[2, 1]])
    bem = BEMStat(p)
    exc = PlaneWaveStat([1, 0, 0])
    pot = exc.potential(p, 520)
    sig, _ = bem.solve(pot)
    return p, sig, exc


@pytest.fixture(scope='module')
def grid_5x5():
    """5x5 grid in the xz-plane from -30 to 30nm."""
    x_1d = np.linspace(-30, 30, 5)
    z_1d = np.linspace(-30, 30, 5)
    x, z = np.meshgrid(x_1d, z_1d)
    y = np.zeros_like(x)
    return x, y, z


# ============================================================================
# MeshField construction tests
# ============================================================================

class TestMeshFieldConstruction(object):

    def test_basic_construction(self, gold_sphere_setup, grid_5x5):
        p, sig, exc = gold_sphere_setup
        x, y, z = grid_5x5
        mf = MeshField(p, x, y, z)
        assert mf.pt is not None
        assert mf.g is not None
        assert mf.x.shape == (5, 5)
        assert mf.y.shape == (5, 5)
        assert mf.z.shape == (5, 5)

    def test_scalar_z(self, gold_sphere_setup):
        p, sig, exc = gold_sphere_setup
        x = np.linspace(-30, 30, 5)
        y = np.linspace(-30, 30, 5)
        x2d, y2d = np.meshgrid(x, y)
        mf = MeshField(p, x2d, y2d, 0.0)
        assert mf.z.shape == mf.x.shape
        assert np.all(mf.z == 0.0)

    def test_default_z(self, gold_sphere_setup):
        p, sig, exc = gold_sphere_setup
        x = np.linspace(-30, 30, 3)
        y = np.linspace(-30, 30, 3)
        x2d, y2d = np.meshgrid(x, y)
        mf = MeshField(p, x2d, y2d)
        assert mf.z.shape == mf.x.shape
        assert np.all(mf.z == 0.0)

    def test_compoint_has_correct_npoints(self, gold_sphere_setup, grid_5x5):
        p, sig, exc = gold_sphere_setup
        x, y, z = grid_5x5
        mf = MeshField(p, x, y, z)
        # 25 total points, but (0,0,0) is inside the sphere
        # Some may be classified differently
        assert mf.pt.n >= 24  # at least 24 points should be outside

    def test_nmax_construction(self, gold_sphere_setup, grid_5x5):
        p, sig, exc = gold_sphere_setup
        x, y, z = grid_5x5
        mf = MeshField(p, x, y, z, nmax=10)
        assert mf.nmax == 10
        assert mf.g is None  # Green function not precomputed

    def test_pos_property(self, gold_sphere_setup, grid_5x5):
        p, sig, exc = gold_sphere_setup
        x, y, z = grid_5x5
        mf = MeshField(p, x, y, z)
        pos = mf.pos
        assert pos.ndim == 2
        assert pos.shape[1] == 3

    def test_repr(self, gold_sphere_setup, grid_5x5):
        p, sig, exc = gold_sphere_setup
        x, y, z = grid_5x5
        mf = MeshField(p, x, y, z)
        r = repr(mf)
        assert 'MeshField' in r


# ============================================================================
# Expand tests
# ============================================================================

class TestExpand(object):

    def test_all_same_shape(self):
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])
        z = np.array([[9, 10], [11, 12]])
        rx, ry, rz = MeshField._expand(x, y, z)
        np.testing.assert_array_equal(rx, x)
        np.testing.assert_array_equal(ry, y)
        np.testing.assert_array_equal(rz, z)

    def test_scalar_z(self):
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])
        z = np.float64(0.0)
        rx, ry, rz = MeshField._expand(x, y, z)
        assert rx.shape == x.shape
        assert ry.shape == y.shape
        assert rz.shape == x.shape
        np.testing.assert_array_equal(rz, np.zeros_like(x))

    def test_1d_z_expansion(self):
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])
        z = np.array([0, 1, 2])
        rx, ry, rz = MeshField._expand(x, y, z)
        assert rx.shape == (2, 2, 3)
        assert ry.shape == (2, 2, 3)
        assert rz.shape == (2, 2, 3)


# ============================================================================
# Field computation tests
# ============================================================================

class TestMeshFieldField(object):

    def test_field_returns_tuple(self, gold_sphere_setup, grid_5x5):
        p, sig, exc = gold_sphere_setup
        x, y, z = grid_5x5
        mf = MeshField(p, x, y, z)
        result = mf.field(sig)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_field_e_shape(self, gold_sphere_setup, grid_5x5):
        p, sig, exc = gold_sphere_setup
        x, y, z = grid_5x5
        mf = MeshField(p, x, y, z)
        e, h = mf.field(sig)
        assert e.shape == (5, 5, 3)

    def test_field_h_none_for_stat(self, gold_sphere_setup, grid_5x5):
        p, sig, exc = gold_sphere_setup
        x, y, z = grid_5x5
        mf = MeshField(p, x, y, z)
        e, h = mf.field(sig)
        assert h is None  # quasistatic: no magnetic field

    def test_field_is_complex(self, gold_sphere_setup, grid_5x5):
        p, sig, exc = gold_sphere_setup
        x, y, z = grid_5x5
        mf = MeshField(p, x, y, z)
        e, h = mf.field(sig)
        assert np.iscomplexobj(e)

    def test_field_matches_matlab(self, gold_sphere_setup, grid_5x5):
        """Test that E-field norms match MATLAB within 5%."""
        p, sig, exc = gold_sphere_setup
        x, y, z = grid_5x5
        mf = MeshField(p, x, y, z)
        e, h = mf.field(sig)
        enorm = np.sqrt(np.sum(np.abs(e) ** 2, axis=-1))

        # Compare with MATLAB reference
        rel_err = np.abs(enorm - MATLAB_ENORM_CURV) / MATLAB_ENORM_CURV
        assert rel_err.max() < 0.05, (
            "Max relative error {:.2f}% exceeds 5%: {}".format(
                rel_err.max() * 100,
                list(zip(
                    [(x[i,j], z[i,j]) for i in range(5) for j in range(5)],
                    enorm.ravel(),
                    MATLAB_ENORM_CURV.ravel(),
                    rel_err.ravel()
                ))
            )
        )

    def test_field_callable(self, gold_sphere_setup, grid_5x5):
        """Test __call__ interface."""
        p, sig, exc = gold_sphere_setup
        x, y, z = grid_5x5
        mf = MeshField(p, x, y, z)
        e1, h1 = mf.field(sig)
        e2, h2 = mf(sig)
        np.testing.assert_allclose(e1, e2)

    def test_field_with_nmax(self, gold_sphere_setup, grid_5x5):
        """Test batched computation gives same result."""
        p, sig, exc = gold_sphere_setup
        x, y, z = grid_5x5
        mf1 = MeshField(p, x, y, z)
        mf2 = MeshField(p, x, y, z, nmax=10)
        e1, _ = mf1.field(sig)
        e2, _ = mf2.field(sig)
        np.testing.assert_allclose(np.abs(e1), np.abs(e2), rtol=0.01)

    def test_field_symmetry(self, gold_sphere_setup, grid_5x5):
        """Test that field has expected symmetries for x-polarized wave."""
        p, sig, exc = gold_sphere_setup
        x, y, z = grid_5x5
        mf = MeshField(p, x, y, z)
        e, h = mf.field(sig)
        enorm = np.sqrt(np.sum(np.abs(e) ** 2, axis=-1))

        # For x-polarized wave on a sphere, the field should be
        # symmetric under z -> -z and antisymmetric should be small
        # enorm[i, j] ~= enorm[4-i, j]
        for i in range(2):
            for j in range(5):
                ratio = enorm[i, j] / enorm[4 - i, j]
                assert abs(ratio - 1.0) < 0.02, (
                    "z-symmetry broken at ({}, {}): ratio = {:.4f}".format(
                        i, j, ratio))

    def test_single_point_field(self, gold_sphere_setup):
        """Test field at a single far point."""
        p, sig, exc = gold_sphere_setup
        x = np.array([[30.0]])
        y = np.array([[0.0]])
        z = np.array([[0.0]])
        mf = MeshField(p, x, y, z)
        e, h = mf.field(sig)
        enorm = np.sqrt(np.sum(np.abs(e) ** 2, axis=-1))
        # For far point along x, the enhancement should be moderate
        assert enorm.ravel()[0] > 0.05
        assert enorm.ravel()[0] < 0.5

    def test_field_enhancement_near_surface(self, gold_sphere_setup):
        """Test that field enhancement is larger near the surface."""
        p, sig, exc = gold_sphere_setup
        # Two points: one near surface (12nm from center), one far (30nm)
        x_near = np.array([[12.0]])
        x_far = np.array([[30.0]])
        y = np.array([[0.0]])
        z = np.array([[0.0]])

        mf_near = MeshField(p, x_near, y, z)
        mf_far = MeshField(p, x_far, y, z)

        e_near, _ = mf_near.field(sig)
        e_far, _ = mf_far.field(sig)

        enorm_near = np.sqrt(np.sum(np.abs(e_near) ** 2, axis=-1))
        enorm_far = np.sqrt(np.sum(np.abs(e_far) ** 2, axis=-1))

        assert enorm_near.ravel()[0] > enorm_far.ravel()[0], (
            "Near-surface field ({:.4f}) should be larger than far field ({:.4f})".format(
                enorm_near.ravel()[0], enorm_far.ravel()[0]))


# ============================================================================
# Potential computation tests
# ============================================================================

class TestMeshFieldPotential(object):

    def test_potential_shape(self, gold_sphere_setup, grid_5x5):
        p, sig, exc = gold_sphere_setup
        x, y, z = grid_5x5
        mf = MeshField(p, x, y, z)
        phi = mf.potential(sig)
        assert phi.shape == (5, 5)


# ============================================================================
# ComPoint __call__ dtype tests (regression test for the complex casting bug)
# ============================================================================

class TestComPointDtype(object):

    def test_complex_values_preserved(self):
        """ComPoint.__call__ must preserve complex dtype."""
        pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        pt = Point(pos)
        cpt = ComPoint([EpsConst(1.0)], [pt], [1])
        vals = np.array([[1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j],
                         [7.0 + 8.0j, 9.0 + 10.0j, 11.0 + 12.0j]])
        result = cpt(vals)
        assert np.iscomplexobj(result), "Complex dtype must be preserved"
        np.testing.assert_array_equal(result, vals)

    def test_float_values_preserved(self):
        """ComPoint.__call__ with float input stays float."""
        pos = np.array([[1.0, 0.0, 0.0]])
        pt = Point(pos)
        cpt = ComPoint([EpsConst(1.0)], [pt], [1])
        vals = np.array([[1.0, 2.0, 3.0]])
        result = cpt(vals)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, vals)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
