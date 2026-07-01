import os
import sys

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mnpbem.geometry.polygon import Polygon
from mnpbem.geometry.edgeprofile import EdgeProfile
from mnpbem.geometry.polygon3 import Polygon3
from mnpbem.geometry.particle import Particle
from mnpbem.geometry.mesh_generators import tripolygon


# ================================================================
# Polygon3 basic tests
# ================================================================

class TestPolygon3Init(object):

    def test_basic_init(self):
        poly = Polygon(4, size = 100)
        p3 = Polygon3(poly, z = 0.0)
        assert p3.z == 0.0
        assert p3.poly is not poly  # should be a copy
        assert p3.edge is not None

    def test_init_with_edge(self):
        poly = Polygon(4, size = 100)
        edge = EdgeProfile(20, 7)
        p3 = Polygon3(poly, z = 10.0, edge = edge)
        assert p3.z == 10.0
        assert p3.edge is edge

    def test_copy(self):
        poly = Polygon(4, size = 100)
        p3 = Polygon3(poly, z = 5.0)
        p3_copy = p3.copy()
        p3_copy.z = 10.0
        assert p3.z == 5.0
        assert p3_copy.z == 10.0

    def test_set(self):
        poly = Polygon(4, size = 100)
        p3 = Polygon3(poly, z = 0.0)
        p3.set(z = 5.0)
        assert p3.z == 5.0

    def test_shift(self):
        poly = Polygon(4, size = 100)
        p3 = Polygon3(poly, z = 0.0)
        original_pos = p3.poly.pos.copy()
        p3.shift(np.array([10.0, 20.0, 5.0]))
        assert p3.z == 5.0
        np.testing.assert_allclose(p3.poly.pos[:, 0], original_pos[:, 0] + 10.0, atol = 1e-10)
        np.testing.assert_allclose(p3.poly.pos[:, 1], original_pos[:, 1] + 20.0, atol = 1e-10)


# ================================================================
# Plate tests
# ================================================================

class TestPolygon3Plate(object):

    def test_plate_square(self):
        poly = Polygon(4, size = 100)
        p3 = Polygon3(poly, z = 0.0)
        plate, _ = p3.plate(dir = 1)

        assert isinstance(plate, Particle)
        assert plate.nverts > 0
        assert plate.nfaces > 0
        # all vertices should be near z = 0
        np.testing.assert_allclose(plate.verts[:, 2], 0.0, atol = 1.0)
        # no NaN in vertices
        assert not np.any(np.isnan(plate.verts))
        # all face indices in range
        faces_valid = plate.faces[:, :3].astype(int)
        assert np.all(faces_valid >= 0)
        assert np.all(faces_valid < plate.nverts)
        # positive area
        assert np.all(plate.area > 0)

    def test_plate_circle(self):
        poly = Polygon(20, size = 80)
        p3 = Polygon3(poly, z = 5.0)
        plate, _ = p3.plate(dir = -1)

        assert isinstance(plate, Particle)
        assert plate.nverts > 0
        assert plate.nfaces > 0
        assert not np.any(np.isnan(plate.verts))
        assert np.all(plate.area > 0)

    def test_plate_normal_direction(self):
        poly = Polygon(4, size = 100)
        p3_up = Polygon3(poly, z = 0.0)
        plate_up, _ = p3_up.plate(dir = 1)

        p3_down = Polygon3(poly, z = 0.0)
        plate_down, _ = p3_down.plate(dir = -1)

        # up plate should have positive z-normals on average
        assert np.sum(plate_up.nvec[:, 2]) > 0
        # down plate should have negative z-normals on average
        assert np.sum(plate_down.nvec[:, 2]) < 0


# ================================================================
# VRibbon tests
# ================================================================

class TestPolygon3VRibbon(object):

    def test_vribbon_square(self):
        poly = Polygon(4, size = 100)
        edge = EdgeProfile(20, 7)
        p3 = Polygon3(poly, z = edge.zmax, edge = edge)

        ribbon, up, lo = p3.vribbon()

        assert isinstance(ribbon, Particle)
        assert ribbon.nverts > 0
        assert ribbon.nfaces > 0
        assert not np.any(np.isnan(ribbon.verts))
        assert np.all(ribbon.area > 0)

        # upper and lower polygon3 should have different z-values
        assert isinstance(up, Polygon3)
        assert isinstance(lo, Polygon3)
        assert up.z > lo.z

    def test_vribbon_circle(self):
        poly = Polygon(20, size = 80)
        edge = EdgeProfile(30, 7)
        p3 = Polygon3(poly, z = edge.zmax, edge = edge)

        ribbon, up, lo = p3.vribbon()

        assert isinstance(ribbon, Particle)
        assert ribbon.nverts > 0
        assert not np.any(np.isnan(ribbon.verts))

    def test_vribbon_with_explicit_z(self):
        poly = Polygon(4, size = 100)
        edge = EdgeProfile(20, 7)
        p3 = Polygon3(poly, z = 0.0, edge = edge)

        z_vals = np.linspace(-10, 10, 5)
        ribbon, up, lo = p3.vribbon(z = z_vals)

        assert isinstance(ribbon, Particle)
        assert ribbon.nverts > 0


# ================================================================
# HRibbon tests
# ================================================================

class TestPolygon3HRibbon(object):

    def test_hribbon_square(self):
        poly = Polygon(4, size = 100)
        p3 = Polygon3(poly, z = 0.0)

        d_vals = np.linspace(0, -5, 4)
        ribbon, inner, outer = p3.hribbon(d_vals, dir = 1)

        assert isinstance(ribbon, Particle)
        assert ribbon.nverts > 0
        assert ribbon.nfaces > 0
        assert not np.any(np.isnan(ribbon.verts))
        assert np.all(ribbon.area > 0)

        assert isinstance(inner, Polygon3)
        assert isinstance(outer, Polygon3)

    def test_hribbon_circle(self):
        poly = Polygon(20, size = 80)
        p3 = Polygon3(poly, z = 5.0)

        d_vals = np.linspace(0, -3, 3)
        ribbon, inner, outer = p3.hribbon(d_vals, dir = -1)

        assert isinstance(ribbon, Particle)
        assert ribbon.nverts > 0
        assert not np.any(np.isnan(ribbon.verts))


# ================================================================
# tripolygon tests
# ================================================================

class TestTripolygon(object):

    def test_tripolygon_square_rounded(self):
        poly = Polygon(4, size = 100)
        edge = EdgeProfile(20, 7)
        p = tripolygon(poly, edge)

        assert isinstance(p, Particle)
        assert p.nverts > 0
        assert p.nfaces > 0
        assert not np.any(np.isnan(p.verts))
        # faces in range
        ind3 = np.where(np.isnan(p.faces[:, 3]))[0]
        if len(ind3) > 0:
            faces3 = p.faces[ind3, :3].astype(int)
            assert np.all(faces3 >= 0)
            assert np.all(faces3 < p.nverts)
        assert np.all(p.area > 0)

    def test_tripolygon_circle_rounded(self):
        poly = Polygon(20, size = 80)
        edge = EdgeProfile(30, 7)
        p = tripolygon(poly, edge)

        assert isinstance(p, Particle)
        assert p.nverts > 0
        assert p.nfaces > 0
        assert not np.any(np.isnan(p.verts))
        assert np.all(p.area > 0)

    def test_tripolygon_sharp_lower(self):
        poly = Polygon(4, size = 100)
        edge = EdgeProfile(20, 7, mode = '10')
        p = tripolygon(poly, edge)

        assert isinstance(p, Particle)
        assert p.nverts > 0
        assert p.nfaces > 0
        assert not np.any(np.isnan(p.verts))
        assert np.all(p.area > 0)

    def test_tripolygon_sharp_upper(self):
        poly = Polygon(4, size = 100)
        edge = EdgeProfile(20, 7, mode = '01')
        p = tripolygon(poly, edge)

        assert isinstance(p, Particle)
        assert p.nverts > 0
        assert p.nfaces > 0
        assert not np.any(np.isnan(p.verts))
        assert np.all(p.area > 0)

    def test_tripolygon_both_sharp(self):
        poly = Polygon(4, size = 100)
        edge = EdgeProfile(20, 7, mode = '11')
        p = tripolygon(poly, edge)

        assert isinstance(p, Particle)
        assert p.nverts > 0
        assert p.nfaces > 0
        assert not np.any(np.isnan(p.verts))
        assert np.all(p.area > 0)

    def test_tripolygon_hexagon(self):
        poly = Polygon(6, size = 60)
        edge = EdgeProfile(15, 7)
        p = tripolygon(poly, edge)

        assert isinstance(p, Particle)
        assert p.nverts > 0
        assert not np.any(np.isnan(p.verts))
        assert np.all(p.area > 0)

    def test_tripolygon_geometry_bounds(self):
        # the generated particle should have z-extent matching the edge profile
        poly = Polygon(4, size = 100)
        edge = EdgeProfile(20, 7)
        p = tripolygon(poly, edge)

        z_min = np.min(p.verts[:, 2])
        z_max = np.max(p.verts[:, 2])

        # z-range should be approximately equal to height
        z_range = z_max - z_min
        assert z_range > 0
        assert z_range <= 25  # some tolerance above 20

    def test_tripolygon_valid_faces(self):
        poly = Polygon(8, size = 50)
        edge = EdgeProfile(10, 5)
        p = tripolygon(poly, edge)

        # all face indices should be valid
        ind3, ind4 = p.index34()
        if len(ind3) > 0:
            f3 = p.faces[ind3, :3].astype(int)
            assert np.all(f3 >= 0), 'negative face index'
            assert np.all(f3 < p.nverts), 'face index out of range'
        if len(ind4) > 0:
            f4 = p.faces[ind4, :4].astype(int)
            assert np.all(f4 >= 0), 'negative face index'
            assert np.all(f4 < p.nverts), 'face index out of range'


# ================================================================
# Polygon midpoints tests
# ================================================================

class TestPolygonMidpoints(object):

    def test_midpoints_square(self):
        poly = Polygon(4, size = 100)
        n_orig = poly.n_verts
        poly.midpoints()
        # midpoints doubles the number of vertices
        assert poly.n_verts == 2 * n_orig

    def test_midpoints_preserves_original(self):
        poly = Polygon(4, size = 100)
        original_pos = poly.pos.copy()
        poly.midpoints()
        # even-indexed positions should be close to originals
        np.testing.assert_allclose(poly.pos[0::2], original_pos, atol = 1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
