import sys
import types

import numpy as np
import pytest

# bypass potential import issues with the full mnpbem package
if "mnpbem" not in sys.modules:
    _stub = types.ModuleType("mnpbem")
    _stub.__path__ = ["mnpbem"]
    sys.modules["mnpbem"] = _stub

from mnpbem.geometry.polygon import Polygon
from mnpbem.geometry.mesh2d import (
    inpoly, triarea, quality, circumcircle, fixmesh,
    smoothmesh, refine, connectivity, mesh2d,
)


# ============================================================================
# Polygon creation
# ============================================================================

class TestPolygonCreation(object):

    def test_regular_polygon_square(self) -> None:
        poly = Polygon(4)
        assert poly.pos.shape == (4, 2)
        assert poly.dir == 1
        assert poly.sym is None

    def test_regular_polygon_triangle(self) -> None:
        poly = Polygon(3)
        assert poly.pos.shape == (3, 2)
        # vertices should be on unit circle
        radii = np.sqrt(np.sum(poly.pos ** 2, axis = 1))
        np.testing.assert_allclose(radii, 1.0, atol = 1e-12)

    def test_regular_polygon_hexagon(self) -> None:
        poly = Polygon(6)
        assert poly.pos.shape == (6, 2)
        radii = np.sqrt(np.sum(poly.pos ** 2, axis = 1))
        np.testing.assert_allclose(radii, 1.0, atol = 1e-12)

    def test_polygon_from_vertices(self) -> None:
        verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype = float)
        poly = Polygon(verts)
        assert poly.pos.shape == (4, 2)
        np.testing.assert_allclose(poly.pos, verts)

    def test_polygon_with_size(self) -> None:
        poly = Polygon(4, size = [2.0, 3.0])
        s = poly.size_
        np.testing.assert_allclose(s[0], 2.0, atol = 1e-10)
        np.testing.assert_allclose(s[1], 3.0, atol = 1e-10)

    def test_polygon_with_scalar_size(self) -> None:
        poly = Polygon(4, size = 5.0)
        s = poly.size_
        np.testing.assert_allclose(s[0], 5.0, atol = 1e-10)
        np.testing.assert_allclose(s[1], 5.0, atol = 1e-10)

    def test_polygon_copy(self) -> None:
        poly = Polygon(4)
        poly2 = poly.copy()
        poly2.pos[0, 0] = 999.0
        assert poly.pos[0, 0] != 999.0


# ============================================================================
# Polygon operations
# ============================================================================

class TestPolygonOperations(object):

    def test_rotate_90(self) -> None:
        verts = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype = float)
        poly = Polygon(verts)
        poly.rot(90)
        # MATLAB rot matrix: [cos, sin; -sin, cos] applied as pos * rot
        # (1,0) @ rot(90) = (cos90, sin90) = (0, 1)
        # (0,1) @ rot(90) = (-sin90, cos90) = (-1, 0)
        expected = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]], dtype = float)
        np.testing.assert_allclose(poly.pos, expected, atol = 1e-10)

    def test_rotate_360(self) -> None:
        poly = Polygon(5)
        original = poly.pos.copy()
        poly.rot(360)
        np.testing.assert_allclose(poly.pos, original, atol = 1e-10)

    def test_scale_uniform(self) -> None:
        poly = Polygon(4)
        original = poly.pos.copy()
        poly.scale(2.0)
        np.testing.assert_allclose(poly.pos, original * 2.0, atol = 1e-12)

    def test_scale_nonuniform(self) -> None:
        poly = Polygon(4)
        original = poly.pos.copy()
        poly.scale([2.0, 3.0])
        np.testing.assert_allclose(poly.pos[:, 0], original[:, 0] * 2.0, atol = 1e-12)
        np.testing.assert_allclose(poly.pos[:, 1], original[:, 1] * 3.0, atol = 1e-12)

    def test_shift(self) -> None:
        poly = Polygon(4)
        original = poly.pos.copy()
        poly.shift([5.0, 3.0])
        np.testing.assert_allclose(poly.pos, original + np.array([5.0, 3.0]), atol = 1e-12)

    def test_flip_x(self) -> None:
        verts = np.array([[1, 2], [3, 4]], dtype = float)
        poly = Polygon(verts)
        poly.flip(0)
        np.testing.assert_allclose(poly.pos[:, 0], np.array([-1, -3]))
        np.testing.assert_allclose(poly.pos[:, 1], np.array([2, 4]))

    def test_flip_y(self) -> None:
        verts = np.array([[1, 2], [3, 4]], dtype = float)
        poly = Polygon(verts)
        poly.flip(1)
        np.testing.assert_allclose(poly.pos[:, 0], np.array([1, 3]))
        np.testing.assert_allclose(poly.pos[:, 1], np.array([-2, -4]))

    def test_round_adds_vertices(self) -> None:
        poly = Polygon(4, size = [2.0, 2.0])
        n_before = poly.n_verts
        poly.round_(rad = 0.2, nrad = 5)
        # rounding should add vertices at each corner
        assert poly.n_verts > n_before

    def test_union_single(self) -> None:
        poly = Polygon(4)
        pos, net = poly.union()
        assert pos.shape[0] == 4
        assert net.shape[0] == 4
        # edges should form a cycle
        assert net.shape[1] == 2

    def test_union_multiple(self) -> None:
        poly1 = Polygon(4)
        poly2 = Polygon(4)
        poly2.shift([3.0, 0.0])
        pos, net = poly1.union(poly2)
        assert pos.shape[0] == 8
        assert net.shape[0] == 8

    def test_dist(self) -> None:
        verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype = float)
        poly = Polygon(verts)
        pts = np.array([[0.5, 0.5], [2.0, 0.0]])
        dmin, imin = poly.dist(pts)
        # center point should be at distance 0.5 from nearest edge
        assert dmin[0] == pytest.approx(0.5, abs = 0.01)
        # point (2,0) should be at distance 1.0 from nearest edge
        assert dmin[1] == pytest.approx(1.0, abs = 0.01)


# ============================================================================
# Inpoly
# ============================================================================

class TestInpoly(object):

    def test_square_inside(self) -> None:
        node = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype = float)
        pts_inside = np.array([[0.5, 0.5], [0.1, 0.1], [0.9, 0.9]])
        cn, _ = inpoly(pts_inside, node)
        assert np.all(cn)

    def test_square_outside(self) -> None:
        node = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype = float)
        pts_outside = np.array([[2.0, 2.0], [-1.0, 0.5], [0.5, -1.0]])
        cn, _ = inpoly(pts_outside, node)
        assert not np.any(cn)

    def test_triangle(self) -> None:
        node = np.array([[0, 0], [4, 0], [2, 3]], dtype = float)
        pts = np.array([[2, 1], [0.5, 0.5], [1.5, 1.0], [5, 5]])
        cn, _ = inpoly(pts, node)
        # first 3 are inside the triangle, last is outside
        assert cn[0] and cn[1] and cn[2]
        assert not cn[3]

    def test_with_edge(self) -> None:
        node = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype = float)
        edge = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        pts = np.array([[0.5, 0.5]])
        cn, _ = inpoly(pts, node, edge)
        assert cn[0]

    def test_single_point(self) -> None:
        node = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype = float)
        cn, _ = inpoly(np.array([0.5, 0.5]), node)
        assert cn[0]


# ============================================================================
# Mesh2d utility functions
# ============================================================================

class TestMesh2dUtils(object):

    def test_triarea_positive(self) -> None:
        # CCW triangle
        p = np.array([[0, 0], [1, 0], [0, 1]], dtype = float)
        t = np.array([[0, 1, 2]])
        A = triarea(p, t)
        assert A[0] > 0

    def test_triarea_negative(self) -> None:
        # CW triangle
        p = np.array([[0, 0], [0, 1], [1, 0]], dtype = float)
        t = np.array([[0, 1, 2]])
        A = triarea(p, t)
        assert A[0] < 0

    def test_triarea_value(self) -> None:
        p = np.array([[0, 0], [2, 0], [0, 2]], dtype = float)
        t = np.array([[0, 1, 2]])
        A = triarea(p, t)
        np.testing.assert_allclose(A[0], 4.0, atol = 1e-12)

    def test_quality_equilateral(self) -> None:
        # equilateral triangle has q = 1 (approximately)
        h = np.sqrt(3) / 2
        p = np.array([[0, 0], [1, 0], [0.5, h]], dtype = float)
        t = np.array([[0, 1, 2]])
        q = quality(p, t)
        np.testing.assert_allclose(q[0], 1.0, atol = 1e-5)

    def test_quality_degenerate(self) -> None:
        # very flat triangle has low quality
        p = np.array([[0, 0], [1, 0], [0.5, 0.001]], dtype = float)
        t = np.array([[0, 1, 2]])
        q = quality(p, t)
        assert q[0] < 0.1

    def test_circumcircle(self) -> None:
        # right triangle at origin
        p = np.array([[0, 0], [1, 0], [0, 1]], dtype = float)
        t = np.array([[0, 1, 2]])
        cc = circumcircle(p, t)
        # circumcenter should be at (0.5, 0.5) with R^2 = 0.5
        np.testing.assert_allclose(cc[0, 0], 0.5, atol = 1e-10)
        np.testing.assert_allclose(cc[0, 1], 0.5, atol = 1e-10)
        np.testing.assert_allclose(cc[0, 2], 0.5, atol = 1e-10)

    def test_fixmesh_ccw(self) -> None:
        # CW triangle should be flipped to CCW
        p = np.array([[0, 0], [0, 1], [1, 0]], dtype = float)
        t = np.array([[0, 1, 2]])
        p_fix, t_fix, _, _ = fixmesh(p, t)
        A = triarea(p_fix, t_fix)
        assert np.all(A > 0)

    def test_fixmesh_removes_degenerate(self) -> None:
        # degenerate triangle (collinear points)
        p = np.array([[0, 0], [1, 0], [2, 0], [0, 1]], dtype = float)
        t = np.array([[0, 1, 2], [0, 1, 3]])
        p_fix, t_fix, _, _ = fixmesh(p, t)
        # degenerate triangle should be removed
        assert t_fix.shape[0] == 1

    def test_connectivity(self) -> None:
        p = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype = float)
        t = np.array([[0, 1, 2], [1, 3, 2]])
        e, te, e2t, bnd = connectivity(p, t)
        assert e.shape[0] == 5  # 5 unique edges in 2 triangles sharing 1 edge
        assert np.sum(bnd) == 4  # all corner nodes are boundary


# ============================================================================
# Mesh generation (polymesh2d)
# ============================================================================

class TestMesh2d(object):

    def test_mesh_square(self) -> None:
        node = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype = float)
        hdata = {'hmax': 0.5}
        p, t = mesh2d(node, hdata = hdata, options = {'output': False, 'maxit': 10})

        assert p.shape[0] > 4  # should have interior nodes
        assert t.shape[0] > 0  # should have triangles
        assert p.shape[1] == 2
        assert t.shape[1] == 3

        # all triangles should have positive area (CCW)
        A = triarea(p, t)
        assert np.all(A > 0), 'some triangles are CW'

    def test_mesh_triangle(self) -> None:
        node = np.array([[0, 0], [4, 0], [2, 3]], dtype = float)
        hdata = {'hmax': 1.0}
        p, t = mesh2d(node, hdata = hdata, options = {'output': False, 'maxit': 10})

        assert p.shape[0] >= 3
        assert t.shape[0] > 0

    def test_polygon_polymesh2d(self) -> None:
        poly = Polygon(4, size = [2.0, 2.0])
        hdata = {'hmax': 0.8}
        verts, faces = poly.polymesh2d(hdata = hdata)

        assert verts.shape[0] > 4
        assert faces.shape[0] > 0
        assert verts.shape[1] == 2
        assert faces.shape[1] == 3

    def test_polygon_hexagon_mesh(self) -> None:
        poly = Polygon(6, size = [2.0, 2.0])
        hdata = {'hmax': 0.8}
        verts, faces = poly.polymesh2d(hdata = hdata)

        assert verts.shape[0] > 6
        assert faces.shape[0] > 0

    def test_mesh_quality(self) -> None:
        poly = Polygon(4, size = [2.0, 2.0])
        hdata = {'hmax': 0.5}
        verts, faces = poly.polymesh2d(hdata = hdata)

        if faces.shape[0] > 0:
            q = quality(verts, faces)
            # mean quality should be reasonable
            assert np.mean(q) > 0.3, 'mean quality {:.3f} is too low'.format(np.mean(q))
            # no degenerate triangles
            assert np.min(q) > 0.0, 'degenerate triangle found'

    def test_mesh_with_edge(self) -> None:
        node = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype = float)
        edge = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        hdata = {'hmax': 0.8}
        p, t = mesh2d(node, edge, hdata = hdata, options = {'output': False, 'maxit': 10})

        assert p.shape[0] > 4
        assert t.shape[0] > 0


# ============================================================================
# Mesh refinement and smoothing
# ============================================================================

class TestMeshRefinement(object):

    def test_refine_uniform(self) -> None:
        # start with simple mesh
        p = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]], dtype = float)
        t = np.array([[0, 1, 2]])

        p_ref, t_ref = refine(p, t)
        # uniform refinement of 1 triangle -> 4 triangles
        assert t_ref.shape[0] == 4
        assert p_ref.shape[0] == 6  # 3 original + 3 midpoints

    def test_smoothmesh(self) -> None:
        # create a slightly irregular mesh and smooth it
        p = np.array([
            [0, 0], [1, 0], [2, 0],
            [0, 1], [1.1, 0.9], [2, 1],
            [0, 2], [1, 2], [2, 2]
        ], dtype = float)
        t = np.array([
            [0, 1, 4], [0, 4, 3],
            [1, 2, 5], [1, 5, 4],
            [3, 4, 7], [3, 7, 6],
            [4, 5, 8], [4, 8, 7]
        ])

        p_smooth, t_smooth = smoothmesh(p, t, maxit = 10)
        # interior node (index 4) should have moved toward (1, 1)
        # find the node closest to (1, 1)
        dists = np.sqrt(np.sum((p_smooth - np.array([1, 1])) ** 2, axis = 1))
        closest = np.argmin(dists)
        assert dists[closest] < 0.2, 'interior node did not smooth properly'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
