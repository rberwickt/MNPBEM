"""Tests for polygon utility methods: plot, norm, symmetry, interp1, union.

Reference geometries are simple (square, triangle, hexagon) so that the
expected normals / mirrored positions can be written out by hand.
"""
import os

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mnpbem.geometry.polygon import Polygon


TOL = 1e-10


def _square(size=1.0):
    # CCW, dir=1 means outward normal points OUT of polygon
    half = 0.5 * size
    verts = np.array([
        [-half, -half],
        [ half, -half],
        [ half,  half],
        [-half,  half]])
    return Polygon(verts)


def _triangle():
    verts = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [1.0, 1.5]])
    return Polygon(verts)


# ---------------------------------------------------------------- norm
def test_norm_alias_matches_compute_normals():
    poly = _square(1.0)
    a = poly.norm()
    b = poly.compute_normals()
    assert a.shape == b.shape
    assert np.allclose(a, b, atol=TOL)


def test_norm_square_points_outward():
    # For a unit square [-0.5,0.5]^2 with CCW winding and dir=1,
    # the outward normal at each vertex is the average of the two
    # adjacent edge normals.
    poly = _square(1.0)
    nv = poly.norm()
    assert nv.shape == (4, 2)

    # Each vertex normal must have unit length
    lens = np.sqrt(np.sum(nv ** 2, axis=1))
    assert np.allclose(lens, 1.0, atol=TOL)

    # Each vertex normal must point AWAY from origin (outward)
    pos = poly.pos
    dots = np.sum(nv * pos, axis=1)
    assert np.all(dots > 0), 'normals must point outward from origin'


def test_norm_triangle_unit_length():
    poly = _triangle()
    nv = poly.norm()
    lens = np.sqrt(np.sum(nv ** 2, axis=1))
    assert np.allclose(lens, 1.0, atol=TOL)


def test_norm_direction_flag_flips():
    poly = _square(1.0)
    n_out = poly.norm()
    poly.dir = -1
    n_in = poly.norm()
    # Flipping dir must negate the normals (equal magnitude, opposite sign)
    assert np.allclose(n_in, -n_out, atol=TOL)


# ---------------------------------------------------------------- plot
def test_plot_returns_axes_without_error():
    poly = _square(2.0)
    fig, ax = plt.subplots()
    out = poly.plot(ax=ax)
    assert out is ax
    # One Line2D object (the closed polygon outline)
    assert len(ax.lines) == 1
    # Line must have n+1 points (closed loop)
    xdata = ax.lines[0].get_xdata()
    assert len(xdata) == poly.pos.shape[0] + 1
    # First and last must coincide
    assert xdata[0] == xdata[-1]
    plt.close(fig)


def test_plot_with_normals_adds_quiver():
    poly = _triangle()
    fig, ax = plt.subplots()
    poly.plot(ax=ax, nvec=True, scale=0.5)
    # A quiver call adds one PatchCollection/Quiver artist
    from matplotlib.quiver import Quiver
    quivers = [a for a in ax.collections if isinstance(a, Quiver)]
    # Matplotlib may place the Quiver under ax.collections or
    # directly as a separate artist; at least one must exist.
    has_quiver = len(quivers) > 0 or any(
        isinstance(a, Quiver) for a in ax.get_children())
    assert has_quiver
    plt.close(fig)


# -------------------------------------------------------------- symmetry
def test_symmetry_x_returns_two_polygons():
    poly = _square(2.0)
    irr, full = poly.symmetry('x')
    assert isinstance(irr, Polygon)
    assert isinstance(full, Polygon)
    assert irr.sym == 'x'
    assert full.sym is None


def _poly_area(p):
    x = p[:, 0]
    y = p[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def test_symmetry_x_circle_area():
    # A regular n-gon (n=20) inscribed in unit circle reduced under 'x'
    # then rebuilt by mirroring should recover the original area.
    poly = Polygon(20)
    area_orig = _poly_area(poly.pos)
    irr, full = poly.symmetry('x')
    area_full = _poly_area(full.pos)
    assert np.isclose(area_full, area_orig, atol=1e-8), \
        'mirrored full area {} must match original {}'.format(
            area_full, area_orig)


def test_symmetry_none_returns_self_copy():
    poly = _triangle()
    irr, full = poly.symmetry(None)
    # Must be equivalent but independent (copy)
    assert np.allclose(irr.pos, poly.pos, atol=TOL)
    assert np.allclose(full.pos, poly.pos, atol=TOL)


def test_symmetry_irreducible_positions_are_nonnegative_x():
    # After 'x' symmetry, irreducible part must have all x >= 0
    verts = np.array([
        [-1.0, -1.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0],
        [-1.0,  1.0]])
    poly = Polygon(verts)
    irr, _ = poly.symmetry('x')
    assert np.all(irr.pos[:, 0] >= -1e-8), \
        'irreducible x-sym positions must satisfy x >= 0'


# -------------------------------------------------------------- interp1
def test_interp1_picks_boundary_points():
    poly = _square(2.0)
    # candidates: two edge midpoints + one off-boundary point
    pts = np.array([
        [ 1.0,  0.0],   # on right edge
        [ 0.0,  1.0],   # on top edge
        [ 0.3,  0.3]])  # interior
    new_poly = poly.copy().interp1(pts)
    # Only two points lie on the boundary
    assert new_poly.pos.shape[0] == 2
    # Interior point must be dropped
    for p in new_poly.pos:
        assert np.min(np.sum((pts - p) ** 2, axis=1)) < TOL
        assert not np.allclose(p, [0.3, 0.3], atol=TOL)


def test_interp1_empty_when_no_match():
    poly = _square(1.0)
    pts = np.array([
        [10.0, 10.0],
        [-5.0, 7.0]])
    new_poly = poly.copy().interp1(pts)
    # No boundary matches -> polygon must be unchanged
    assert np.allclose(new_poly.pos, poly.pos, atol=TOL)


# -------------------------------------------------------------- union
def test_union_concatenates_positions():
    a = _square(1.0)
    b = _square(2.0)
    upos, unet = a.union(b)
    assert upos.shape[0] == a.n_verts + b.n_verts
    assert unet.shape == (upos.shape[0], 2)
    # Each loop must have its own closed ring
    # Second polygon edges use indices >= 4 (first polygon's vertex count)
    assert np.all(unet[a.n_verts:, 0] >= a.n_verts)


def test_union_single_polygon_same_as_self_edges():
    a = _square(1.0)
    upos, unet = a.union()
    assert np.allclose(upos, a.pos, atol=TOL)
    expected = np.column_stack([np.arange(4), np.roll(np.arange(4), -1)])
    assert np.array_equal(unet, expected)


# -------------------------------------------------------------- integration
def test_plot_norm_symmetry_pipeline():
    # Sanity: a full MATLAB-style workflow runs end-to-end without error
    poly = _square(2.0)
    irr, full = poly.symmetry('xy')
    nv = full.norm()
    assert nv.shape == (full.pos.shape[0], 2)

    fig, ax = plt.subplots()
    full.plot(ax=ax, nvec=True)
    plt.close(fig)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
