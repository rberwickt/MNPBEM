import sys
import os
import types

import numpy as np
import pytest
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure mnpbem can be imported even if top-level __init__ has issues
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mnpbem.geometry.particle import Particle
from mnpbem.geometry.mesh_generators import (
    fvgrid,
    trispheresegment,
    trirod,
    tricube,
    tritorus,
    trispherescale,
)
from mnpbem.geometry.edgeprofile import EdgeProfile


# ============================================================================
# Helper: validate a Particle object has expected basic properties
# ============================================================================

def _validate_particle(p: Particle,
        label: str = 'particle',
        min_verts: int = 4,
        min_faces: int = 2) -> None:

    assert p is not None, '[error] {} is None'.format(label)
    assert p.nverts >= min_verts, '[error] {} has {} verts, expected >= {}'.format(label, p.nverts, min_verts)
    assert p.nfaces >= min_faces, '[error] {} has {} faces, expected >= {}'.format(label, p.nfaces, min_faces)

    # No NaN in vertex positions
    assert not np.any(np.isnan(p.verts)), '[error] {} has NaN in verts'.format(label)

    # Faces should reference valid vertex indices
    faces_int = p.faces[:, :3].astype(int)
    assert np.all(faces_int >= 0), '[error] {} has negative face indices'.format(label)
    assert np.all(faces_int < p.nverts), '[error] {} has face indices >= nverts'.format(label)

    # Area should be positive
    total_area = np.sum(p.area)
    assert total_area > 0, '[error] {} has non-positive total area'.format(label)


# ============================================================================
# Tests: fvgrid
# ============================================================================

class TestFvgrid(object):

    def test_basic_grid(self) -> None:
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        verts, faces = fvgrid(x, y)
        assert verts is not None
        assert faces is not None
        assert verts.shape[1] == 3
        assert verts.shape[0] > 0
        assert faces.shape[0] > 0

    def test_2d_input(self) -> None:
        x, y = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
        verts, faces = fvgrid(x, y)
        assert verts.shape[0] > 0
        assert faces.shape[0] > 0


# ============================================================================
# Tests: trispheresegment
# ============================================================================

class TestTrispheresegment(object):

    def test_hemisphere(self) -> None:
        # Create upper hemisphere
        phi = np.linspace(0, 2 * np.pi, 15)
        theta = np.linspace(0, np.pi / 2, 10)
        p = trispheresegment(phi, theta, diameter = 2.0)
        _validate_particle(p, label = 'hemisphere')

        # All z should be >= 0 (upper hemisphere)
        assert np.all(p.verts[:, 2] >= -1e-10), '[error] hemisphere has negative z'

    def test_full_sphere_segment(self) -> None:
        phi = np.linspace(0, 2 * np.pi, 12)
        theta = np.linspace(0, np.pi, 12)
        p = trispheresegment(phi, theta, diameter = 1.0)
        _validate_particle(p, label = 'full sphere segment')

    def test_diameter_scaling(self) -> None:
        phi = np.linspace(0, 2 * np.pi, 10)
        theta = np.linspace(0, np.pi / 2, 8)

        p1 = trispheresegment(phi, theta, diameter = 1.0)
        p2 = trispheresegment(phi, theta, diameter = 2.0)

        max_extent_1 = np.max(np.linalg.norm(p1.verts, axis = 1))
        max_extent_2 = np.max(np.linalg.norm(p2.verts, axis = 1))

        # p2 should be roughly 2x p1
        ratio = max_extent_2 / max_extent_1
        assert abs(ratio - 2.0) < 0.2, '[error] diameter scaling ratio {} != 2.0'.format(ratio)


# ============================================================================
# Tests: trirod
# ============================================================================

class TestTrirod(object):

    def test_basic_rod(self) -> None:
        p = trirod(1.0, 3.0, n = [10, 8, 8])
        _validate_particle(p, label = 'rod')

    def test_rod_dimensions(self) -> None:
        diameter = 2.0
        height = 6.0
        p = trirod(diameter, height, n = [12, 10, 10])

        # Check that the rod extends approximately to +/- height/2 along z
        z_min = np.min(p.verts[:, 2])
        z_max = np.max(p.verts[:, 2])
        assert abs(z_max - height / 2.0) < 0.5, '[error] rod z_max {} != {}'.format(z_max, height / 2.0)
        assert abs(z_min + height / 2.0) < 0.5, '[error] rod z_min {} != {}'.format(z_min, -height / 2.0)

        # Check radial extent (approximately diameter/2)
        r_max = np.max(np.sqrt(p.verts[:, 0] ** 2 + p.verts[:, 1] ** 2))
        assert abs(r_max - diameter / 2.0) < 0.3, '[error] rod r_max {} != {}'.format(r_max, diameter / 2.0)

    def test_rod_symmetry(self) -> None:
        p = trirod(1.0, 3.0, n = [10, 8, 8])
        # Rod should be symmetric about z=0
        z_center = np.mean(p.verts[:, 2])
        assert abs(z_center) < 0.1, '[error] rod center {} not near 0'.format(z_center)


# ============================================================================
# Tests: tricube
# ============================================================================

class TestTricube(object):

    def test_basic_cube(self) -> None:
        p = tricube(8, length = 1.0, e = 0.25)
        _validate_particle(p, label = 'cube')

    def test_cube_extent(self) -> None:
        length = 2.0
        p = tricube(8, length = length, e = 0.25)

        # Cube should extend to approximately +/- length/2
        for dim in range(3):
            vmin = np.min(p.verts[:, dim])
            vmax = np.max(p.verts[:, dim])
            assert abs(vmax - length / 2.0) < 0.3, '[error] cube dim {} max {} != {}'.format(dim, vmax, length / 2.0)
            assert abs(vmin + length / 2.0) < 0.3, '[error] cube dim {} min {} != {}'.format(dim, vmin, -length / 2.0)

    def test_cube_anisotropic_scale(self) -> None:
        p = tricube(8, length = [1.0, 2.0, 3.0], e = 0.25)
        _validate_particle(p, label = 'aniso cube')

        # Check that extent in each dimension matches
        for dim, expected_len in enumerate([1.0, 2.0, 3.0]):
            vmax = np.max(p.verts[:, dim])
            assert abs(vmax - expected_len / 2.0) < 0.3, \
                '[error] cube dim {} max {} != {}'.format(dim, vmax, expected_len / 2.0)


# ============================================================================
# Tests: tritorus
# ============================================================================

class TestTritorus(object):

    def test_basic_torus(self) -> None:
        p = tritorus(2.0, 0.5, n = [15, 10])
        _validate_particle(p, label = 'torus')

    def test_torus_dimensions(self) -> None:
        diameter = 4.0
        rad = 1.0
        p = tritorus(diameter, rad, n = [20, 15])

        # Major radius = diameter/2, tube radius = rad
        # Max radial distance should be diameter/2 + rad
        r_xy = np.sqrt(p.verts[:, 0] ** 2 + p.verts[:, 1] ** 2)
        expected_outer = diameter / 2.0 + rad
        actual_outer = np.max(r_xy)
        assert abs(actual_outer - expected_outer) < 0.5, \
            '[error] torus outer radius {} != {}'.format(actual_outer, expected_outer)

        # Min radial distance should be diameter/2 - rad
        expected_inner = diameter / 2.0 - rad
        actual_inner = np.min(r_xy)
        assert abs(actual_inner - expected_inner) < 0.5, \
            '[error] torus inner radius {} != {}'.format(actual_inner, expected_inner)

    def test_torus_z_extent(self) -> None:
        diameter = 2.0
        rad = 0.5
        p = tritorus(diameter, rad, n = [15, 10])

        # z extent should be about +/- rad
        z_max = np.max(p.verts[:, 2])
        z_min = np.min(p.verts[:, 2])
        assert abs(z_max - rad) < 0.2, '[error] torus z_max {} != {}'.format(z_max, rad)
        assert abs(z_min + rad) < 0.2, '[error] torus z_min {} != {}'.format(z_min, -rad)


# ============================================================================
# Tests: trispherescale
# ============================================================================

class TestTrispherescale(object):

    def test_uniform_scale(self) -> None:
        # Start with a simple sphere
        phi = np.linspace(0, 2 * np.pi, 12)
        theta = np.linspace(0, np.pi, 10)
        p = trispheresegment(phi, theta, diameter = 1.0)

        # Uniform scale by 2
        scale = 2.0 * np.ones(p.nverts)
        p_scaled = trispherescale(p, scale)
        _validate_particle(p_scaled, label = 'scaled sphere')

    def test_nonuniform_scale(self) -> None:
        phi = np.linspace(0, 2 * np.pi, 12)
        theta = np.linspace(0, np.pi, 10)
        p = trispheresegment(phi, theta, diameter = 1.0)

        # Non-uniform scale
        scale = 1.0 + 0.5 * np.sin(np.arctan2(p.verts[:, 1], p.verts[:, 0]))
        p_scaled = trispherescale(p, scale)
        _validate_particle(p_scaled, label = 'deformed sphere')


# ============================================================================
# Tests: EdgeProfile
# ============================================================================

class TestEdgeProfile(object):

    def test_empty_profile(self) -> None:
        ep = EdgeProfile()
        assert ep.pos is None
        assert ep.z is None

    def test_basic_profile(self) -> None:
        ep = EdgeProfile(10.0, 7)
        assert ep.pos is not None
        assert ep.z is not None
        assert ep.pos.shape[1] == 2
        assert len(ep.z) == 7

    def test_properties(self) -> None:
        ep = EdgeProfile(10.0, 7)
        assert ep.zmin < ep.zmax
        assert ep.zmax > 0
        assert ep.zmin < 0

    def test_mode_11(self) -> None:
        ep = EdgeProfile(10.0, 7, mode = '11')
        assert ep.pos is not None
        # Mode '11' has NaN entries at boundaries
        assert np.any(np.isnan(ep.pos[:, 0]))

    def test_hshift(self) -> None:
        ep = EdgeProfile(10.0, 11)
        z_vals = np.linspace(ep.zmin + 0.1, ep.zmax - 0.1, 5)
        shifts = ep.hshift(z_vals)
        assert shifts.shape == z_vals.shape
        assert not np.any(np.isnan(shifts))

    def test_shift_center(self) -> None:
        ep = EdgeProfile(10.0, 7, center = 5.0)
        # Center should shift the profile
        z_mid = 0.5 * (ep.zmin + ep.zmax)
        assert abs(z_mid - 5.0) < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
