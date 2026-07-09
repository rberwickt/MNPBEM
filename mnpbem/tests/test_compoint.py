import sys
import os
import copy

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mnpbem.geometry.compoint import Point, ComPoint


# ---------------------------------------------------------------------------
# Mock dielectric function
# ---------------------------------------------------------------------------

class MockEps(object):

    def __init__(self, value = 1.0 + 0.0j):
        self._value = value

    def __call__(self, enei):
        return complex(self._value), 2 * np.pi / enei * np.sqrt(complex(self._value))


class MockEpsTuple(object):

    def __init__(self, value = 1.0 + 0.0j):
        self._value = value

    def __call__(self, enei):
        return (complex(self._value), 2 * np.pi / enei * np.sqrt(complex(self._value)))


# ---------------------------------------------------------------------------
# Point tests
# ---------------------------------------------------------------------------

class TestPoint(object):

    def test_construction_basic(self):
        pos = np.array([[1.0, 2.0, 3.0]])
        pt = Point(pos)
        assert pt.n == 1
        assert pt.pos.shape == (1, 3)
        np.testing.assert_allclose(pt.pos[0], [1.0, 2.0, 3.0])

    def test_construction_multiple(self):
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype = np.float64)
        pt = Point(pos)
        assert pt.n == 3
        assert pt.nfaces == 3

    def test_default_nvec_zeros(self):
        pos = np.array([[1, 2, 3], [4, 5, 6]], dtype = np.float64)
        pt = Point(pos)
        np.testing.assert_array_equal(pt.nvec, np.zeros((2, 3)))

    def test_default_area_ones(self):
        pos = np.array([[0, 0, 0], [1, 1, 1]], dtype = np.float64)
        pt = Point(pos)
        np.testing.assert_array_equal(pt.area, np.ones(2))

    def test_custom_nvec(self):
        pos = np.array([[0, 0, 0]], dtype = np.float64)
        nvec = np.array([[0, 0, 1]], dtype = np.float64)
        pt = Point(pos, nvec = nvec)
        np.testing.assert_array_equal(pt.nvec, [[0, 0, 1]])

    def test_custom_area(self):
        pos = np.array([[0, 0, 0], [1, 0, 0]], dtype = np.float64)
        area = np.array([2.5, 3.0])
        pt = Point(pos, area = area)
        np.testing.assert_allclose(pt.area, [2.5, 3.0])

    def test_add_two_points(self):
        pt1 = Point(np.array([[0, 0, 0]], dtype = np.float64))
        pt2 = Point(np.array([[1, 1, 1]], dtype = np.float64))
        combined = pt1 + pt2
        assert combined.n == 2
        np.testing.assert_allclose(combined.pos[0], [0, 0, 0])
        np.testing.assert_allclose(combined.pos[1], [1, 1, 1])

    def test_select_by_index(self):
        pos = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype = np.float64)
        pt = Point(pos)
        selected = pt.select(index = np.array([0, 2]))
        assert selected.n == 2
        np.testing.assert_allclose(selected.pos[0], [0, 0, 0])
        np.testing.assert_allclose(selected.pos[1], [2, 0, 0])

    def test_select_by_carfun(self):
        pos = np.array([[-1, 0, 0], [1, 0, 0], [2, 0, 0]], dtype = np.float64)
        pt = Point(pos)
        selected = pt.select(carfun = lambda x, y, z: x > 0)
        assert selected.n == 2

    def test_repr(self):
        pt = Point(np.array([[0, 0, 0]]))
        s = repr(pt)
        assert 'Point' in s
        assert '1' in s


# ---------------------------------------------------------------------------
# ComPoint tests
# ---------------------------------------------------------------------------

class TestComPoint(object):

    def test_construction_from_pos_list(self):
        eps = [MockEps(1.0)]
        pos_list = [np.array([[0, 0, 10], [0, 0, 20]])]
        inout = [1]
        pt = ComPoint(eps, pos_list, inout)

        assert pt.n == 2
        assert pt.pos.shape == (2, 3)
        assert len(pt.eps) == 1

    def test_construction_from_point_list(self):
        eps = [MockEps(1.0), MockEps(-11.4 + 1.0j)]
        p1 = Point(np.array([[0, 0, 20]]))
        p2 = Point(np.array([[0, 0, -5]]))
        inout = [1, 2]
        pt = ComPoint(eps, [p1, p2], inout)

        assert pt.n == 2
        assert pt.np == 2

    def test_pos_attribute(self):
        eps = [MockEps(1.0)]
        positions = np.array([[1, 2, 3], [4, 5, 6]], dtype = np.float64)
        pt = ComPoint(eps, [positions], [1])
        np.testing.assert_allclose(pt.pos, positions)

    def test_nvec_default_zeros(self):
        eps = [MockEps(1.0)]
        pt = ComPoint(eps, [np.array([[0, 0, 0]])], [1])
        np.testing.assert_array_equal(pt.nvec, np.zeros((1, 3)))

    def test_area_default_ones(self):
        eps = [MockEps(1.0)]
        pt = ComPoint(eps, [np.array([[0, 0, 0]])], [1])
        np.testing.assert_array_equal(pt.area, np.ones(1))

    def test_index_single_group(self):
        eps = [MockEps(1.0)]
        pt = ComPoint(eps, [np.array([[0, 0, 0], [1, 0, 0]])], [1])
        idx = pt.index
        assert len(idx) == 1
        np.testing.assert_array_equal(idx[0], [0, 1])

    def test_index_multiple_groups(self):
        eps = [MockEps(1.0), MockEps(2.0)]
        pt = ComPoint(eps, [np.array([[0, 0, 0]]), np.array([[1, 0, 0]])], [1, 2])
        idx = pt.index
        assert len(idx) == 2
        np.testing.assert_array_equal(idx[0], [0])
        np.testing.assert_array_equal(idx[1], [1])

    def test_inout_array(self):
        eps = [MockEps(1.0), MockEps(2.0)]
        pt = ComPoint(eps, [np.array([[0, 0, 0]]), np.array([[1, 0, 0]])], [1, 2])
        np.testing.assert_array_equal(pt.inout, [1, 2])

    def test_eps1_single_medium(self):
        eps = [MockEps(1.0)]
        pt = ComPoint(eps, [np.array([[0, 0, 0], [1, 0, 0]])], [1])
        vals = pt.eps1(500.0)
        assert vals.shape == (2,)
        np.testing.assert_allclose(vals, [1.0 + 0j, 1.0 + 0j])

    def test_eps1_multiple_media(self):
        eps_out = MockEps(1.0)
        eps_in = MockEps(-11.4 + 1.0j)
        eps = [eps_out, eps_in]
        pt = ComPoint(eps,
                       [np.array([[0, 0, 20]]), np.array([[0, 0, -5]])],
                       [1, 2])
        vals = pt.eps1(500.0)
        assert vals.shape == (2,)
        np.testing.assert_allclose(vals[0], 1.0 + 0j)
        np.testing.assert_allclose(vals[1], -11.4 + 1.0j)

    def test_eps1_tuple_return(self):
        eps = [MockEpsTuple(2.5 + 0.1j)]
        pt = ComPoint(eps, [np.array([[0, 0, 0]])], [1])
        vals = pt.eps1(500.0)
        np.testing.assert_allclose(vals[0], 2.5 + 0.1j)

    def test_closedparticle_returns_none(self):
        eps = [MockEps(1.0)]
        pt = ComPoint(eps, [np.array([[0, 0, 0]])], [1])
        p, d, loc = pt.closedparticle(1)
        assert p is None
        assert d is None
        assert loc is None

    def test_flip_x(self):
        eps = [MockEps(1.0)]
        pos = np.array([[5, 3, 7]], dtype = np.float64)
        pt = ComPoint(eps, [pos], [1])
        flipped = pt.flip(1)
        np.testing.assert_allclose(flipped.pos[0], [-5, 3, 7])
        # Original unchanged
        np.testing.assert_allclose(pt.pos[0], [5, 3, 7])

    def test_flip_y(self):
        eps = [MockEps(1.0)]
        pos = np.array([[5, 3, 7]], dtype = np.float64)
        pt = ComPoint(eps, [pos], [1])
        flipped = pt.flip(2)
        np.testing.assert_allclose(flipped.pos[0], [5, -3, 7])

    def test_flip_xy(self):
        eps = [MockEps(1.0)]
        pos = np.array([[5, 3, 7]], dtype = np.float64)
        pt = ComPoint(eps, [pos], [1])
        flipped = pt.flip([1, 2])
        np.testing.assert_allclose(flipped.pos[0], [-5, -3, 7])

    def test_nfaces_equals_n(self):
        eps = [MockEps(1.0)]
        pt = ComPoint(eps, [np.array([[0, 0, 0], [1, 1, 1]])], [1])
        assert pt.nfaces == pt.n
        assert pt.nfaces == 2

    def test_repr(self):
        eps = [MockEps(1.0)]
        pt = ComPoint(eps, [np.array([[0, 0, 0]])], [1])
        s = repr(pt)
        assert 'ComPoint' in s

    def test_call_expand_values(self):
        eps = [MockEps(1.0)]
        pos_list = [np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])]
        pt = ComPoint(eps, pos_list, [1])

        valpt = np.array([10.0, 20.0, 30.0])
        result = pt(valpt)
        assert result.shape[0] == 3
        np.testing.assert_allclose(result, [10.0, 20.0, 30.0])

    def test_multiple_points_n(self):
        eps = [MockEps(1.0)]
        pt = ComPoint(eps, [np.array([[0, 0, i] for i in range(5)])], [1])
        assert pt.n == 5


# ---------------------------------------------------------------------------
# ComPoint interface compatibility with DipoleStat
# ---------------------------------------------------------------------------

class TestComPointDipoleInterface(object):
    """Verify ComPoint provides the interface expected by DipoleStat."""

    def _make_compoint(self):
        eps = [MockEps(1.0)]
        pos = np.array([[0, 0, 20]], dtype = np.float64)
        return ComPoint(eps, [pos], [1])

    def test_has_pos(self):
        pt = self._make_compoint()
        assert hasattr(pt, 'pos')
        assert pt.pos.shape == (1, 3)

    def test_has_n(self):
        pt = self._make_compoint()
        assert hasattr(pt, 'n')
        assert pt.n == 1

    def test_has_eps(self):
        pt = self._make_compoint()
        assert hasattr(pt, 'eps')
        assert isinstance(pt.eps, list)

    def test_has_eps1(self):
        pt = self._make_compoint()
        assert callable(pt.eps1)
        vals = pt.eps1(500.0)
        assert isinstance(vals, np.ndarray)
        assert vals.shape == (1,)

    def test_has_index(self):
        pt = self._make_compoint()
        idx = pt.index
        assert isinstance(idx, list)
        assert len(idx) == 1

    def test_has_inout(self):
        pt = self._make_compoint()
        assert hasattr(pt, 'inout')
        assert isinstance(pt.inout, np.ndarray)

    def test_has_flip(self):
        pt = self._make_compoint()
        assert callable(pt.flip)

    def test_has_closedparticle(self):
        pt = self._make_compoint()
        assert callable(pt.closedparticle)

    def test_index_boolean_mask(self):
        """DipoleStat.farfield uses: pt.index[pt.inout == spec.medium]"""
        eps = [MockEps(1.0), MockEps(2.0)]
        pt = ComPoint(eps,
                       [np.array([[0, 0, 20]]), np.array([[0, 0, -5]])],
                       [1, 2])

        # This pattern is used in farfield:
        #   ind = pt.index[pt.inout == spec.medium]
        # pt.index is a list, pt.inout is an array, so we need
        # to handle the boolean mask on a list. In practice the
        # dipole code loops over this. Let's verify compatibility.
        medium = 1
        mask = pt.inout == medium
        matched_indices = [pt.index[i] for i in range(len(pt.index)) if mask[i]]
        assert len(matched_indices) == 1
        np.testing.assert_array_equal(matched_indices[0], [0])


# ---------------------------------------------------------------------------
# Integration test with DipoleStat
# ---------------------------------------------------------------------------

class TestComPointWithDipoleStat(object):

    def test_dipole_stat_construction(self):
        from mnpbem.simulation.dipole_stat import DipoleStat

        eps = [MockEps(1.0)]
        pos = np.array([[0, 0, 20]], dtype = np.float64)
        pt = ComPoint(eps, [pos], [1])

        dip = np.array([0, 0, 1], dtype = np.float64)
        exc = DipoleStat(pt, dip)

        assert exc.pt is pt
        assert exc.pt.n == 1
        assert exc.dip.shape == (1, 3, 1)
        np.testing.assert_allclose(exc.dip[0, :, 0], [0, 0, 1])

    def test_dipole_stat_three_dipoles(self):
        from mnpbem.simulation.dipole_stat import DipoleStat

        eps = [MockEps(1.0)]
        pos = np.array([[0, 0, 20]], dtype = np.float64)
        pt = ComPoint(eps, [pos], [1])

        exc = DipoleStat(pt)
        assert exc.dip.shape == (1, 3, 3)

    def test_dipole_stat_multiple_positions(self):
        from mnpbem.simulation.dipole_stat import DipoleStat

        eps = [MockEps(1.0)]
        pos = np.array([[0, 0, 10], [0, 0, 20], [0, 0, 30]], dtype = np.float64)
        pt = ComPoint(eps, [pos], [1])

        dip = np.array([0, 0, 1], dtype = np.float64)
        exc = DipoleStat(pt, dip)

        assert exc.pt.n == 3
        assert exc.dip.shape == (3, 3, 1)
