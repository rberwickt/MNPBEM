import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mnpbem.geometry import Compound
from mnpbem.geometry.particle import Particle
from mnpbem.geometry.compoint import Point
from mnpbem.geometry.mesh_generators import trisphere
from mnpbem.materials.eps_const import EpsConst


REF_PATH = os.path.join(os.path.dirname(__file__), 'data', 'compound_ref.txt')


def _load_ref() -> Dict[str, List[str]]:
    ref = dict()
    with open(REF_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            ref[tokens[0]] = tokens[1:]
    return ref


def _make_compound_particles() -> Tuple[List[Any], List[Any], np.ndarray]:
    eps_list = [EpsConst(1.0), EpsConst(-10 + 1j), EpsConst(2.25)]
    p1 = trisphere(144, 5)
    p2 = trisphere(256, 10)
    p3 = trisphere(60, 3)
    # MATLAB inout: [ 2, 1; 3, 1; 2, 1 ]
    inout = np.array([[2, 1], [3, 1], [2, 1]], dtype = int)
    return eps_list, [p1, p2, p3], inout


class TestCompoundSize(object):

    def test_size_n_np(self):
        ref = _load_ref()
        eps_list, parts, inout = _make_compound_particles()
        c = Compound(eps_list, parts, inout)

        expected_size = [int(x) for x in ref['size']]
        expected_n = int(ref['n'][0])
        expected_np = int(ref['np'][0])

        np.testing.assert_array_equal(c.size, expected_size)
        assert int(c.n) == expected_n
        assert int(c.np) == expected_np


class TestCompoundDielectric(object):

    def test_dielectric_inside(self):
        ref = _load_ref()
        eps_list, parts, inout = _make_compound_particles()
        c = Compound(eps_list, parts, inout)

        d_in = c.dielectric(600.0, 1)
        assert len(d_in) == int(ref['d_in_count'][0])
        for i, val in enumerate(d_in):
            # val may be (eps, k) tuple or scalar
            if isinstance(val, tuple):
                eps_v = val[0]
            else:
                eps_v = val
            eps_v = complex(np.asarray(eps_v).flat[0])
            rr, ii = ref['d_in_{}'.format(i + 1)]
            assert abs(eps_v.real - float(rr)) < 1e-10
            assert abs(eps_v.imag - float(ii)) < 1e-10

    def test_dielectric_outside(self):
        ref = _load_ref()
        eps_list, parts, inout = _make_compound_particles()
        c = Compound(eps_list, parts, inout)

        d_out = c.dielectric(600.0, 2)
        assert len(d_out) == int(ref['d_out_count'][0])
        for i, val in enumerate(d_out):
            if isinstance(val, tuple):
                eps_v = val[0]
            else:
                eps_v = val
            eps_v = complex(np.asarray(eps_v).flat[0])
            rr, ii = ref['d_out_{}'.format(i + 1)]
            assert abs(eps_v.real - float(rr)) < 1e-10
            assert abs(eps_v.imag - float(ii)) < 1e-10


class TestCompoundIndex(object):

    def test_index_single(self):
        ref = _load_ref()
        eps_list, parts, inout = _make_compound_particles()
        c = Compound(eps_list, parts, inout)

        # MATLAB is 1-indexed; Compound.index uses 1-indexed input
        # but returns 0-indexed slice ranges. Convert MATLAB refs to 0-indexed.
        idx1 = c.index(1)
        assert len(idx1) == int(ref['idx1_len'][0])
        matlab_first, matlab_last = ref['idx1_first_last']
        assert int(idx1[0]) == int(matlab_first) - 1
        assert int(idx1[-1]) == int(matlab_last) - 1

        idx2 = c.index(2)
        assert len(idx2) == int(ref['idx2_len'][0])
        matlab_first, matlab_last = ref['idx2_first_last']
        assert int(idx2[0]) == int(matlab_first) - 1
        assert int(idx2[-1]) == int(matlab_last) - 1

        idx3 = c.index(3)
        assert len(idx3) == int(ref['idx3_len'][0])
        matlab_first, matlab_last = ref['idx3_first_last']
        assert int(idx3[0]) == int(matlab_first) - 1
        assert int(idx3[-1]) == int(matlab_last) - 1

    def test_index_multi(self):
        ref = _load_ref()
        eps_list, parts, inout = _make_compound_particles()
        c = Compound(eps_list, parts, inout)

        idx13 = c.index([1, 3])
        assert len(idx13) == int(ref['idx13_len'][0])


class TestCompoundIpart(object):

    def test_ipart(self):
        ref = _load_ref()
        eps_list, parts, inout = _make_compound_particles()
        c = Compound(eps_list, parts, inout)

        sizes = c.size
        # MATLAB global indices (1-indexed)
        matlab_ind = np.array([1, 50, int(sizes[0]) + 1, int(sizes[0]) + 100,
                               int(sizes[0]) + int(sizes[1]) + 1])

        ipt, rel = c.ipart(matlab_ind)
        expected_ipt = np.array([int(x) for x in ref['ipt']])
        expected_rel = np.array([int(x) for x in ref['rel']])

        np.testing.assert_array_equal(ipt, expected_ipt)
        np.testing.assert_array_equal(rel, expected_rel)


class TestCompoundExpand(object):

    def test_expand_scalar(self):
        ref = _load_ref()
        eps_list, parts, inout = _make_compound_particles()
        c = Compound(eps_list, parts, inout)

        out = c.expand(42)
        assert len(out) == int(ref['exp_scalar_len'][0])
        assert float(out[0]) == float(ref['exp_scalar_first'][0])
        assert np.all(out == 42)

    def test_expand_cell(self):
        ref = _load_ref()
        eps_list, parts, inout = _make_compound_particles()
        c = Compound(eps_list, parts, inout)

        out = c.expand([10, 20, 30])
        assert len(out) == int(ref['exp_cell_len'][0])

        sizes = c.size
        # End of particle 1 chunk
        p1_last = int(out[int(sizes[0]) - 1])
        assert p1_last == int(float(ref['exp_cell_p1_last'][0]))

        p2_last = int(out[int(sizes[0]) + int(sizes[1]) - 1])
        assert p2_last == int(float(ref['exp_cell_p2_last'][0]))

        p3_last = int(out[int(sizes[0]) + int(sizes[1]) + int(sizes[2]) - 1])
        assert p3_last == int(float(ref['exp_cell_p3_last'][0]))


class TestCompoundEq(object):

    def test_eq_identical(self):
        eps_list, parts, inout = _make_compound_particles()
        c1 = Compound(eps_list, parts, inout)
        c2 = Compound(eps_list, parts, inout)
        ref = _load_ref()
        assert (c1 == c2) == bool(int(ref['eq'][0]))
        assert (c1 != c2) == bool(int(ref['ne'][0]))

    def test_eq_shuffled(self):
        eps_list, parts, inout = _make_compound_particles()
        c1 = Compound(eps_list, parts, inout)
        shuffled_parts = [parts[1], parts[0], parts[2]]
        c3 = Compound(eps_list, shuffled_parts, inout)
        ref = _load_ref()
        assert (c1 == c3) == bool(int(ref['eq_shuffle'][0]))


class TestCompoundMask(object):

    def test_mask_with_matlab_style(self):
        ref = _load_ref()
        eps_list, parts, inout = _make_compound_particles()
        c = Compound(eps_list, parts, inout)
        c.set_mask_matlab([1, 3])

        assert int(c.n) == int(ref['masked_n'][0])
        expected = [int(x) for x in ref['masked_size']]
        np.testing.assert_array_equal(c.size, expected)

        masked_idx1 = c.index(1)
        assert len(masked_idx1) == int(ref['masked_idx1_len'][0])

        masked_idx2 = c.index(2)
        assert len(masked_idx2) == int(ref['masked_idx2_len'][0])
        # MATLAB masked_idx2_first is 1-indexed; our index returns 0-indexed
        assert int(masked_idx2[0]) == int(ref['masked_idx2_first'][0]) - 1


class TestCompoundSet(object):

    def test_set_property(self):
        eps_list, parts, inout = _make_compound_particles()
        c = Compound(eps_list, parts, inout)
        # Set an arbitrary attribute on pc
        c.set(tag = 'masked-compound')
        assert getattr(c.pc, 'tag') == 'masked-compound'


class TestCompoundSubsrefFallback(object):

    def test_pc_attribute_fallback(self):
        eps_list, parts, inout = _make_compound_particles()
        c = Compound(eps_list, parts, inout)
        # pos should be delegated to pc
        pos = c.pos
        assert pos.shape[0] == c.n
        assert pos.shape[1] == 3

    def test_eps1_shape(self):
        eps_list, parts, inout = _make_compound_particles()
        c = Compound(eps_list, parts, inout)
        e1 = c.eps1(600.0)
        # expand broadcasts dielectric list -> per-face values
        assert e1.shape[0] == c.n


class TestCompoundWithPoints(object):

    def test_points_compound_single_column_inout(self):
        # Compound of points: inout is a single column
        eps_list = [EpsConst(1.0), EpsConst(2.0)]
        pt1 = Point(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        pt2 = Point(np.array([[0.0, 1.0, 0.0]]))
        c = Compound(eps_list, [pt1, pt2], np.array([1, 2]))

        assert c.np == 2
        assert c.n == 3
        np.testing.assert_array_equal(c.size, [2, 1])

        # dielectric with single-column inout -> one eps per group
        d = c.dielectric(500.0, 1)
        assert len(d) == 2

    def test_ipart_points(self):
        eps_list = [EpsConst(1.0), EpsConst(2.0)]
        pt1 = Point(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        pt2 = Point(np.array([[0.0, 1.0, 0.0]]))
        c = Compound(eps_list, [pt1, pt2], np.array([1, 2]))

        # MATLAB 1-indexed global: index 1 -> group 1 rel 1; index 3 -> group 2 rel 1
        ipt, rel = c.ipart([1, 2, 3])
        np.testing.assert_array_equal(ipt, [1, 1, 2])
        np.testing.assert_array_equal(rel, [1, 2, 1])
