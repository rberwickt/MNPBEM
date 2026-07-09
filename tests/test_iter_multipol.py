"""Multi-pol (npol >= 2) BEM iterative solve correctness vs single-pol concat.

When PlaneWaveRet/PlaneWaveStat is constructed with multiple polarization
vectors, the BEM excitation arrays carry an extra trailing axis of size
npol. The iterative solvers (BEMRetIter, BEMStatIter, BEMRetLayerIter)
must broadcast (n, 3, npol) shapes correctly through their _afun/_mfun
internals. This test verifies that running the multi-pol solve once
matches running each polarization sequentially.
"""
import os
import sys

import numpy as np
import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle, LayerStructure
from mnpbem.simulation import PlaneWaveRet, PlaneWaveStat
from mnpbem.bem import BEMRetIter, BEMStatIter, BEMRetLayerIter


_ENEI = 600.0
_TOL = 1e-10


def _sphere_particle():
    epstab = [EpsConst(1.0), EpsTable('gold.dat')]
    sphere = trisphere(144, diameter = 10)
    return ComParticle(epstab, [sphere], [[2, 1]])


def test_bem_ret_iter_multipol_matches_sequential():
    p = _sphere_particle()

    bem_m = BEMRetIter(p)
    exc_m = PlaneWaveRet([[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [0, 0, 1]])
    sig_m, _ = bem_m.solve(exc_m(p, _ENEI))

    bem_x = BEMRetIter(p)
    sig_x, _ = bem_x.solve(PlaneWaveRet([1, 0, 0], [0, 0, 1])(p, _ENEI))
    bem_y = BEMRetIter(p)
    sig_y, _ = bem_y.solve(PlaneWaveRet([0, 1, 0], [0, 0, 1])(p, _ENEI))

    assert np.max(np.abs(sig_m.sig1[:, 0] - sig_x.sig1)) < _TOL
    assert np.max(np.abs(sig_m.sig1[:, 1] - sig_y.sig1)) < _TOL
    assert np.max(np.abs(sig_m.sig2[:, 0] - sig_x.sig2)) < _TOL
    assert np.max(np.abs(sig_m.sig2[:, 1] - sig_y.sig2)) < _TOL
    assert np.max(np.abs(sig_m.h1[:, :, 0] - sig_x.h1)) < _TOL
    assert np.max(np.abs(sig_m.h1[:, :, 1] - sig_y.h1)) < _TOL
    assert np.max(np.abs(sig_m.h2[:, :, 0] - sig_x.h2)) < _TOL
    assert np.max(np.abs(sig_m.h2[:, :, 1] - sig_y.h2)) < _TOL


def test_bem_stat_iter_multipol_matches_sequential():
    p = _sphere_particle()

    bem_m = BEMStatIter(p)
    exc_m = PlaneWaveStat([[1, 0, 0], [0, 1, 0]])
    sig_m, _ = bem_m.solve(exc_m(p, _ENEI))

    bem_x = BEMStatIter(p)
    sig_x, _ = bem_x.solve(PlaneWaveStat([1, 0, 0])(p, _ENEI))
    bem_y = BEMStatIter(p)
    sig_y, _ = bem_y.solve(PlaneWaveStat([0, 1, 0])(p, _ENEI))

    assert np.max(np.abs(sig_m.sig[:, 0] - sig_x.sig)) < _TOL
    assert np.max(np.abs(sig_m.sig[:, 1] - sig_y.sig)) < _TOL


def test_bem_ret_layer_iter_multipol_matches_sequential():
    epstab = [EpsConst(1.0), EpsTable('gold.dat'), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 3], [0.0])
    sphere = trisphere(60, 20.0)
    sphere.shift([0, 0, -sphere.pos[:, 2].min() + 1.0])
    p = ComParticle(epstab, [sphere], [[2, 1]], [1])

    bem_m = BEMRetLayerIter(p, layer = layer)
    exc_m = PlaneWaveRet([[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [0, 0, 1]])
    sig_m, _ = bem_m.solve(exc_m(p, _ENEI))

    bem_x = BEMRetLayerIter(p, layer = layer)
    sig_x, _ = bem_x.solve(PlaneWaveRet([1, 0, 0], [0, 0, 1])(p, _ENEI))
    bem_y = BEMRetLayerIter(p, layer = layer)
    sig_y, _ = bem_y.solve(PlaneWaveRet([0, 1, 0], [0, 0, 1])(p, _ENEI))

    assert np.max(np.abs(sig_m.sig1[:, 0] - sig_x.sig1)) < _TOL
    assert np.max(np.abs(sig_m.sig1[:, 1] - sig_y.sig1)) < _TOL
    assert np.max(np.abs(sig_m.sig2[:, 0] - sig_x.sig2)) < _TOL
    assert np.max(np.abs(sig_m.sig2[:, 1] - sig_y.sig2)) < _TOL
    assert np.max(np.abs(sig_m.h1[:, :, 0] - sig_x.h1)) < _TOL
    assert np.max(np.abs(sig_m.h1[:, :, 1] - sig_y.h1)) < _TOL
    assert np.max(np.abs(sig_m.h2[:, :, 0] - sig_x.h2)) < _TOL
    assert np.max(np.abs(sig_m.h2[:, :, 1] - sig_y.h2)) < _TOL
