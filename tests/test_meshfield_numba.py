"""
Unit tests for MeshField Numba acceleration.

Verifies that the Numba-accelerated meshfield path produces results that
match the numpy reference path within 1e-12 relative tolerance for both
quasistatic and retarded simulations.
"""

import importlib
import os

import numpy as np
import pytest

from mnpbem.bem import BEMRet, BEMStat
from mnpbem.geometry import ComParticle, trisphere
from mnpbem.materials import EpsConst
from mnpbem.simulation import MeshField, PlaneWaveRet, PlaneWaveStat


def _setup():
    """Build a small reference particle and a 4 x 4 grid."""
    eps = [EpsConst(1.0), EpsConst(-10 + 1j)]
    p = ComParticle(eps, [trisphere(144, 10.0)], [[2, 1]])
    x, z = np.meshgrid(np.linspace(-20, 20, 4), np.linspace(-20, 20, 4))
    return p, x, z


def _reload_numba_module():
    """Re-read MNPBEM_NUMBA after the env var changes."""
    import mnpbem.simulation._meshfield_numba as mod
    importlib.reload(mod)


def test_meshfield_stat_numba_matches_numpy():
    p, x, z = _setup()
    bem = BEMStat(p)
    exc = PlaneWaveStat([1, 0, 0])
    sig, _ = bem.solve(exc(p, 600.0))

    os.environ['MNPBEM_NUMBA'] = '1'
    _reload_numba_module()
    mf_nb = MeshField(p, x, 0, z, sim = 'stat')
    e_nb, _ = mf_nb.field(sig, inout = 2)

    os.environ['MNPBEM_NUMBA'] = '0'
    _reload_numba_module()
    mf_np = MeshField(p, x, 0, z, sim = 'stat')
    e_np, _ = mf_np.field(sig, inout = 2)

    err = np.max(np.abs(e_nb - e_np))
    norm = np.max(np.abs(e_nb))
    assert err / norm < 1e-12, f'rel err {err/norm} > 1e-12'


def test_meshfield_ret_numba_matches_numpy():
    p, x, z = _setup()
    bem = BEMRet(p)
    exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])
    sig, _ = bem.solve(exc(p, 600.0))

    os.environ['MNPBEM_NUMBA'] = '1'
    _reload_numba_module()
    mf_nb = MeshField(p, x, 0, z, sim = 'ret')
    e_nb, h_nb = mf_nb.field(sig, inout = 2)

    os.environ['MNPBEM_NUMBA'] = '0'
    _reload_numba_module()
    mf_np = MeshField(p, x, 0, z, sim = 'ret')
    e_np, h_np = mf_np.field(sig, inout = 2)

    e_err = np.max(np.abs(e_nb - e_np))
    e_norm = np.max(np.abs(e_nb))
    h_err = np.max(np.abs(h_nb - h_np))
    h_norm = np.max(np.abs(h_nb))
    assert e_err / e_norm < 1e-12, f'E rel err {e_err/e_norm} > 1e-12'
    assert h_err / h_norm < 1e-12, f'H rel err {h_err/h_norm} > 1e-12'


def test_meshfield_ret_numba_inout1_matches_numpy():
    """Inside-the-particle field eval (inout = 1) parity."""
    p, x, z = _setup()
    bem = BEMRet(p)
    exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])
    sig, _ = bem.solve(exc(p, 600.0))

    os.environ['MNPBEM_NUMBA'] = '1'
    _reload_numba_module()
    mf_nb = MeshField(p, x, 0, z, sim = 'ret')
    e_nb, h_nb = mf_nb.field(sig, inout = 1)

    os.environ['MNPBEM_NUMBA'] = '0'
    _reload_numba_module()
    mf_np = MeshField(p, x, 0, z, sim = 'ret')
    e_np, h_np = mf_np.field(sig, inout = 1)

    e_err = np.max(np.abs(e_nb - e_np))
    e_norm = max(np.max(np.abs(e_nb)), 1e-30)
    h_err = np.max(np.abs(h_nb - h_np))
    h_norm = max(np.max(np.abs(h_nb)), 1e-30)
    assert e_err / e_norm < 1e-12
    assert h_err / h_norm < 1e-12
