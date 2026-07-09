import os
import sys

import numpy as np
import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mnpbem.simulation.meshfield_fmm import (
    fmm_available,
    eval_freespace_field,
    eval_freespace_potential,
)


pytestmark = pytest.mark.skipif(not fmm_available(), reason = 'fmm3dpy not installed')


def _dense_phi_grad(zk, src_pos, src_area, tgt_pos, sig_scalar):
    diff = tgt_pos[:, None, :] - src_pos[None, :, :]   # (n_tgt, n_src, 3)
    d = np.sqrt(np.sum(diff * diff, axis = 2))
    d = np.maximum(d, np.finfo(float).eps)
    phase = np.exp(1j * zk * d)
    G = phase / d * src_area[None, :]                  # (n_tgt, n_src)
    phi = G @ sig_scalar.astype(np.complex128)         # (n_tgt,)
    Gp_factor = phase * (1j * zk - 1.0 / d) / (d * d) * src_area[None, :]
    grad_phi = np.einsum('ij,ijk->ik', Gp_factor * sig_scalar[None, :].astype(np.complex128), diff)
    return phi, grad_phi


def _dense_a_field(zk, src_pos, src_area, tgt_pos, h_xyz):
    diff = tgt_pos[:, None, :] - src_pos[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis = 2))
    d = np.maximum(d, np.finfo(float).eps)
    phase = np.exp(1j * zk * d)
    G = phase / d * src_area[None, :]
    a = G @ h_xyz                                      # (n_tgt, 3)
    Gp_factor = phase * (1j * zk - 1.0 / d) / (d * d) * src_area[None, :]    # (n_tgt, n_src)
    grad_a = np.empty((3, tgt_pos.shape[0], 3), dtype=np.complex128)         # (axis_a, tgt, dx)
    for axis in range(3):
        grad_a[axis] = np.einsum('ij,ijk->ik', Gp_factor * h_xyz[None, :, axis], diff)
    h = np.empty((tgt_pos.shape[0], 3), dtype=np.complex128)
    h[:, 0] = grad_a[2, :, 1] - grad_a[1, :, 2]
    h[:, 1] = grad_a[0, :, 2] - grad_a[2, :, 0]
    h[:, 2] = grad_a[1, :, 0] - grad_a[0, :, 1]
    return a, h


def _make_mesh_data(n_src = 144, n_tgt = 32, seed = 0, separation = 8.0):
    rng = np.random.RandomState(seed)
    src_pos = rng.randn(n_src, 3) * 1.5
    src_area = rng.uniform(0.5, 1.5, size = n_src)
    tgt_pos = rng.randn(n_tgt, 3) + np.array([separation, 0.0, 0.0])
    sig_scalar = rng.randn(n_src) + 1j * rng.randn(n_src)
    h_xyz = rng.randn(n_src, 3) + 1j * rng.randn(n_src, 3)
    return src_pos, src_area, tgt_pos, sig_scalar, h_xyz


def test_potential_dense_vs_fmm():
    src_pos, src_area, tgt_pos, sig_scalar, _ = _make_mesh_data()
    zk = 0.05 + 0.0j

    phi_fmm = eval_freespace_potential(zk, src_pos, src_area, tgt_pos, sig_scalar, eps = 1e-12)
    phi_dense, _ = _dense_phi_grad(zk, src_pos, src_area, tgt_pos, sig_scalar)

    rel_err = np.max(np.abs(phi_fmm - phi_dense)) / np.max(np.abs(phi_dense))
    assert rel_err < 1e-9, '[error] potential rel err {} too large'.format(rel_err)


def test_field_dense_vs_fmm():
    src_pos, src_area, tgt_pos, sig_scalar, h_xyz = _make_mesh_data()
    zk = 0.05 + 0.0j
    k_wave = zk

    e_fmm, h_fmm = eval_freespace_field(zk, k_wave, src_pos, src_area, tgt_pos, sig_scalar, h_xyz, eps = 1e-12)

    _, grad_phi_dense = _dense_phi_grad(zk, src_pos, src_area, tgt_pos, sig_scalar)
    a_dense, h_dense = _dense_a_field(zk, src_pos, src_area, tgt_pos, h_xyz)
    e_dense = 1j * k_wave * a_dense - grad_phi_dense

    e_rel = np.max(np.abs(e_fmm - e_dense)) / np.max(np.abs(e_dense))
    h_rel = np.max(np.abs(h_fmm - h_dense)) / np.max(np.abs(h_dense))
    assert e_rel < 1e-9, '[error] e rel err {} too large'.format(e_rel)
    assert h_rel < 1e-9, '[error] h rel err {} too large'.format(h_rel)


def test_field_complex_zk_lossy():
    src_pos, src_area, tgt_pos, sig_scalar, h_xyz = _make_mesh_data(seed = 7)
    zk = 0.08 + 0.01j
    k_wave = zk

    e_fmm, h_fmm = eval_freespace_field(zk, k_wave, src_pos, src_area, tgt_pos, sig_scalar, h_xyz, eps = 1e-12)
    _, grad_phi_dense = _dense_phi_grad(zk, src_pos, src_area, tgt_pos, sig_scalar)
    a_dense, h_dense = _dense_a_field(zk, src_pos, src_area, tgt_pos, h_xyz)
    e_dense = 1j * k_wave * a_dense - grad_phi_dense

    e_rel = np.max(np.abs(e_fmm - e_dense)) / np.max(np.abs(e_dense))
    h_rel = np.max(np.abs(h_fmm - h_dense)) / np.max(np.abs(h_dense))
    assert e_rel < 1e-9
    assert h_rel < 1e-9


# ---------------------------------------------------------------------------
# Integration: MeshField dense vs FMM
# ---------------------------------------------------------------------------

def test_meshfield_dense_vs_fmm_ret():
    from mnpbem.materials import EpsConst, EpsDrude
    from mnpbem.geometry import trisphere, ComParticle
    from mnpbem.bem import BEMRet
    from mnpbem.simulation import PlaneWaveRet, MeshField

    epsm = EpsConst(1.0)
    epsAu = EpsDrude(eps0 = 10.0, wp = 9.065, gammad = 0.0708, name = 'gold')
    sphere = trisphere(144, 30.0)
    p = ComParticle([epsm, epsAu], [sphere], [[2, 1]], 1)

    bem = BEMRet(p)
    exc = PlaneWaveRet(np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.0]]))
    enei = 600.0
    sig, _ = bem.solve(exc(p, enei))

    rng = np.random.RandomState(42)
    tgt = rng.randn(32, 3) * 1.5 + np.array([200.0, 0.0, 0.0])  # outside, well separated
    x = tgt[:, 0]
    y = tgt[:, 1]
    z = tgt[:, 2]

    mf = MeshField(p, x, y, z, sim = 'ret', refine = False)

    e_dense, h_dense = mf(sig, inout = 2, fmm = False)
    e_fmm, h_fmm = mf(sig, inout = 2, fmm = True, fmm_eps = 1e-12)

    e_rel = np.max(np.abs(e_fmm - e_dense)) / np.max(np.abs(e_dense))
    h_rel = np.max(np.abs(h_fmm - h_dense)) / np.max(np.abs(h_dense))
    print('[info] meshfield e rel err: {:.3e}'.format(e_rel))
    print('[info] meshfield h rel err: {:.3e}'.format(h_rel))
    assert e_rel < 1e-9, '[error] e rel err {} too large'.format(e_rel)
    assert h_rel < 1e-9, '[error] h rel err {} too large'.format(h_rel)
