import os
import sys
import time

import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mnpbem.simulation.meshfield_fmm import (
    fmm_available,
    eval_freespace_field,
)


def _make_dataset(n_src, n_tgt, seed = 0):
    rng = np.random.RandomState(seed)
    src_pos = rng.randn(n_src, 3) * 1.5
    src_area = rng.uniform(0.5, 1.5, size = n_src)
    tgt_pos = rng.randn(n_tgt, 3) + np.array([8.0, 0.0, 0.0])
    sig_scalar = rng.randn(n_src) + 1j * rng.randn(n_src)
    h_xyz = rng.randn(n_src, 3) + 1j * rng.randn(n_src, 3)
    return src_pos, src_area, tgt_pos, sig_scalar, h_xyz


def _dense_eval(zk, k_wave, src_pos, src_area, tgt_pos, sig_scalar, h_xyz):
    n_tgt = tgt_pos.shape[0]
    diff = tgt_pos[:, None, :] - src_pos[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis = 2))
    d = np.maximum(d, np.finfo(float).eps)
    phase = np.exp(1j * zk * d)
    G = phase / d * src_area[None, :]
    Gp_factor = phase * (1j * zk - 1.0 / d) / (d * d) * src_area[None, :]

    a = G @ h_xyz
    grad_phi = np.einsum('ij,ijk->ik', Gp_factor * sig_scalar[None, :].astype(np.complex128), diff)

    grad_a = np.empty((3, n_tgt, 3), dtype=np.complex128)
    for axis in range(3):
        grad_a[axis] = np.einsum('ij,ijk->ik', Gp_factor * h_xyz[None, :, axis], diff)

    e = 1j * k_wave * a - grad_phi
    h = np.empty((n_tgt, 3), dtype=np.complex128)
    h[:, 0] = grad_a[2, :, 1] - grad_a[1, :, 2]
    h[:, 1] = grad_a[0, :, 2] - grad_a[2, :, 0]
    h[:, 2] = grad_a[1, :, 0] - grad_a[0, :, 1]
    return e, h


def main():
    assert fmm_available(), '[error] fmm3dpy not available'

    cases = [(5000, 1000), (5000, 10000), (5000, 100000), (5000, 1000000)]
    zk = 0.1 + 0.0j
    k_wave = zk

    for n_src, n_tgt in cases:
        print('[info] n_src={}, n_tgt={}, prod={:.2e}'.format(n_src, n_tgt, n_src * n_tgt))
        src_pos, src_area, tgt_pos, sig_scalar, h_xyz = _make_dataset(n_src, n_tgt)

        # FMM
        t0 = time.perf_counter()
        e_fmm, h_fmm = eval_freespace_field(zk, k_wave, src_pos, src_area, tgt_pos, sig_scalar, h_xyz, eps = 1e-9)
        t_fmm = time.perf_counter() - t0

        # Dense
        if n_src * n_tgt <= 5e7:
            t0 = time.perf_counter()
            e_dense, h_dense = _dense_eval(zk, k_wave, src_pos, src_area, tgt_pos, sig_scalar, h_xyz)
            t_dense = time.perf_counter() - t0

            e_rel = np.max(np.abs(e_fmm - e_dense)) / np.max(np.abs(e_dense))
            h_rel = np.max(np.abs(h_fmm - h_dense)) / np.max(np.abs(h_dense))
            speedup = t_dense / t_fmm
            print('  fmm: {:.3f}s, dense: {:.3f}s, speedup: {:.2f}x, e_rel: {:.2e}, h_rel: {:.2e}'.format(
                t_fmm, t_dense, speedup, e_rel, h_rel))
        else:
            print('  fmm: {:.3f}s (dense skipped, too large)'.format(t_fmm))


if __name__ == '__main__':
    main()
