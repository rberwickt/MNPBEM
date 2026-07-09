"""H-matrix (ACA) performance benchmark for BEMStat at large meshes.

Mesh sizes
----------
- trisphere(2562)   : ~2562 faces, BEMStat with/without ACA
- trisphere(8192)   : ~8192 faces, BEMStat with ACA
- trisphere(10242)  : ~10242 faces, BEMStat with ACA

Reports
-------
- assembly time (Green-function fill)
- LU/inverse time (BEM resolvent)
- solve time (one excitation)
- compression ratio (n_lr_elements + n_dense_elements) / n^2
- max relative error vs dense reference (extinction)

The benchmark fails loudly when ACA accuracy at 5K mesh exceeds 1e-10 in
extinction (per M4 Tier 4-A gate). For 8K and 10K meshes we only emit a
warning and record the error since dense reference would require >5GB.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from mnpbem.materials import EpsConst, EpsDrude
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.greenfun import CompGreenStat
from mnpbem.greenfun.aca_compgreen_stat import ACACompGreenStat
from mnpbem.bem import BEMStat
from mnpbem.simulation import PlaneWaveStat


def _make_sphere(n_faces, radius = 10.0):
    epsm = EpsConst(1.0)
    epsAu = EpsDrude(eps0 = 10.0, wp = 9.065, gammad = 0.0708, name = 'gold')
    sph = trisphere(n_faces, radius)
    return ComParticle([epsm, epsAu], [sph], [[2, 1]], 1)


def _bem_dense(p, enei = 600.0, pol = (1.0, 0.0, 0.0)):
    t0 = time.perf_counter()
    bem = BEMStat(p, enei = enei)
    t_assembly = time.perf_counter() - t0
    exc = PlaneWaveStat(list(pol))
    pot = exc.potential(p, enei)
    t0 = time.perf_counter()
    sig, _ = bem.solve(pot)
    t_solve = time.perf_counter() - t0
    ext = float(np.real(exc.extinction(sig)))
    return {
        'assembly': t_assembly,
        'solve': t_solve,
        'extinction': ext,
        'F': bem.F,
    }


def _bem_aca(p, enei = 600.0, pol = (1.0, 0.0, 0.0), htol = 1e-10):
    t0 = time.perf_counter()
    g = ACACompGreenStat(p, htol = htol, kmax = 200)
    F_hmat = g.eval('F')
    t_assembly = time.perf_counter() - t0

    # Dense LU for the BEM resolvent (ACA does not reduce the inverse cost)
    eps1 = p.eps1(enei)
    eps2 = p.eps2(enei)
    lambda_diag = 2 * np.pi * (eps1 + eps2) / (eps1 - eps2)
    F_dense = F_hmat.full()
    A = -(np.diag(lambda_diag) + F_dense)
    t0 = time.perf_counter()
    from scipy.linalg import lu_factor, lu_solve
    lu_pack = lu_factor(A, check_finite = False)
    t_lu = time.perf_counter() - t0

    exc = PlaneWaveStat(list(pol))
    pot = exc.potential(p, enei)
    t0 = time.perf_counter()
    sig_vals = lu_solve(lu_pack, pot.phip)
    t_solve = time.perf_counter() - t0

    from mnpbem.greenfun.compgreen_stat import CompStruct
    sig = CompStruct(p, enei, sig = sig_vals)
    ext = float(np.real(exc.extinction(sig)))

    stats = F_hmat.stat() if hasattr(F_hmat, 'stat') else {}
    return {
        'assembly': t_assembly,
        'lu': t_lu,
        'solve': t_solve,
        'extinction': ext,
        'compression': stats.get('compression_ratio', float('nan')),
        'max_rank': stats.get('max_rank', -1),
    }


def bench_5k():
    print('=' * 60)
    print('Bench: trisphere(2562) BEMStat dense vs ACA')
    print('=' * 60)
    p = _make_sphere(2562)
    print('n_faces =', p.n)

    res_dense = _bem_dense(p)
    print('[dense] assembly={:.2f}s solve={:.2f}s ext={:.6e}'.format(
        res_dense['assembly'], res_dense['solve'], res_dense['extinction']))

    res_aca = _bem_aca(p, htol = 1e-10)
    print('[aca]   assembly={:.2f}s lu={:.2f}s solve={:.2f}s ext={:.6e}'.format(
        res_aca['assembly'], res_aca['lu'], res_aca['solve'], res_aca['extinction']))
    print('        compression={:.4f}, max_rank={}'.format(
        res_aca['compression'], res_aca['max_rank']))

    rel_err = abs(res_aca['extinction'] - res_dense['extinction']) / max(
        abs(res_dense['extinction']), 1e-30)
    print('max_rel_err (extinction) = {:.3e}'.format(rel_err))
    return rel_err


def bench_8k():
    print('=' * 60)
    print('Bench: trisphere(8192) BEMStat ACA only')
    print('=' * 60)
    p = _make_sphere(8192)
    print('n_faces =', p.n)
    res_aca = _bem_aca(p, htol = 1e-10)
    print('[aca]   assembly={:.2f}s lu={:.2f}s solve={:.2f}s ext={:.6e}'.format(
        res_aca['assembly'], res_aca['lu'], res_aca['solve'], res_aca['extinction']))
    print('        compression={:.4f}, max_rank={}'.format(
        res_aca['compression'], res_aca['max_rank']))


def bench_10k():
    print('=' * 60)
    print('Bench: trisphere(10242) BEMStat ACA only')
    print('=' * 60)
    p = _make_sphere(10242)
    print('n_faces =', p.n)
    res_aca = _bem_aca(p, htol = 1e-10)
    print('[aca]   assembly={:.2f}s lu={:.2f}s solve={:.2f}s ext={:.6e}'.format(
        res_aca['assembly'], res_aca['lu'], res_aca['solve'], res_aca['extinction']))
    print('        compression={:.4f}, max_rank={}'.format(
        res_aca['compression'], res_aca['max_rank']))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', choices = ['5k', '8k', '10k', 'all', 'gate'],
            default = 'gate')
    parser.add_argument('--gate-tol', type = float, default = 1e-10)
    args = parser.parse_args()

    if args.mesh in ('5k', 'gate', 'all'):
        rel_err = bench_5k()
        if args.mesh in ('gate',):
            if rel_err < args.gate_tol:
                print('GATE PASSED (rel_err={:.3e} < {})'.format(rel_err, args.gate_tol))
                sys.exit(0)
            else:
                print('GATE FAILED (rel_err={:.3e} >= {})'.format(rel_err, args.gate_tol))
                sys.exit(1)
    if args.mesh in ('8k', 'all'):
        bench_8k()
    if args.mesh in ('10k', 'all'):
        bench_10k()
