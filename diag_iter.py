"""
Au@Ag dimer iter convergence diagnostic.

Reproduces the case_g_auag_dimer_small mid-band drift, then sweeps
htol / preconditioner / maxit / tol to identify the dominant cause.

Outputs JSON results and a markdown summary.
"""

import os
import sys
import json
import time
import numpy as np

# Ensure we use the worktree mnpbem
sys.path.insert(0, '/home/yoojk20/scratch/v151_beta_iter_drift')

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.bem import BEMRet, BEMRetIter
from mnpbem.simulation import PlaneWaveRet


def build_auag_case():
    """case_auag_dimer_small (1136 face) — same as validation case_g."""
    enei = np.linspace(380.0, 700.0, 7)
    core_d = 5.0
    shell_t = 1.5
    outer_d = core_d + 2.0 * shell_t
    gap = 0.6
    n_faces_core = 144
    n_faces_shell = 144

    epstab = [EpsConst(1.77),
              EpsTable('gold.dat'),
              EpsTable('silver.dat')]
    half = (outer_d + gap) / 2.0
    p1_shell = trisphere(n_faces_shell, outer_d)
    p1_core = trisphere(n_faces_core, core_d)
    p1_shell.shift([-half, 0.0, 0.0]); p1_core.shift([-half, 0.0, 0.0])
    p2_shell = trisphere(n_faces_shell, outer_d)
    p2_core = trisphere(n_faces_core, core_d)
    p2_shell.shift([+half, 0.0, 0.0]); p2_core.shift([+half, 0.0, 0.0])
    inds = [[3, 1], [2, 3], [3, 1], [2, 3]]
    p = ComParticle(epstab, [p1_shell, p1_core, p2_shell, p2_core],
            inds, [1, 2], interp = 'curv')
    return p, enei


def excitation():
    pol = np.array([1.0, 0.0, 0.0])
    dir_vec = np.array([0.0, 0.0, 1.0])
    return PlaneWaveRet(pol, dir_vec)


def cross_sections(bem, exc, p, enei, label = ''):
    n = len(enei)
    ext = np.zeros(n); sca = np.zeros(n); abso = np.zeros(n)
    info_per_wl = []
    for i, e in enumerate(enei):
        sig, bem = bem.solve(exc.potential(p, e))
        ext_v = float(np.real(np.ravel(exc.extinction(sig)))[0])
        sca_raw = exc.scattering(sig)
        sca_v = float(np.real(np.ravel(sca_raw[0]
                if isinstance(sca_raw, tuple) else sca_raw))[0])
        abs_v = ext_v - sca_v
        ext[i] = ext_v; sca[i] = sca_v; abso[i] = abs_v
        # Capture last GMRES info
        info_entry = {'enei': float(e), 'ext': ext_v}
        try:
            flags, relres, iters = bem.info()
            if flags is not None and len(flags) > 0:
                info_entry['flag_last'] = int(flags[-1])
                info_entry['relres_last'] = float(relres[-1])
                info_entry['iter_last'] = list(map(int, np.atleast_1d(iters[-1]).tolist()))
        except Exception:
            pass
        info_per_wl.append(info_entry)
        print('  [{}] enei={:.1f}: ext={:.3f}{}'.format(
            label, e, ext_v,
            ' it={} relres={:.2e}'.format(
                info_entry.get('iter_last'),
                info_entry.get('relres_last', 0.0))
            if 'iter_last' in info_entry else ''))
    return dict(ext = ext.tolist(), sca = sca.tolist(), abs_ = abso.tolist(),
            info_per_wl = info_per_wl)


def rel_diff(a, b):
    a = np.array(a); b = np.array(b)
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-12)


def main():
    print('===== Building Au@Ag dimer (case_g, 1136 face) =====')
    t0 = time.time()
    p, enei = build_auag_case()
    print('  nfaces = {}'.format(p.nfaces))
    print('  build time = {:.1f}s'.format(time.time() - t0))

    exc = excitation()

    runs = {}

    # Reference: dense
    print('\n===== Reference: BEMRet (dense) =====')
    t0 = time.time()
    bem = BEMRet(p)
    runs['dense'] = cross_sections(bem, exc, p, enei, 'dense')
    runs['dense']['wall_s'] = time.time() - t0
    ref = runs['dense']['ext']

    # Variant configurations to test hypotheses.
    variants = [
        # baseline iter (reproduce case_g warn)
        ('iter_baseline',     dict(hmatrix=True, htol=1e-6, tol=1e-4, maxit=200, precond='hmat')),
        # Stronger tol
        ('iter_tol1e-6',      dict(hmatrix=True, htol=1e-6, tol=1e-6, maxit=200, precond='hmat')),
        ('iter_tol1e-8',      dict(hmatrix=True, htol=1e-6, tol=1e-8, maxit=400, precond='hmat')),
        # htol sweep
        ('iter_htol1e-8',     dict(hmatrix=True, htol=1e-8, tol=1e-6, maxit=400, precond='hmat')),
        ('iter_htol1e-10',    dict(hmatrix=True, htol=1e-10, tol=1e-6, maxit=400, precond='hmat')),
        # No hmatrix (dense iter)
        ('iter_no_hmat',      dict(hmatrix=False, tol=1e-8, maxit=400, precond='hmat')),
        # Restart larger
        ('iter_restart200',   dict(hmatrix=True, htol=1e-8, tol=1e-8, maxit=400, restart=200,
                                   precond='hmat')),
    ]

    for name, kwargs in variants:
        print('\n===== {} kwargs={} =====' .format(name, kwargs))
        try:
            t0 = time.time()
            bem = BEMRetIter(p, **kwargs)
            runs[name] = cross_sections(bem, exc, p, enei, name)
            runs[name]['wall_s'] = time.time() - t0
            runs[name]['kwargs'] = {k: (v if not callable(v) else str(v))
                                     for k, v in kwargs.items()}
        except Exception as e:
            import traceback
            runs[name] = {'error': '{}: {}'.format(type(e).__name__, e),
                          'tb': traceback.format_exc(),
                          'kwargs': {k: (v if not callable(v) else str(v))
                                     for k, v in kwargs.items()}}
            print('  ERROR: {}'.format(e))

    # rel diff vs dense
    print('\n===== Rel diff vs dense (per wl, ext) =====')
    for name, run in runs.items():
        if name == 'dense' or 'error' in run:
            continue
        rd = rel_diff(run['ext'], ref)
        run['rel_diff_vs_dense'] = rd.tolist()
        run['max_rel_diff'] = float(rd.max())
        print('  {}: max={:.3f} ({})'.format(
            name, rd.max(),
            ', '.join('{:.0%}'.format(x) for x in rd)))

    out_path = '/tmp/auag_iter_diag.json'
    with open(out_path, 'w') as f:
        json.dump(runs, f, indent = 2)
    print('\nResults: {}'.format(out_path))


if __name__ == '__main__':
    main()
