import os
import sys
import time
import resource

from typing import Any, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mnpbem.greenfun import CompStruct
from mnpbem.materials import EpsConst, make_nonlocal_pair
from mnpbem.geometry import ComParticle, trisphere
from mnpbem.greenfun import coverlayer
from mnpbem.bem import BEMRetIter
from mnpbem.simulation.planewave_ret import PlaneWaveRet


# v1.5.0 Lane E2 nonlocal benchmark: BEMRetIter on a cover-layer
# (EpsNonlocal) particle.  Compares two configurations:
#
#   v1.4.0 baseline  : BEMRetIter + hmatrix=True  (no Schur)
#   v1.5.0 new path  : BEMRetIter + hmatrix=True  + schur=True
#
# Expected: Schur reduction eliminates the cover-layer face block
# from the GMRES Krylov subspace.  Wall time and (Krylov-vector)
# memory should drop by ~30-50 percent because each iteration cycles
# through (2 * core_faces * 8) DOF instead of (total_faces * 8) DOF.
#
# CUSTOMIZE: scale via LANE_E2_NL_NFACES (per-sphere core face count).
# Default 12800 -> ~25k effective faces after coverlayer.shift.
N_FACE_CORE = int(os.environ.get('LANE_E2_NL_NFACES', '12800'))
DIAM = 30.0
ENEI = 636.36
DELTA_D = 0.05


def _make_nonlocal_sphere(n_face_core: int) -> Tuple[ComParticle, Any]:

    eps_b = EpsConst(1.0)
    core_eps, shell_eps = make_nonlocal_pair('gold',
            eps_embed = eps_b,
            delta_d = DELTA_D)

    delta_d = shell_eps.delta_d
    p_core = trisphere(n_face_core, DIAM - 2 * delta_d)
    p_shell = coverlayer.shift(p_core, delta_d)

    epstab = [eps_b, core_eps, shell_eps]
    inds = [[3, 1], [2, 3]]
    cp = ComParticle(epstab, [p_shell, p_core], inds, 1, 2)
    refun = coverlayer.refine(cp, [[1, 2]])
    return cp, refun


def _peak_rss_mb() -> float:

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _bench(label: str,
        cp: ComParticle,
        schur: bool) -> dict:

    print('\n=== {} (faces={}, hmatrix=True, schur={}) ==='.format(
            label, cp.nfaces, schur))

    pol = np.array([[1.0, 0.0, 0.0]])
    dirn = np.array([[0.0, 0.0, 1.0]])
    exc = PlaneWaveRet(pol = pol, dir = dirn)

    t0 = time.time()
    bem = BEMRetIter(cp,
            schur = schur,
            hmatrix = True,
            htol = 1e-6,
            kmax = [4, 100],
            cleaf = 64,
            tol = 1e-5,
            maxit = 400)
    t_build = time.time() - t0

    exc_struct = exc(cp, ENEI)

    # Initialize matrices (so init time is separate from GMRES time).
    t0 = time.time()
    bem._init_matrices(ENEI)
    t_init = time.time() - t0

    t0 = time.time()
    sig, bem_done = bem.solve(exc_struct)
    t_solve = time.time() - t0

    info = {}
    info['label'] = label
    info['schur_active'] = bool(getattr(bem, '_schur_active', False))
    info['n_shell'] = int(getattr(bem, '_shell_face_idx', np.array([])).size) if bem._shell_face_idx is not None else 0
    info['n_core'] = int(getattr(bem, '_core_face_idx', np.array([])).size) if bem._core_face_idx is not None else 0
    info['t_build'] = t_build
    info['t_init'] = t_init
    info['t_solve'] = t_solve
    info['t_total'] = t_build + t_init + t_solve
    info['peak_rss_mb'] = _peak_rss_mb()
    info['sig1_max'] = float(np.abs(sig.sig1).max())

    flags, relres, iter_info = bem_done.info()
    if relres:
        info['gmres_relres'] = float(relres[0])
        info['gmres_flag'] = int(flags[0])
        info['gmres_iter'] = int(iter_info[0][0]) if hasattr(iter_info[0], '__len__') else int(iter_info[0])

    print('[info] schur_active : {}'.format(info['schur_active']))
    print('[info] n_shell/core : {} / {}'.format(info['n_shell'], info['n_core']))
    print('[info] build  wall  : {:.2f}s'.format(info['t_build']))
    print('[info] init   wall  : {:.2f}s'.format(info['t_init']))
    print('[info] solve  wall  : {:.2f}s'.format(info['t_solve']))
    print('[info] total  wall  : {:.2f}s'.format(info['t_total']))
    print('[info] peak RSS     : {:.1f} MB'.format(info['peak_rss_mb']))
    print('[info] |sig1| max   : {:.4e}'.format(info['sig1_max']))
    if 'gmres_relres' in info:
        print('[info] gmres relres : {:.3e}, flag={}'.format(info['gmres_relres'], info['gmres_flag']))

    return info


def main() -> None:

    print('[info] v1.5.0 Lane E2 nonlocal benchmark')
    print('[info] N_FACE_CORE={}, diameter={} nm, enei={} nm, delta_d={} nm'.format(
            N_FACE_CORE, DIAM, ENEI, DELTA_D))

    cp, _ = _make_nonlocal_sphere(N_FACE_CORE)
    print('[info] mesh built: nfaces (shell+core)={}'.format(cp.nfaces))

    info_baseline = _bench('v1.4.0 baseline (hmatrix only)', cp, schur = False)
    info_schur = _bench('v1.5.0 (hmatrix + Schur)', cp, schur = True)

    print('\n=== Comparison ===')
    if info_schur['schur_active']:
        ratio_solve = info_schur['t_solve'] / max(info_baseline['t_solve'], 1e-9)
        ratio_total = info_schur['t_total'] / max(info_baseline['t_total'], 1e-9)
        print('[info] solve wall  v1.5.0 / v1.4.0 = {:.3f}'.format(ratio_solve))
        print('[info] total wall  v1.5.0 / v1.4.0 = {:.3f}'.format(ratio_total))
        delta_solve_pct = (1.0 - ratio_solve) * 100.0
        delta_total_pct = (1.0 - ratio_total) * 100.0
        print('[info] solve wall savings : {:.1f}%'.format(delta_solve_pct))
        print('[info] total wall savings : {:.1f}%'.format(delta_total_pct))
        print('[info] peak RSS  v1.5.0 / v1.4.0 = {:.3f}'.format(
                info_schur['peak_rss_mb'] / max(info_baseline['peak_rss_mb'], 1e-9)))
    else:
        print('[info] Schur not active -- partition not detected. '
              'Verify that the particle carries an EpsNonlocal cover layer.')


if __name__ == '__main__':
    main()
