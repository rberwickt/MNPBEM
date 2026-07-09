import os
import sys
import time
import resource

from typing import Any, Tuple

import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mnpbem.materials import EpsConst
from mnpbem.geometry import ComParticle
from mnpbem.geometry.mesh_generators import _trisphere_fibonacci, tricube
from mnpbem.bem import BEMRetIter
from mnpbem.simulation.planewave_ret import PlaneWaveRet


# CUSTOMIZE: Mesh face count. Defaults to 5120 for fast smoke runs; pass
# LANE_E2_NFACES=25344 (or similar) for the full-size benchmark.
N_FACE = int(os.environ.get('LANE_E2_NFACES', '5120'))
SHAPE = os.environ.get('LANE_E2_SHAPE', 'fib')
DIAM = 30.0
ENEI = 636.36


def _make_sphere(n_face: int) -> ComParticle:

    eps_b = EpsConst(1.0)
    eps_m = EpsConst(-10.0 + 0.5j)
    if SHAPE == 'cube':
        n_per_edge = max(int(np.ceil(np.sqrt(n_face / 12.0))) + 1, 4)
        p = tricube(n_per_edge, DIAM, e = 0.25)
    else:
        p = _trisphere_fibonacci(max(n_face // 2, 64), DIAM)
    cp = ComParticle([eps_b, eps_m], [p], [[2, 1]])
    return cp


def _peak_rss_mb() -> float:

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _bench(label: str,
        cp: ComParticle,
        precond: str) -> Tuple[int, float, float, float]:

    print('\n=== {} (faces={}, preconditioner={}) ==='.format(label, cp.nfaces, precond))

    pol = np.array([[1.0, 0.0, 0.0]])
    dirn = np.array([[0.0, 0.0, 1.0]])
    exc = PlaneWaveRet(pol = pol, dir = dirn)

    t0 = time.time()
    bem = BEMRetIter(cp, hmatrix = True, htol = 1e-6,
            kmax = [4, 100], cleaf = 64, tol = 1e-5, maxit = 600,
            preconditioner = precond)
    t_init = time.time() - t0

    exc_struct = exc(cp, ENEI)

    # Build everything once (matrices, optional preconditioner) so that the
    # GMRES timing isolates the iterations themselves.
    bem._init_matrices(ENEI)
    phi, a, De, alpha = bem._excitation(exc_struct)
    b = bem._pack(phi, a, De, alpha)
    N = b.shape[0]

    fm = None
    if precond != 'none':
        t0 = time.time()
        fm = bem._build_hlu_preconditioner(N)
        t_precond = time.time() - t0
    else:
        t_precond = 0.0

    A_op = LinearOperator((N, N), matvec = bem._afun, dtype = complex)
    M_op = LinearOperator((N, N), matvec = fm, dtype = complex) if fm is not None else None

    iter_count = [0]

    def _cb(rk: Any) -> None:
        iter_count[0] += 1

    t0 = time.time()
    x, info = gmres(A_op, b, rtol = 1e-5, maxiter = 600, restart = 30,
            callback = _cb, callback_type = 'legacy', M = M_op)
    t_gmres = time.time() - t0

    print('[info] init      wall: {:.2f}s'.format(t_init))
    print('[info] precond   wall: {:.2f}s'.format(t_precond))
    print('[info] gmres     wall: {:.2f}s   iters={}, info={}'.format(t_gmres, iter_count[0], info))
    print('[info] peak RSS       : {:.1f} MB'.format(_peak_rss_mb()))

    return iter_count[0], t_init, t_precond, t_gmres


def main() -> None:

    print('[info] v1.5.0 alpha — H-matrix LU preconditioner Lane E2 benchmark')
    print('[info] N_FACE={}, diameter={} nm, enei={} nm'.format(N_FACE, DIAM, ENEI))

    cp = _make_sphere(N_FACE)
    print('[info] mesh built: nfaces={}'.format(cp.nfaces))

    # Baseline (v1.3.0 behaviour): no preconditioner on the H-matrix path.
    it_none, _, _, t_none = _bench('baseline (none)', cp, 'none')

    # v1.5.0 default ('auto').
    it_auto, _, t_pre, t_auto = _bench('v1.5.0 (auto)', cp, 'auto')

    print('\n=== summary (faces={}) ==='.format(cp.nfaces))
    print('  none      iters={:4d}  wall={:.2f}s'.format(it_none, t_none))
    print('  auto      iters={:4d}  wall={:.2f}s (precond build {:.2f}s)'.format(
            it_auto, t_auto, t_pre))
    if it_auto > 0 and it_none > 0:
        print('  iter ratio (none/auto)  : {:.2f}x'.format(it_none / it_auto))
        print('  wall ratio (none/auto)  : {:.2f}x'.format(t_none / (t_auto + t_pre)))


if __name__ == '__main__':
    main()
