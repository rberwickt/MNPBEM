import os
import sys
import time
import resource

from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mnpbem.materials import EpsConst
from mnpbem.geometry import ComParticle
from mnpbem.geometry.mesh_generators import _trisphere_fibonacci, tricube
from mnpbem.bem import BEMRetIter
from mnpbem.simulation.planewave_ret import PlaneWaveRet


# CUSTOMIZE: Mesh face count. Lane E2 baseline is ~25k; we use a smaller
# default here so the script can run inside CI and lightweight test
# environments. Pass a positive integer via env var LANE_E2_NFACES to
# override (e.g. 25344 for the full Lane E2 size).
N_FACE = int(os.environ.get('LANE_E2_NFACES', '5120'))
SHAPE = os.environ.get('LANE_E2_SHAPE', 'fib')   # 'fib' or 'cube'
DIAM = 30.0
ENEI = 636.36


def _make_sphere(n_face: int) -> ComParticle:

    eps_b = EpsConst(1.0)
    eps_m = EpsConst(-10.0 + 0.5j)
    if SHAPE == 'cube':
        # tricube approx: 6 * 2 * (n-1)^2 faces.
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
        hmatrix: bool) -> None:

    print('\n=== {} (faces={}, hmatrix={}) ==='.format(label, cp.nfaces, hmatrix))

    pol = np.array([[1.0, 0.0, 0.0]])
    dirn = np.array([[0.0, 0.0, 1.0]])
    exc = PlaneWaveRet(pol = pol, dir = dirn)

    t0 = time.time()
    if hmatrix:
        bem = BEMRetIter(cp, hmatrix = True, htol = 1e-6,
                kmax = [4, 100], cleaf = 64, tol = 1e-5, maxit = 400)
    else:
        bem = BEMRetIter(cp, tol = 1e-5, maxit = 400)
    t_init = time.time() - t0

    exc_struct = exc(cp, ENEI)

    t0 = time.time()
    sig, bem_done = bem.solve(exc_struct)
    t_solve = time.time() - t0

    print('[info] init   wall: {:.2f}s'.format(t_init))
    print('[info] solve  wall: {:.2f}s'.format(t_solve))
    print('[info] |sig1| max  : {:.4e}'.format(np.abs(sig.sig1).max()))
    print('[info] peak RSS    : {:.1f} MB'.format(_peak_rss_mb()))
    flags, relres, iter_info = bem_done.info()
    if relres:
        print('[info] gmres relres: {:.3e}, flag={}'.format(relres[0], flags[0]))

    if hmatrix and hasattr(bem.g, '_cache'):
        # Report H-matrix compression for the first cached G block.
        for cache_key, hmat in bem.g._cache.items():
            if hasattr(hmat, 'compression'):
                comp = hmat.compression()
                print('[info] {} compression: {:.4f}'.format(cache_key, comp))
                break


def main() -> None:

    print('[info] Lane E2 BEMRetIter H-matrix benchmark')
    print('[info] N_FACE={}, diameter={} nm, enei={} nm'.format(N_FACE, DIAM, ENEI))

    cp = _make_sphere(N_FACE)
    print('[info] mesh built: nfaces={}'.format(cp.nfaces))

    # H-matrix path (the focus of Lane E2).
    _bench('hmatrix', cp, hmatrix = True)


if __name__ == '__main__':
    main()
