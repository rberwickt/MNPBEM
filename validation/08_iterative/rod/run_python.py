import os
import sys
import time

from typing import Tuple

import numpy as np

MNPBEM_ROOT = '/home/yoojk20/workspace/MNPBEM'
sys.path.insert(0, MNPBEM_ROOT)
sys.path.insert(0, os.path.join(MNPBEM_ROOT, 'validation'))

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trirod, ComParticle
from mnpbem.bem import BEMStat, BEMRet, BEMStatIter, BEMRetIter
from mnpbem.simulation import PlaneWaveStat, PlaneWaveRet

from _common import save_csv, load_csv, save_timing, plot_comparison


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')


def setup() -> ComParticle:
    epstab = [EpsConst(1), EpsTable('gold.dat')]
    return ComParticle(epstab, [trirod(10.0, 40.0, n = [15, 15, 15])], [[2, 1]], 1, interp = 'curv')


def run_bem_stat(p: ComParticle, wls: np.ndarray, iterative: bool) -> Tuple[np.ndarray, float]:
    bem = BEMStatIter(p) if iterative else BEMStat(p)
    exc = PlaneWaveStat([1, 0, 0])
    n = len(wls)
    ext = np.zeros(n)
    t0 = time.time()
    for i, en in enumerate(wls):
        sig, bem = bem.solve(exc(p, en))
        ext[i] = exc.extinction(sig)
    return ext, time.time() - t0


def run_bem_ret(p: ComParticle, wls: np.ndarray, iterative: bool) -> Tuple[np.ndarray, float]:
    bem = BEMRetIter(p) if iterative else BEMRet(p)
    exc = PlaneWaveRet(np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.0]]))
    n = len(wls)
    ext = np.zeros(n)
    t0 = time.time()
    for i, en in enumerate(wls):
        pot = exc.potential(p, en)
        sig, bem = bem.solve(pot)
        ext[i] = float(np.real(np.ravel(exc.extinction(sig))[0]))
    return ext, time.time() - t0


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok = True)
    os.makedirs(FIG_DIR, exist_ok = True)

    wls = np.linspace(400, 800, 41)
    timings = {}

    print('[info] === 08_iterative / rod (Python) ===')

    p = setup()

    print('[info] BEMStat vs BEMStatIter')
    ext_sd, t_sd = run_bem_stat(p, wls, False)
    ext_si, t_si = run_bem_stat(p, wls, True)
    timings['stat_direct'] = t_sd
    timings['stat_iter'] = t_si
    print('[info]   direct {:.3f}s, iter {:.3f}s'.format(t_sd, t_si))

    print('[info] BEMRet vs BEMRetIter')
    ext_rd, t_rd = run_bem_ret(p, wls, False)
    ext_ri, t_ri = run_bem_ret(p, wls, True)
    timings['ret_direct'] = t_rd
    timings['ret_iter'] = t_ri
    print('[info]   direct {:.3f}s, iter {:.3f}s'.format(t_rd, t_ri))

    save_csv(os.path.join(DATA_DIR, 'stat_python.csv'),
        wls, [ext_sd, ext_si], ['ext_direct', 'ext_iter'])
    save_csv(os.path.join(DATA_DIR, 'ret_python.csv'),
        wls, [ext_rd, ext_ri], ['ext_direct', 'ext_iter'])
    save_timing(os.path.join(DATA_DIR, 'python_timing.csv'), timings)

    ml_timing_path = os.path.join(DATA_DIR, 'matlab_timing.csv')
    ml_timings = {}
    if os.path.exists(ml_timing_path):
        with open(ml_timing_path) as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    ml_timings[parts[0]] = float(parts[1])

    ml_s = load_csv(os.path.join(DATA_DIR, 'stat_matlab.csv'))
    if ml_s is not None:
        max_rms, rms = plot_comparison(wls,
            [ext_sd, ext_si], [ml_s[:, 1], ml_s[:, 2]],
            ['direct', 'iter'],
            'stat iterative rod comparison',
            os.path.join(FIG_DIR, 'stat_comparison.png'),
            t_py = t_sd + t_si,
            t_ml = ml_timings.get('stat_direct', 0) + ml_timings.get('stat_iter', 0))
        print('[info] stat comparison: max RMS = {:.2e}'.format(max_rms))

    ml_r = load_csv(os.path.join(DATA_DIR, 'ret_matlab.csv'))
    if ml_r is not None:
        max_rms, rms = plot_comparison(wls,
            [ext_rd, ext_ri], [ml_r[:, 1], ml_r[:, 2]],
            ['direct', 'iter'],
            'ret iterative rod comparison',
            os.path.join(FIG_DIR, 'ret_comparison.png'),
            t_py = t_rd + t_ri,
            t_ml = ml_timings.get('ret_direct', 0) + ml_timings.get('ret_iter', 0))
        print('[info] ret comparison: max RMS = {:.2e}'.format(max_rms))


if __name__ == '__main__':
    main()
