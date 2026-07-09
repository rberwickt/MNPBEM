import os
import sys
import time

from typing import Tuple

import numpy as np

MNPBEM_ROOT = '/home/yoojk20/workspace/MNPBEM'
sys.path.insert(0, MNPBEM_ROOT)
sys.path.insert(0, os.path.join(MNPBEM_ROOT, 'validation'))

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.bem import BEMStat, BEMRet
from mnpbem.simulation import EELSStat, EELSRet

from _common import save_csv, load_csv, save_timing, plot_comparison


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')

IMPACT = np.array([[15.0, 0.0]])
WIDTH = 0.5
VEL = 0.5


def run_stat(p: ComParticle, wls: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    bem = BEMStat(p)
    eels = EELSStat(p, impact = IMPACT, width = WIDTH, vel = VEL)
    n = len(wls)
    psurf = np.zeros(n); pbulk = np.zeros(n)
    t0 = time.time()
    for i, en in enumerate(wls):
        exc = eels(p, en)
        sig, _ = bem.solve(exc)
        ps, pb = eels.loss(sig)
        psurf[i] = float(np.real(np.ravel(ps)[0]))
        pbulk[i] = float(np.real(np.ravel(pb)[0]))
    return psurf, pbulk, time.time() - t0


def run_ret(p: ComParticle, wls: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    bem = BEMRet(p)
    eels = EELSRet(p, impact = IMPACT, width = WIDTH, vel = VEL)
    n = len(wls)
    psurf = np.zeros(n); pbulk = np.zeros(n)
    t0 = time.time()
    for i, en in enumerate(wls):
        exc = eels(p, en)
        sig, _ = bem.solve(exc)
        ps, pb = eels.loss(sig)
        psurf[i] = float(np.real(np.ravel(ps)[0]))
        pbulk[i] = float(np.real(np.ravel(pb)[0]))
    return psurf, pbulk, time.time() - t0


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok = True)
    os.makedirs(FIG_DIR, exist_ok = True)

    wls = np.linspace(450, 650, 21)
    timings = {}

    print('[info] === 11_eels / sphere (Python) ===')
    print('[info] EELS, 20nm Au sphere, impact=[15,0], width=0.5, vel=0.5')

    epstab = [EpsConst(1.0), EpsTable('gold.dat')]
    p = ComParticle(epstab, [trisphere(144, 20.0)], [[2, 1]], [1])

    ps_s, pb_s, t_s = run_stat(p, wls)
    timings['stat'] = t_s
    save_csv(os.path.join(DATA_DIR, 'stat_python.csv'),
        wls, [ps_s, pb_s], ['psurf', 'pbulk'])
    print('[info] stat {:.3f}s'.format(t_s))

    ps_r, pb_r, t_r = run_ret(p, wls)
    timings['ret'] = t_r
    save_csv(os.path.join(DATA_DIR, 'ret_python.csv'),
        wls, [ps_r, pb_r], ['psurf', 'pbulk'])
    print('[info] ret  {:.3f}s'.format(t_r))

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
        max_rms, rms = plot_comparison(wls, [ps_s, pb_s],
            [ml_s[:, 1], ml_s[:, 2]], ['psurf', 'pbulk'],
            'EELS stat sphere comparison',
            os.path.join(FIG_DIR, 'stat_comparison.png'),
            t_py = t_s, t_ml = ml_timings.get('stat', 0),
            ylabel = 'loss probability')
        print('[info] stat comparison: max RMS = {:.2e}'.format(max_rms))

    ml_r = load_csv(os.path.join(DATA_DIR, 'ret_matlab.csv'))
    if ml_r is not None:
        max_rms, rms = plot_comparison(wls, [ps_r, pb_r],
            [ml_r[:, 1], ml_r[:, 2]], ['psurf', 'pbulk'],
            'EELS ret sphere comparison',
            os.path.join(FIG_DIR, 'ret_comparison.png'),
            t_py = t_r, t_ml = ml_timings.get('ret', 0),
            ylabel = 'loss probability')
        print('[info] ret comparison: max RMS = {:.2e}'.format(max_rms))


if __name__ == '__main__':
    main()
