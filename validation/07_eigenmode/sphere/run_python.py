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
from mnpbem.bem import BEMStat, BEMStatEig
from mnpbem.simulation import PlaneWaveStat

from _common import save_csv, load_csv, save_timing, plot_comparison


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')


def setup() -> ComParticle:
    epstab = [EpsConst(1.0), EpsTable('gold.dat')]
    sphere = trisphere(144, 20.0)
    return ComParticle(epstab, [sphere], [[2, 1]], 1, interp = 'curv')


def run_eig(p: ComParticle, wls: np.ndarray, nev: int = 20) -> Tuple[np.ndarray, float]:
    bem = BEMStatEig(p, nev = nev)
    exc = PlaneWaveStat([1, 0, 0])
    n = len(wls)
    ext = np.zeros(n)
    t0 = time.time()
    for i, en in enumerate(wls):
        sig, bem = bem.solve(exc(p, en))
        ext[i] = exc.extinction(sig)
    return ext, time.time() - t0


def run_dir(p: ComParticle, wls: np.ndarray) -> Tuple[np.ndarray, float]:
    bem = BEMStat(p)
    exc = PlaneWaveStat([1, 0, 0])
    n = len(wls)
    ext = np.zeros(n)
    t0 = time.time()
    for i, en in enumerate(wls):
        sig, bem = bem.solve(exc(p, en))
        ext[i] = exc.extinction(sig)
    return ext, time.time() - t0


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok = True)
    os.makedirs(FIG_DIR, exist_ok = True)

    wls = np.linspace(400, 800, 41)
    timings = {}

    print('[info] === 07_eigenmode / sphere (Python) ===')
    print('[info] BEMStatEig(nev=20) vs BEMStat direct, trisphere(144,20) Au')

    p = setup()

    ext_e, t_e = run_eig(p, wls, nev = 20)
    ext_d, t_d = run_dir(p, wls)
    timings['eig'] = t_e
    timings['dir'] = t_d
    print('[info]   eig {:.3f}s, direct {:.3f}s'.format(t_e, t_d))

    save_csv(os.path.join(DATA_DIR, 'eig_python.csv'), wls, [ext_e], ['extinction'])
    save_csv(os.path.join(DATA_DIR, 'dir_python.csv'), wls, [ext_d], ['extinction'])
    save_timing(os.path.join(DATA_DIR, 'python_timing.csv'), timings)

    ml_timing_path = os.path.join(DATA_DIR, 'matlab_timing.csv')
    ml_timings = {}
    if os.path.exists(ml_timing_path):
        with open(ml_timing_path) as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    ml_timings[parts[0]] = float(parts[1])

    ml_e = load_csv(os.path.join(DATA_DIR, 'eig_matlab.csv'))
    ml_d = load_csv(os.path.join(DATA_DIR, 'dir_matlab.csv'))
    if ml_e is not None and ml_d is not None:
        max_rms, rms = plot_comparison(wls,
            [ext_e, ext_d], [ml_e[:, 1], ml_d[:, 1]],
            ['ext_eig', 'ext_direct'],
            'eigenmode sphere comparison',
            os.path.join(FIG_DIR, 'eigenmode_comparison.png'),
            t_py = t_e + t_d, t_ml = ml_timings.get('eig', 0) + ml_timings.get('dir', 0))
        print('[info] comparison: max RMS = {:.2e}'.format(max_rms))
        rms_eig = float(np.max(np.abs(ext_e - ext_d) / (np.abs(ext_d) + 1e-30)))
        print('[info] self-consistency eig vs direct: {:.2e}'.format(rms_eig))


if __name__ == '__main__':
    main()
