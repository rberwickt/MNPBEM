import os
import sys
import time

from typing import Dict, Tuple

import numpy as np

MNPBEM_ROOT = '/home/yoojk20/workspace/MNPBEM'
sys.path.insert(0, MNPBEM_ROOT)
sys.path.insert(0, os.path.join(MNPBEM_ROOT, 'validation'))

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.bem import BEMRet
from mnpbem.simulation import PlaneWaveRet
from mnpbem.mie import MieRet

from _common import save_csv, load_csv, save_timing, plot_spectrum, plot_comparison


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')


def run_bem(enei: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    epstab = [EpsConst(1.0), EpsTable('gold.dat')]
    sp = trisphere(144, 20)
    p = ComParticle(epstab, [sp], [[2, 1]], 1, interp = 'curv')
    bem = BEMRet(p)
    pol = np.array([[1.0, 0.0, 0.0]])
    dir_vec = np.array([[0.0, 0.0, 1.0]])
    exc = PlaneWaveRet(pol, dir_vec)
    n = len(enei)
    ext = np.zeros(n); sca = np.zeros(n)
    t0 = time.time()
    for i in range(n):
        pot = exc.potential(p, enei[i])
        sig, bem = bem.solve(pot)
        ext[i] = float(np.real(np.ravel(exc.extinction(sig))[0]))
        sca_val = exc.scattering(sig)
        sca[i] = float(np.real(np.ravel(sca_val[0] if isinstance(sca_val, tuple) else sca_val)[0]))
    absc = ext - sca
    return ext, sca, absc, time.time() - t0


def run_mie(enei: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mie = MieRet(EpsTable('gold.dat'), EpsConst(1.0), diameter = 20)
    t0 = time.time()
    ext = mie.extinction(enei)
    sca = mie.scattering(enei)
    absc = ext - sca
    return ext, sca, absc, time.time() - t0


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok = True)
    os.makedirs(FIG_DIR, exist_ok = True)

    enei = np.linspace(400, 800, 41)
    timings = {}

    print('[info] === 03_bemret / sphere (Python) ===')
    print('[info] BEMRet 20nm Au sphere, PlaneWaveRet([1,0,0],[0,0,1])')

    ext_b, sca_b, abs_b, t_bem = run_bem(enei)
    timings['bem'] = t_bem
    save_csv(os.path.join(DATA_DIR, 'bemret_python.csv'), enei,
        [ext_b, sca_b, abs_b], ['extinction', 'scattering', 'absorption'])
    plot_spectrum(enei, [ext_b, sca_b, abs_b], ['ext', 'sca', 'abs'],
        'Python BEMRet 20nm Au sphere (t={:.3f}s)'.format(t_bem),
        os.path.join(FIG_DIR, 'bemret_python.png'))
    print('[info]   BEM time = {:.4f} s'.format(t_bem))

    ext_m, sca_m, abs_m, t_mie = run_mie(enei)
    timings['mie'] = t_mie
    save_csv(os.path.join(DATA_DIR, 'mie_python.csv'), enei,
        [ext_m, sca_m, abs_m], ['extinction', 'scattering', 'absorption'])
    print('[info]   Mie time = {:.4f} s'.format(t_mie))

    save_timing(os.path.join(DATA_DIR, 'python_timing.csv'), timings)

    ml_timing_path = os.path.join(DATA_DIR, 'matlab_timing.csv')
    ml_timings = {}
    if os.path.exists(ml_timing_path):
        with open(ml_timing_path) as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    ml_timings[parts[0]] = float(parts[1])

    ml_bem = load_csv(os.path.join(DATA_DIR, 'bemret_matlab.csv'))
    if ml_bem is not None:
        t_ml = ml_timings.get('bem', 0.0)
        py_cols = [ext_b, sca_b, abs_b]
        ml_cols = [ml_bem[:, 1], ml_bem[:, 2], ml_bem[:, 3]]
        max_rms, rms = plot_comparison(enei, py_cols, ml_cols,
            ['ext', 'sca', 'abs'],
            'BEMRet sphere comparison',
            os.path.join(FIG_DIR, 'bemret_comparison.png'),
            t_py = t_bem, t_ml = t_ml)
        print('[info] BEM comparison: max RMS = {:.2e} ({})'.format(max_rms,
            ', '.join(['{}={:.2e}'.format(k, v) for k, v in rms.items()])))
    else:
        print('[info] MATLAB data not found, skip compare')


if __name__ == '__main__':
    main()
