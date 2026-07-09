import os
import sys
import time

from typing import Dict, Tuple

import numpy as np

MNPBEM_ROOT = '/home/yoojk20/workspace/MNPBEM'
sys.path.insert(0, MNPBEM_ROOT)
sys.path.insert(0, os.path.join(MNPBEM_ROOT, 'validation'))

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trirod, ComParticle
from mnpbem.bem import BEMRet
from mnpbem.simulation import PlaneWaveRet

from _common import save_csv, load_csv, save_timing, plot_spectrum, plot_comparison


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')


def run_bem(enei: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    epstab = [EpsConst(1.0), EpsTable('gold.dat')]
    rod = trirod(10.0, 40.0, n = [15, 15, 15])
    p = ComParticle(epstab, [rod], [[2, 1]], 1, interp = 'curv')
    bem = BEMRet(p)
    pol = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    dir_vec = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    exc = PlaneWaveRet(pol, dir_vec)
    n = len(enei)
    ext_x = np.zeros(n); sca_x = np.zeros(n)
    ext_z = np.zeros(n); sca_z = np.zeros(n)
    t0 = time.time()
    for i in range(n):
        pot = exc.potential(p, enei[i])
        sig, bem = bem.solve(pot)
        e = np.real(np.ravel(exc.extinction(sig)))
        sca_val = exc.scattering(sig)
        s = np.real(np.ravel(sca_val[0] if isinstance(sca_val, tuple) else sca_val))
        ext_x[i], ext_z[i] = float(e[0]), float(e[1])
        sca_x[i], sca_z[i] = float(s[0]), float(s[1])
    return ext_x, sca_x, ext_z, sca_z, time.time() - t0


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok = True)
    os.makedirs(FIG_DIR, exist_ok = True)

    enei = np.linspace(400, 800, 41)
    timings = {}

    print('[info] === 03_bemret / rod (Python) ===')
    print('[info] BEMRet trirod(10, 40, [15,15,15]) Au, x-pol + z-pol')

    ext_x, sca_x, ext_z, sca_z, t_bem = run_bem(enei)
    timings['bem'] = t_bem
    save_csv(os.path.join(DATA_DIR, 'bemret_python.csv'), enei,
        [ext_x, sca_x, ext_z, sca_z],
        ['extinction_xpol', 'scattering_xpol', 'extinction_zpol', 'scattering_zpol'])
    plot_spectrum(enei, [ext_x, ext_z, sca_x, sca_z],
        ['ext xpol', 'ext zpol', 'sca xpol', 'sca zpol'],
        'Python BEMRet rod (t={:.3f}s)'.format(t_bem),
        os.path.join(FIG_DIR, 'bemret_python.png'))
    print('[info]   BEM time = {:.4f} s'.format(t_bem))

    save_timing(os.path.join(DATA_DIR, 'python_timing.csv'), timings)

    ml_timing_path = os.path.join(DATA_DIR, 'matlab_timing.csv')
    ml_timings = {}
    if os.path.exists(ml_timing_path):
        with open(ml_timing_path) as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    ml_timings[parts[0]] = float(parts[1])

    ml = load_csv(os.path.join(DATA_DIR, 'bemret_matlab.csv'))
    if ml is not None:
        t_ml = ml_timings.get('bem', 0.0)
        col_names = ['ext_x', 'sca_x', 'ext_z', 'sca_z']
        py_cols = [ext_x, sca_x, ext_z, sca_z]
        ml_cols = [ml[:, 1], ml[:, 2], ml[:, 3], ml[:, 4]]
        max_rms, rms = plot_comparison(enei, py_cols, ml_cols, col_names,
            'BEMRet rod comparison',
            os.path.join(FIG_DIR, 'bemret_comparison.png'),
            t_py = t_bem, t_ml = t_ml)
        print('[info] BEM comparison: max RMS = {:.2e} ({})'.format(max_rms,
            ', '.join(['{}={:.2e}'.format(k, v) for k, v in rms.items()])))
    else:
        print('[info] MATLAB data not found, skip compare')


if __name__ == '__main__':
    main()
