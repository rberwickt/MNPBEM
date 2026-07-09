import os
import sys
import time

from typing import List, Dict, Tuple, Optional

import numpy as np

MNPBEM_ROOT = '/home/yoojk20/workspace/MNPBEM'
sys.path.insert(0, MNPBEM_ROOT)
sys.path.insert(0, os.path.join(MNPBEM_ROOT, 'validation'))

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.mie import MieStat, MieRet, MieGans

from _common import (save_csv, load_csv, save_timing, plot_spectrum, plot_comparison)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')


def run_miestat(enei: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    epsin = EpsTable('gold.dat')
    epsout = EpsConst(1.0)
    mie = MieStat(epsin, epsout, diameter = 20)
    t0 = time.time()
    ext = mie.extinction(enei)
    sca = mie.scattering(enei)
    absc = mie.absorption(enei)
    return ext, sca, absc, time.time() - t0


def run_mieret(enei: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    epsin = EpsTable('gold.dat')
    epsout = EpsConst(1.0)
    mie = MieRet(epsin, epsout, diameter = 100)
    t0 = time.time()
    ext = mie.extinction(enei)
    sca = mie.scattering(enei)
    absc = mie.absorption(enei)
    return ext, sca, absc, time.time() - t0


def run_miegans(enei: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    epsin = EpsTable('gold.dat')
    epsout = EpsConst(1.0)
    mie = MieGans(epsin, epsout, ax = np.array([20.0, 10.0, 10.0]))
    t0 = time.time()
    ext_x = mie.extinction(enei, pol = np.array([1.0, 0.0, 0.0]))
    ext_z = mie.extinction(enei, pol = np.array([0.0, 0.0, 1.0]))
    return ext_x, ext_z, time.time() - t0


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok = True)
    os.makedirs(FIG_DIR, exist_ok = True)

    enei = np.linspace(400, 800, 41)
    timings = {}

    print('[info] === 01_mie / sphere (Python) ===')

    print('[info] MieStat: 20nm Au sphere')
    ext_s, sca_s, abs_s, t_s = run_miestat(enei)
    timings['miestat'] = t_s
    save_csv(os.path.join(DATA_DIR, 'miestat_python.csv'),
        enei, [ext_s, sca_s, abs_s], ['extinction', 'scattering', 'absorption'])
    plot_spectrum(enei, [ext_s, sca_s, abs_s], ['ext', 'sca', 'abs'],
        'Python MieStat 20nm Au sphere (t={:.4f}s)'.format(t_s),
        os.path.join(FIG_DIR, 'miestat_python.png'))
    print('[info]   time = {:.4f} s'.format(t_s))

    print('[info] MieRet: 100nm Au sphere')
    ext_r, sca_r, abs_r, t_r = run_mieret(enei)
    timings['mieret'] = t_r
    save_csv(os.path.join(DATA_DIR, 'mieret_python.csv'),
        enei, [ext_r, sca_r, abs_r], ['extinction', 'scattering', 'absorption'])
    plot_spectrum(enei, [ext_r, sca_r, abs_r], ['ext', 'sca', 'abs'],
        'Python MieRet 100nm Au sphere (t={:.4f}s)'.format(t_r),
        os.path.join(FIG_DIR, 'mieret_python.png'))
    print('[info]   time = {:.4f} s'.format(t_r))

    print('[info] MieGans: [20,10,10]nm ellipsoid')
    ext_gx, ext_gz, t_g = run_miegans(enei)
    timings['miegans'] = t_g
    save_csv(os.path.join(DATA_DIR, 'miegans_python.csv'),
        enei, [ext_gx, ext_gz], ['extinction_xpol', 'extinction_zpol'])
    plot_spectrum(enei, [ext_gx, ext_gz], ['x-pol', 'z-pol'],
        'Python MieGans [20,10,10]nm ellipsoid (t={:.4f}s)'.format(t_g),
        os.path.join(FIG_DIR, 'miegans_python.png'))
    print('[info]   time = {:.4f} s'.format(t_g))

    save_timing(os.path.join(DATA_DIR, 'python_timing.csv'), timings)
    print('[info] total python = {:.4f} s'.format(sum(timings.values())))

    ml_timing_path = os.path.join(DATA_DIR, 'matlab_timing.csv')
    ml_timings = {}
    if os.path.exists(ml_timing_path):
        with open(ml_timing_path) as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    ml_timings[parts[0]] = float(parts[1])

    for case in ['miestat', 'mieret', 'miegans']:
        ml = load_csv(os.path.join(DATA_DIR, '{}_matlab.csv'.format(case)))
        py = load_csv(os.path.join(DATA_DIR, '{}_python.csv'.format(case)))
        if ml is None or py is None:
            print('[info] {}: MATLAB data not found, skip compare'.format(case))
            continue
        t_ml = ml_timings.get(case, 0.0)
        t_py = timings[case]
        ncol = py.shape[1] - 1
        col_names = ['ext', 'sca', 'abs'][:ncol] if case in ('miestat', 'mieret') else ['x-pol', 'z-pol']
        py_cols = [py[:, j + 1] for j in range(ncol)]
        ml_cols = [ml[:, j + 1] for j in range(ncol)]
        max_rms, rms = plot_comparison(enei, py_cols, ml_cols, col_names,
            '{} sphere comparison'.format(case),
            os.path.join(FIG_DIR, '{}_comparison.png'.format(case)),
            t_py = t_py, t_ml = t_ml)
        print('[info] {}: max RMS = {:.2e} ({} )'.format(case, max_rms,
            ', '.join(['{}={:.2e}'.format(k, v) for k, v in rms.items()])))


if __name__ == '__main__':
    main()
