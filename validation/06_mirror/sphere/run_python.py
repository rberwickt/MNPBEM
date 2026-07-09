import os
import sys
import time

from typing import Tuple

import numpy as np

MNPBEM_ROOT = '/home/yoojk20/workspace/MNPBEM'
sys.path.insert(0, MNPBEM_ROOT)
sys.path.insert(0, os.path.join(MNPBEM_ROOT, 'validation'))

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trispheresegment, ComParticleMirror
from mnpbem.bem import BEMStat, BEMRet
from mnpbem.bem.bem_stat_mirror import BEMStatMirror
from mnpbem.bem.bem_ret_mirror import BEMRetMirror
from mnpbem.simulation import PlaneWaveStat, PlaneWaveRet
from mnpbem.simulation.planewave_stat_mirror import PlaneWaveStatMirror
from mnpbem.simulation.planewave_ret_mirror import PlaneWaveRetMirror

from _common import save_csv, load_csv, save_timing, plot_spectrum, plot_comparison


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')


def make_mirror_sphere() -> ComParticleMirror:
    epstab = [EpsConst(1.0), EpsTable('gold.dat')]
    n = 13
    phi = np.linspace(0, np.pi / 2, n)
    theta = np.linspace(0, np.pi, 2 * n - 1)
    seg = trispheresegment(phi, theta, diameter = 20.0)
    return ComParticleMirror(epstab, [seg], [[2, 1]], sym = 'xy', closed_args = (1,))


def run_stat(p_mir: ComParticleMirror, p_full, wls: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    pol3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype = float)

    bem_m = BEMStatMirror(p_mir)
    exc_m = PlaneWaveStatMirror(pol3)
    n_wl = len(wls)
    ext_m = np.empty((n_wl, 3))
    t0 = time.perf_counter()
    for i, en in enumerate(wls):
        pot = exc_m.potential(p_mir, en)
        sig, bem_m = bem_m.solve(pot)
        ext_m[i] = exc_m.extinction(sig)
    t_m = time.perf_counter() - t0

    bem_f = BEMStat(p_full)
    exc_f = PlaneWaveStat(pol3)
    ext_f = np.empty((n_wl, 3))
    t0 = time.perf_counter()
    for i, en in enumerate(wls):
        pot = exc_f.potential(p_full, en)
        sig, bem_f = bem_f.solve(pot)
        ext_f[i] = exc_f.extinction(sig)
    t_f = time.perf_counter() - t0

    return ext_m, ext_f, t_m, t_f


def run_ret(p_mir: ComParticleMirror, p_full, wls: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    pol2 = np.array([[1, 0, 0], [0, 1, 0]], dtype = float)
    dir2 = np.array([[0, 0, 1], [0, 0, 1]], dtype = float)

    bem_m = BEMRetMirror(p_mir)
    exc_m = PlaneWaveRetMirror(pol2, dir2)
    n_wl = len(wls)
    ext_m = np.empty((n_wl, 2))
    t0 = time.perf_counter()
    for i, en in enumerate(wls):
        pot = exc_m.potential(p_mir, en)
        sig, bem_m = bem_m.solve(pot)
        v = exc_m.scattering(sig)
        v_arr = v[0] if isinstance(v, tuple) else v
        ext_m[i] = np.real(np.ravel(v_arr)[:2])
    t_m = time.perf_counter() - t0

    bem_f = BEMRet(p_full)
    exc_f = PlaneWaveRet(pol2, dir2)
    ext_f = np.empty((n_wl, 2))
    t0 = time.perf_counter()
    for i, en in enumerate(wls):
        pot = exc_f.potential(p_full, en)
        sig, bem_f = bem_f.solve(pot)
        v = exc_f.scattering(sig)
        v_arr = v[0] if isinstance(v, tuple) else v
        ext_f[i] = np.real(np.ravel(v_arr)[:2])
    t_f = time.perf_counter() - t0

    return ext_m, ext_f, t_m, t_f


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok = True)
    os.makedirs(FIG_DIR, exist_ok = True)

    wls = np.linspace(400, 800, 41)
    timings = {}

    print('[info] === 06_mirror / sphere (Python) ===')
    print('[info] trispheresegment 20nm Au + sym=xy, Mirror vs Full')

    p_mir = make_mirror_sphere()
    p_full = p_mir.full()
    print('[info]   mirror nfaces (1/4): {}'.format(p_mir.nfaces))
    print('[info]   full nfaces: {}'.format(p_full.nfaces))

    print('[info] Quasistatic mirror vs full')
    ext_sm, ext_sf, t_sm, t_sf = run_stat(p_mir, p_full, wls)
    timings['stat_mirror'] = t_sm
    timings['stat_full'] = t_sf
    save_csv(os.path.join(DATA_DIR, 'stat_mirror_python.csv'),
        wls, [ext_sm[:, 0], ext_sm[:, 1], ext_sm[:, 2]], ['ext_x', 'ext_y', 'ext_z'])
    save_csv(os.path.join(DATA_DIR, 'stat_full_python.csv'),
        wls, [ext_sf[:, 0], ext_sf[:, 1], ext_sf[:, 2]], ['ext_x', 'ext_y', 'ext_z'])
    print('[info]   stat mirror {:.3f}s, full {:.3f}s, speedup {:.2f}x'.format(
        t_sm, t_sf, t_sf / t_sm if t_sm > 0 else 0))

    print('[info] Retarded mirror vs full')
    ext_rm, ext_rf, t_rm, t_rf = run_ret(p_mir, p_full, wls)
    timings['ret_mirror'] = t_rm
    timings['ret_full'] = t_rf
    save_csv(os.path.join(DATA_DIR, 'ret_mirror_python.csv'),
        wls, [ext_rm[:, 0], ext_rm[:, 1]], ['ext_x', 'ext_y'])
    save_csv(os.path.join(DATA_DIR, 'ret_full_python.csv'),
        wls, [ext_rf[:, 0], ext_rf[:, 1]], ['ext_x', 'ext_y'])
    print('[info]   ret mirror {:.3f}s, full {:.3f}s, speedup {:.2f}x'.format(
        t_rm, t_rf, t_rf / t_rm if t_rm > 0 else 0))

    save_timing(os.path.join(DATA_DIR, 'python_timing.csv'), timings)

    plot_spectrum(wls, [ext_sm[:, 0], ext_sm[:, 1], ext_sm[:, 2]],
        ['mirror x', 'mirror y', 'mirror z'],
        'Python stat mirror sphere',
        os.path.join(FIG_DIR, 'stat_mirror_python.png'))
    plot_spectrum(wls, [ext_rm[:, 0], ext_rm[:, 1]],
        ['mirror x', 'mirror y'],
        'Python ret mirror sphere',
        os.path.join(FIG_DIR, 'ret_mirror_python.png'))

    ml_timing_path = os.path.join(DATA_DIR, 'matlab_timing.csv')
    ml_timings = {}
    if os.path.exists(ml_timing_path):
        with open(ml_timing_path) as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    ml_timings[parts[0]] = float(parts[1])

    ml_sm = load_csv(os.path.join(DATA_DIR, 'stat_mirror_matlab.csv'))
    if ml_sm is not None:
        t_ml = ml_timings.get('stat_mirror', 0.0)
        max_rms, rms = plot_comparison(wls,
            [ext_sm[:, 0], ext_sm[:, 1], ext_sm[:, 2]],
            [ml_sm[:, 1], ml_sm[:, 2], ml_sm[:, 3]],
            ['ext_x', 'ext_y', 'ext_z'],
            'stat mirror sphere comparison',
            os.path.join(FIG_DIR, 'stat_mirror_comparison.png'),
            t_py = t_sm, t_ml = t_ml)
        print('[info] stat mirror comparison: max RMS = {:.2e}'.format(max_rms))

    ml_rm = load_csv(os.path.join(DATA_DIR, 'ret_mirror_matlab.csv'))
    if ml_rm is not None:
        t_ml = ml_timings.get('ret_mirror', 0.0)
        max_rms, rms = plot_comparison(wls,
            [ext_rm[:, 0], ext_rm[:, 1]],
            [ml_rm[:, 1], ml_rm[:, 2]],
            ['ext_x', 'ext_y'],
            'ret mirror sphere comparison',
            os.path.join(FIG_DIR, 'ret_mirror_comparison.png'),
            t_py = t_rm, t_ml = t_ml)
        print('[info] ret mirror comparison: max RMS = {:.2e}'.format(max_rms))


if __name__ == '__main__':
    main()
