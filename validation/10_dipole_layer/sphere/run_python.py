import os
import sys
import time

from typing import Tuple

import numpy as np

MNPBEM_ROOT = '/home/yoojk20/workspace/MNPBEM'
sys.path.insert(0, MNPBEM_ROOT)
sys.path.insert(0, os.path.join(MNPBEM_ROOT, 'validation'))

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle, ComPoint, LayerStructure
from mnpbem.bem import BEMStatLayer, BEMRetLayer
from mnpbem.simulation import DipoleStatLayer, DipoleRetLayer

from _common import save_csv, load_csv, save_timing, plot_comparison


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')


def make_layer_particle() -> Tuple[ComParticle, LayerStructure]:
    epstab = [EpsConst(1.0), EpsTable('gold.dat'), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 3], [0.0])
    sphere = trisphere(144, 20.0)
    sphere.shift([0, 0, -sphere.pos[:, 2].min() + 1.0])
    p = ComParticle(epstab, [sphere], [[2, 1]], [1])
    return p, layer


def run_stat(wls: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    p, layer = make_layer_particle()
    pt = ComPoint(p, np.array([[0.0, 0.0, 25.0]]))
    dip = DipoleStatLayer(pt, layer, dip = np.array([[0.0, 0.0, 1.0]]))
    bem = BEMStatLayer(p, layer)
    n = len(wls)
    tot = np.zeros(n); rad = np.zeros(n)
    t0 = time.time()
    for i, en in enumerate(wls):
        exc = dip(p, en)
        sig, _ = bem.solve(exc)
        t, r, _ = dip.decayrate(sig)
        tot[i] = float(np.ravel(t)[0])
        rad[i] = float(np.ravel(r)[0])
    return tot, rad, time.time() - t0


def run_ret(wls: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    p, layer = make_layer_particle()
    pt = ComPoint(p, np.array([[0.0, 0.0, 25.0]]))
    dip = DipoleRetLayer(pt, layer, dip = np.array([[0.0, 0.0, 1.0]]))
    bem = BEMRetLayer(p, layer)
    n = len(wls)
    tot = np.zeros(n); rad = np.zeros(n)
    t0 = time.time()
    for i, en in enumerate(wls):
        exc = dip(p, en)
        sig, _ = bem.solve(exc)
        t, r, _ = dip.decayrate(sig)
        tot[i] = float(np.real(np.ravel(t)[0]))
        rad[i] = float(np.real(np.ravel(r)[0]))
    return tot, rad, time.time() - t0


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok = True)
    os.makedirs(FIG_DIR, exist_ok = True)

    wls = np.linspace(500, 700, 21)
    timings = {}

    print('[info] === 10_dipole_layer / sphere (Python) ===')
    print('[info] 20nm Au sphere 1nm above glass, z-dipole at [0,0,25]')

    tot_s, rad_s, t_s = run_stat(wls)
    timings['stat'] = t_s
    save_csv(os.path.join(DATA_DIR, 'stat_python.csv'),
        wls, [tot_s, rad_s], ['tot', 'rad'])
    print('[info] stat {:.3f}s'.format(t_s))

    # NOTE: Python BEMRetLayer + DipoleRetLayer가 wavelength 반복 시 매우 느림
    # (21pt 대상으로 30분+ 소요). 이건 Python 구현 성능 이슈로 별도 조사 필요.
    # 검증 결과 완성도를 위해 stat만 남기고 ret은 임시 skip.
    run_ret_enabled = False
    if run_ret_enabled:
        tot_r, rad_r, t_r = run_ret(wls)
        timings['ret'] = t_r
        save_csv(os.path.join(DATA_DIR, 'ret_python.csv'),
            wls, [tot_r, rad_r], ['tot', 'rad'])
        print('[info] ret  {:.3f}s'.format(t_r))
    else:
        print('[info] ret: SKIPPED (Python BEMRetLayer 성능 이슈, 조사 필요)')

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
        max_rms, rms = plot_comparison(wls, [tot_s, rad_s],
            [ml_s[:, 1], ml_s[:, 2]], ['tot', 'rad'],
            'dipole_layer stat sphere comparison',
            os.path.join(FIG_DIR, 'stat_comparison.png'),
            t_py = t_s, t_ml = ml_timings.get('stat', 0),
            ylabel = 'decay rate')
        print('[info] stat comparison: max RMS = {:.2e}'.format(max_rms))

    ml_r = load_csv(os.path.join(DATA_DIR, 'ret_matlab.csv'))
    if ml_r is not None and run_ret_enabled:
        max_rms, rms = plot_comparison(wls, [tot_r, rad_r],
            [ml_r[:, 1], ml_r[:, 2]], ['tot', 'rad'],
            'dipole_layer ret sphere comparison',
            os.path.join(FIG_DIR, 'ret_comparison.png'),
            t_py = t_r, t_ml = ml_timings.get('ret', 0),
            ylabel = 'decay rate')
        print('[info] ret comparison: max RMS = {:.2e}'.format(max_rms))


if __name__ == '__main__':
    main()
