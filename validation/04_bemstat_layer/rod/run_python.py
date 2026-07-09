import os
import sys
import time

from typing import Tuple

import numpy as np

MNPBEM_ROOT = '/home/yoojk20/workspace/MNPBEM'
sys.path.insert(0, MNPBEM_ROOT)
sys.path.insert(0, os.path.join(MNPBEM_ROOT, 'validation'))

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trirod, ComParticle, LayerStructure
from mnpbem.bem import BEMStatLayer
from mnpbem.simulation import PlaneWaveStatLayer

from _common import save_csv, load_csv, save_timing, plot_spectrum, plot_comparison


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')


def setup() -> Tuple[ComParticle, LayerStructure]:
    epstab = [EpsConst(1.0), EpsTable('gold.dat'), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 3], [0.0])
    rod = trirod(10.0, 40.0, n = [15, 15, 15])
    rod.shift([0, 0, -rod.pos[:, 2].min() + 1])
    p = ComParticle(epstab, [rod], [[2, 1]], [1])
    return p, layer


def run_case(p: ComParticle, layer: LayerStructure,
        pol: np.ndarray, dir_vec: np.ndarray,
        enei_arr: np.ndarray, tag: str) -> Tuple[np.ndarray, np.ndarray, float]:
    exc = PlaneWaveStatLayer(pol, layer)
    exc.dir = dir_vec.reshape(1, -1)
    bem = BEMStatLayer(p, layer)
    n = len(enei_arr)
    ext = np.zeros(n); sca = np.zeros(n)
    print('[info] Python {}: BEMStatLayer rod loop'.format(tag))
    t0 = time.time()
    for i, enei in enumerate(enei_arr):
        exc_pot = exc(p, enei)
        sig, _ = bem.solve(exc_pot)
        ext[i] = exc.extinction(sig)
        sca_val = exc.scattering(sig)
        sca[i] = float(np.real(sca_val[0] if isinstance(sca_val, tuple) else sca_val))
    elapsed = time.time() - t0
    print('[info]   {} time = {:.4f} s'.format(tag, elapsed))
    return ext, sca, elapsed


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok = True)
    os.makedirs(FIG_DIR, exist_ok = True)

    enei = np.linspace(400, 800, 41)
    timings = {}

    print('[info] === 04_bemstat_layer / rod (Python) ===')
    print('[info] trirod(10,40,[15,15,15]) Au, 1nm above glass, normal + oblique(45 TM)')

    p, layer = setup()

    pol_n = np.array([1.0, 0.0, 0.0])
    dir_n = np.array([0.0, 0.0, -1.0])
    ext_n, sca_n, t_n = run_case(p, layer, pol_n, dir_n, enei, 'normal')
    timings['normal'] = t_n
    save_csv(os.path.join(DATA_DIR, 'normal_python.csv'),
        enei, [ext_n, sca_n], ['extinction', 'scattering'])

    theta = np.pi / 4.0
    pol_o = np.array([np.cos(theta), 0.0, np.sin(theta)])
    dir_o = np.array([np.sin(theta), 0.0, -np.cos(theta)])
    ext_o, sca_o, t_o = run_case(p, layer, pol_o, dir_o, enei, 'oblique')
    timings['oblique'] = t_o
    save_csv(os.path.join(DATA_DIR, 'oblique_python.csv'),
        enei, [ext_o, sca_o], ['extinction', 'scattering'])

    plot_spectrum(enei, [ext_n, sca_n], ['ext', 'sca'],
        'Python BEMStatLayer rod normal (t={:.3f}s)'.format(t_n),
        os.path.join(FIG_DIR, 'normal_python.png'))
    plot_spectrum(enei, [ext_o, sca_o], ['ext', 'sca'],
        'Python BEMStatLayer rod oblique (t={:.3f}s)'.format(t_o),
        os.path.join(FIG_DIR, 'oblique_python.png'))

    save_timing(os.path.join(DATA_DIR, 'python_timing.csv'), timings)

    ml_timing_path = os.path.join(DATA_DIR, 'matlab_timing.csv')
    ml_timings = {}
    if os.path.exists(ml_timing_path):
        with open(ml_timing_path) as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    ml_timings[parts[0]] = float(parts[1])

    for case, py_ext, py_sca, t_py in [
            ('normal', ext_n, sca_n, t_n),
            ('oblique', ext_o, sca_o, t_o)]:
        ml = load_csv(os.path.join(DATA_DIR, '{}_matlab.csv'.format(case)))
        if ml is None:
            print('[info] {}: MATLAB data missing, skip'.format(case))
            continue
        t_ml = ml_timings.get(case, 0.0)
        max_rms, rms = plot_comparison(enei,
            [py_ext, py_sca], [ml[:, 1], ml[:, 2]],
            ['ext', 'sca'],
            '{} comparison'.format(case),
            os.path.join(FIG_DIR, '{}_comparison.png'.format(case)),
            t_py = t_py, t_ml = t_ml)
        print('[info] {}: max RMS = {:.2e} ({})'.format(case, max_rms,
            ', '.join(['{}={:.2e}'.format(k, v) for k, v in rms.items()])))


if __name__ == '__main__':
    main()
