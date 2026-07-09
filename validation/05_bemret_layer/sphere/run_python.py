import os
import sys
import time

from typing import Tuple

import numpy as np

MNPBEM_ROOT = '/home/yoojk20/workspace/MNPBEM'
sys.path.insert(0, MNPBEM_ROOT)
sys.path.insert(0, os.path.join(MNPBEM_ROOT, 'validation'))

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle, LayerStructure
from mnpbem.bem import BEMRetLayer
from mnpbem.simulation import PlaneWaveRetLayer
from mnpbem.greenfun import GreenTabLayer

from _common import save_csv, load_csv, save_timing, plot_spectrum, plot_comparison


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok = True)
    os.makedirs(FIG_DIR, exist_ok = True)

    enei = np.linspace(450, 750, 16)
    timings = {}

    print('[info] === 05_bemret_layer / sphere (Python) ===')
    print('[info] 20nm Au sphere, 1nm above glass, normal incidence (planewave)')

    epstab = [EpsConst(1.0), EpsTable('gold.dat'), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 3], [0.0])
    sphere = trisphere(144, 20.0)
    sphere.shift([0, 0, -sphere.pos[:, 2].min() + 1.0])
    p = ComParticle(epstab, [sphere], [[2, 1]], [1])

    tab = layer.tabspace(p)
    gt = GreenTabLayer(layer, tab = tab)
    gt.set(np.linspace(350, 800, 5))
    bem = BEMRetLayer(p, layer, greentab = gt)

    pol = np.array([[1.0, 0.0, 0.0]])
    dir_vec = np.array([[0.0, 0.0, -1.0]])
    exc = PlaneWaveRetLayer(pol, dir_vec, layer)

    n = len(enei)
    ext = np.zeros(n); sca = np.zeros(n)

    print('[info] BEMRetLayer loop')
    t0 = time.time()
    for i, en in enumerate(enei):
        exc_pot = exc(p, en)
        sig, _ = bem.solve(exc_pot)
        sca_val, _ = exc.scattering(sig)
        sca[i] = float(sca_val)
        ext[i] = float(exc.extinction(sig))
    t_bem = time.time() - t0
    timings['bem'] = t_bem
    absc = ext - sca
    print('[info]   BEM time = {:.4f} s'.format(t_bem))

    save_csv(os.path.join(DATA_DIR, 'bemretlayer_python.csv'),
        enei, [ext, sca, absc], ['extinction', 'scattering', 'absorption'])
    plot_spectrum(enei, [ext, sca, absc], ['ext', 'sca', 'abs'],
        'Python BEMRetLayer sphere (t={:.3f}s)'.format(t_bem),
        os.path.join(FIG_DIR, 'bemretlayer_python.png'))

    save_timing(os.path.join(DATA_DIR, 'python_timing.csv'), timings)

    ml_timing_path = os.path.join(DATA_DIR, 'matlab_timing.csv')
    ml_timings = {}
    if os.path.exists(ml_timing_path):
        with open(ml_timing_path) as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    ml_timings[parts[0]] = float(parts[1])

    ml = load_csv(os.path.join(DATA_DIR, 'bemretlayer_matlab.csv'))
    if ml is not None:
        t_ml = ml_timings.get('bem', 0.0)
        max_rms, rms = plot_comparison(enei,
            [ext, sca, absc], [ml[:, 1], ml[:, 2], ml[:, 3]],
            ['ext', 'sca', 'abs'],
            'BEMRetLayer sphere comparison',
            os.path.join(FIG_DIR, 'bemretlayer_comparison.png'),
            t_py = t_bem, t_ml = t_ml)
        print('[info] comparison: max RMS = {:.2e} ({})'.format(max_rms,
            ', '.join(['{}={:.2e}'.format(k, v) for k, v in rms.items()])))
    else:
        print('[info] MATLAB data not found, skip compare')


if __name__ == '__main__':
    main()
