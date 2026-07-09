import os
import sys
import time

from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MNPBEM_ROOT = '/home/yoojk20/workspace/MNPBEM'
sys.path.insert(0, MNPBEM_ROOT)
sys.path.insert(0, os.path.join(MNPBEM_ROOT, 'validation'))

from mnpbem.materials.eps_table import EpsTable
from mnpbem.materials.eps_const import EpsConst
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.simulation import PlaneWaveStat, PlaneWaveRet, MeshField
from mnpbem.bem import BEMStat, BEMRet

from _common import load_csv, save_timing, compute_rms


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')

WL = 520.0
GRID_N = 31
GRID_RANGE = 30.0


def setup_grid() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_1d = np.linspace(-GRID_RANGE, GRID_RANGE, GRID_N)
    z_1d = np.linspace(-GRID_RANGE, GRID_RANGE, GRID_N)
    x, z = np.meshgrid(x_1d, z_1d)
    y = np.zeros_like(x)
    return x, y, z


def build_particle() -> ComParticle:
    epstab = [EpsConst(1.0), EpsTable('gold.dat')]
    return ComParticle(epstab, [trisphere(144, 20)], [[2, 1]])


def run_stat(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, float]:
    p = build_particle()
    bem = BEMStat(p)
    exc = PlaneWaveStat([1, 0, 0])
    t0 = time.time()
    pot = exc.potential(p, WL)
    sig, _ = bem.solve(pot)
    mf = MeshField(p, x, y, z)
    e, _ = mf.field(sig)
    e2 = np.sum(np.abs(e) ** 2, axis = -1)
    return e2, time.time() - t0


def run_ret(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, float]:
    p = build_particle()
    bem = BEMRet(p)
    exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])
    t0 = time.time()
    pot = exc.potential(p, WL)
    sig, _ = bem.solve(pot)
    mf = MeshField(p, x, y, z, sim = 'ret')
    e, _ = mf.field(sig)
    e2 = np.sum(np.abs(e) ** 2, axis = -1)
    return e2, time.time() - t0


def save_map(filepath: str, x: np.ndarray, z: np.ndarray, e2: np.ndarray) -> None:
    xr = x.ravel(order = 'F'); zr = z.ravel(order = 'F'); er = e2.ravel(order = 'F')
    n = xr.shape[0]
    data = np.empty((n, 3), dtype = float)
    data[:, 0] = xr
    data[:, 1] = zr
    data[:, 2] = er
    np.savetxt(filepath, data, delimiter = ',', header = 'x_nm,z_nm,e2',
        comments = '', fmt = '%.10e')


def plot_map(e2: np.ndarray, title: str, savepath: str) -> None:
    fig, ax = plt.subplots(figsize = (6, 5))
    im = ax.imshow(np.log10(e2 + 1e-30),
        extent = [-GRID_RANGE, GRID_RANGE, -GRID_RANGE, GRID_RANGE],
        origin = 'lower', cmap = 'inferno')
    plt.colorbar(im, ax = ax, label = 'log10 |E|^2')
    ax.set_xlabel('x (nm)'); ax.set_ylabel('z (nm)')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(savepath, dpi = 150)
    plt.close(fig)


def plot_compare(e2_py: np.ndarray, e2_ml: np.ndarray,
        title: str, savepath: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize = (18, 5))
    for ax, data, t in zip(axes[:2], [e2_py, e2_ml], ['Python', 'MATLAB']):
        im = ax.imshow(np.log10(data + 1e-30),
            extent = [-GRID_RANGE, GRID_RANGE, -GRID_RANGE, GRID_RANGE],
            origin = 'lower', cmap = 'inferno')
        plt.colorbar(im, ax = ax, label = 'log10 |E|^2')
        ax.set_title(t)
        ax.set_xlabel('x'); ax.set_ylabel('z')
    diff = np.abs(e2_py - e2_ml) / (np.abs(e2_ml) + 1e-30)
    ax = axes[2]
    im = ax.imshow(np.log10(diff + 1e-30),
        extent = [-GRID_RANGE, GRID_RANGE, -GRID_RANGE, GRID_RANGE],
        origin = 'lower', cmap = 'viridis')
    plt.colorbar(im, ax = ax, label = 'log10 rel err')
    ax.set_title('rel err')
    ax.set_xlabel('x'); ax.set_ylabel('z')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(savepath, dpi = 150)
    plt.close(fig)


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok = True)
    os.makedirs(FIG_DIR, exist_ok = True)

    x, y, z = setup_grid()
    timings = {}

    print('[info] === 12_nearfield / sphere (Python) ===')
    print('[info] trisphere(144,20) Au, 31x31 grid at y=0, WL={}nm'.format(WL))

    e2_s, t_s = run_stat(x, y, z)
    timings['stat'] = t_s
    save_map(os.path.join(DATA_DIR, 'stat_python.csv'), x, z, e2_s)
    plot_map(e2_s, 'Python BEMStat |E|^2 sphere',
        os.path.join(FIG_DIR, 'stat_python.png'))
    print('[info] stat {:.3f}s'.format(t_s))

    e2_r, t_r = run_ret(x, y, z)
    timings['ret'] = t_r
    save_map(os.path.join(DATA_DIR, 'ret_python.csv'), x, z, e2_r)
    plot_map(e2_r, 'Python BEMRet |E|^2 sphere',
        os.path.join(FIG_DIR, 'ret_python.png'))
    print('[info] ret  {:.3f}s'.format(t_r))

    save_timing(os.path.join(DATA_DIR, 'python_timing.csv'), timings)

    for case, e2_py in [('stat', e2_s), ('ret', e2_r)]:
        ml = load_csv(os.path.join(DATA_DIR, '{}_matlab.csv'.format(case)))
        if ml is None:
            print('[info] {}: MATLAB data missing'.format(case))
            continue
        e2_ml = ml[:, 2].reshape(GRID_N, GRID_N, order = 'F')
        rms = compute_rms(e2_py, e2_ml)
        print('[info] {}: RMS = {:.2e}'.format(case, rms))
        plot_compare(e2_py, e2_ml, '{} sphere |E|^2 compare'.format(case),
            os.path.join(FIG_DIR, '{}_comparison.png'.format(case)))


if __name__ == '__main__':
    main()
