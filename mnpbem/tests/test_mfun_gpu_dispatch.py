import os
import sys

import numpy as np
import pytest

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.bem import BEMRetIter
from mnpbem.simulation import PlaneWaveRet


try:
    import cupy as _cp
    _CUPY_OK = True
except ImportError:
    _CUPY_OK = False


_ENEI_TEST = 540.0
_POL = np.array([1.0, 0.0, 0.0])
_DIR = np.array([0.0, 0.0, 1.0])


def _auag_dimer_small():
    epstab = [EpsConst(1.77), EpsTable('gold.dat'), EpsTable('silver.dat')]
    core_d, shell_t = 5.0, 1.5
    outer_d = core_d + 2.0 * shell_t
    gap = 0.6
    half = (outer_d + gap) / 2.0
    p1_shell = trisphere(144, outer_d); p1_core = trisphere(144, core_d)
    p1_shell.shift([-half, 0.0, 0.0]); p1_core.shift([-half, 0.0, 0.0])
    p2_shell = trisphere(144, outer_d); p2_core = trisphere(144, core_d)
    p2_shell.shift([+half, 0.0, 0.0]); p2_core.shift([+half, 0.0, 0.0])
    inds = [[3, 1], [2, 3], [3, 1], [2, 3]]
    p = ComParticle(epstab, [p1_shell, p1_core, p2_shell, p2_core],
            inds, 1, 2, interp = 'curv')
    return p


def _ext_value(p, exc, enei):
    bem = BEMRetIter(p, hmatrix = True, htol = 1e-6, tol = 1e-6,
            maxit = 300, preconditioner = 'auto')
    sig, _ = bem.solve(exc.potential(p, enei))
    return float(np.real(np.ravel(exc.extinction(sig))[0]))


@pytest.mark.skipif(not _CUPY_OK,
        reason = 'cupy unavailable; v1.6.4 GPU mfun dispatch needs cuda runtime')
def test_mfun_gpu_dispatch_flag_off_on_match():

    p = _auag_dimer_small()
    exc = PlaneWaveRet(_POL, _DIR)

    saved_gpu = os.environ.get('MNPBEM_GPU')
    saved_flag = os.environ.get('MNPBEM_AGGRESSIVE_GPU_MFUN')
    try:
        os.environ['MNPBEM_GPU'] = '1'
        os.environ.pop('MNPBEM_AGGRESSIVE_GPU_MFUN', None)
        ext_off = _ext_value(p, exc, _ENEI_TEST)

        os.environ['MNPBEM_AGGRESSIVE_GPU_MFUN'] = '1'
        ext_on = _ext_value(p, exc, _ENEI_TEST)
    finally:
        if saved_gpu is None:
            os.environ.pop('MNPBEM_GPU', None)
        else:
            os.environ['MNPBEM_GPU'] = saved_gpu
        if saved_flag is None:
            os.environ.pop('MNPBEM_AGGRESSIVE_GPU_MFUN', None)
        else:
            os.environ['MNPBEM_AGGRESSIVE_GPU_MFUN'] = saved_flag

    rel_diff = abs(ext_off - ext_on) / abs(ext_off)
    assert rel_diff < 1e-10, \
        '[error] flag on/off mfun GPU dispatch differs: rel = {:.3e} ' \
        '(off = {:.6e}, on = {:.6e})'.format(rel_diff, ext_off, ext_on)
