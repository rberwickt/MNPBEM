"""Unit tests for Lane B GPU dispatch:
- PlaneWaveRet.potential / field
- SpectrumRet.farfield / scattering
- EpsTable cupy 입력 호환

CPU vs GPU 결과가 (가능한 한) bit-identical인지 확인.
"""

import importlib
import os

import numpy as np
import pytest


def _setup_dimer():
    from mnpbem.materials import EpsConst, EpsTable
    from mnpbem.geometry import tricube, ComParticle
    cube1 = tricube(24, 47, e = 0.2, refine = 2); cube1.shift([-24, 0, 0])
    cube2 = tricube(24, 47, e = 0.2, refine = 2); cube2.shift([+24, 0, 0])
    epstab = [EpsConst(1.33 ** 2), EpsTable('gold.dat')]
    p = ComParticle(epstab, [cube1, cube2], [[2, 1], [2, 1]],
                    interp = 'curv', refine = 2)
    return p


def _toggle_gpu(use_gpu):
    import mnpbem.utils.gpu as gmod
    gmod.USE_GPU = bool(use_gpu)


def test_planewave_potential_cpu_vs_gpu():
    try:
        import cupy  # noqa: F401
    except ImportError:
        pytest.skip("cupy not installed")

    os.environ['MNPBEM_NUMBA'] = '1'
    p = _setup_dimer()
    from mnpbem.simulation import PlaneWaveRet

    _toggle_gpu(False)
    exc = PlaneWaveRet([[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [0, 0, 1]])
    cs_cpu = exc(p, 600.0)

    _toggle_gpu(True)
    exc2 = PlaneWaveRet([[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [0, 0, 1]])
    cs_gpu = exc2(p, 600.0)

    for k in ['a1', 'a1p', 'a2', 'a2p']:
        a = getattr(cs_cpu, k)
        b = getattr(cs_gpu, k)
        rel = np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-30)
        assert rel < 1e-12, '[error] PlaneWaveRet {} CPU/GPU rel={:.3e}'.format(k, rel)


def test_spectrum_farfield_scattering_cpu_vs_gpu():
    try:
        import cupy  # noqa: F401
    except ImportError:
        pytest.skip("cupy not installed")

    os.environ['MNPBEM_NUMBA'] = '1'
    p = _setup_dimer()
    from mnpbem.bem import BEMRet
    from mnpbem.simulation import PlaneWaveRet

    bem = BEMRet(p)

    _toggle_gpu(False)
    exc = PlaneWaveRet([[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [0, 0, 1]])
    sig_cpu, _ = bem.solve(exc(p, 600.0))
    ext_cpu = np.asarray(exc.extinction(sig_cpu)).real.flatten()
    sca_cpu, _ = exc.scattering(sig_cpu)
    sca_cpu = np.asarray(sca_cpu).real.flatten()

    _toggle_gpu(True)
    exc2 = PlaneWaveRet([[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [0, 0, 1]])
    sig_gpu, _ = bem.solve(exc2(p, 600.0))
    ext_gpu = np.asarray(exc2.extinction(sig_gpu)).real.flatten()
    sca_gpu, _ = exc2.scattering(sig_gpu)
    sca_gpu = np.asarray(sca_gpu).real.flatten()

    rel_ext = np.max(np.abs(ext_cpu - ext_gpu)) / max(np.max(np.abs(ext_cpu)), 1e-30)
    rel_sca = np.max(np.abs(sca_cpu - sca_gpu)) / max(np.max(np.abs(sca_cpu)), 1e-30)
    assert rel_ext < 1e-12, '[error] extinction CPU/GPU rel={:.3e}'.format(rel_ext)
    assert rel_sca < 1e-12, '[error] scattering CPU/GPU rel={:.3e}'.format(rel_sca)


def test_epstable_cupy_input_matches_numpy():
    try:
        import cupy as cp
    except ImportError:
        pytest.skip("cupy not installed")
    from mnpbem.materials import EpsTable
    e = EpsTable('gold.dat')

    wls = np.linspace(500, 1000, 10)
    eps_n, k_n = e(wls)
    eps_g, k_g = e(cp.asarray(wls))

    assert np.allclose(eps_n, eps_g, atol = 0, rtol = 0)
    assert np.allclose(k_n, k_g, atol = 0, rtol = 0)
