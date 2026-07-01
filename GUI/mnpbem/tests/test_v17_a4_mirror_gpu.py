"""v1.7 A4 — Mirror BEM solver GPU audit (BEMRetMirror / BEMStatMirror /
BEMStatEigMirror / BEMLayerMirror).

The mirror code path is exercised under both MNPBEM_GPU=0 and MNPBEM_GPU=1
with a low threshold so a 144-face sphere routes through cupy.  Each test
asserts numerical equivalence between the two backends.  These tests
guard against three regressions:

  Bug 1 (v1.7 A4): CompGreenRetMirror/CompGreenStatMirror.eval silently
                   produced a zero list when the upstream base eval
                   returned cupy ndarrays under MNPBEM_GPU=1, breaking
                   every mirror solver.  Fix is host-promoting wrappers
                   in bem_ret_mirror / bem_stat_mirror.

  Bug 2 (v1.7 A4): BEMStatEigMirror used ``p.np`` / ``p.index_func`` from
                   the *full* particle which returned indices > nfaces of
                   the half mesh.  Fix builds half-particle index ranges
                   directly from ``self.p.p``.

  Bug 3 (v1.7 A4): BEMLayerMirror remains a dummy that must raise on
                   construction (not yet implemented).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest


os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')


try:
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    cp = None  # type: ignore
    _HAS_CUPY = False


cupy_required = pytest.mark.skipif(not _HAS_CUPY, reason = 'cupy not installed')


def _build_mirror_sphere(sym: str = 'x'):

    from GUI.mnpbem.geometry import trisphere, ComParticleMirror
    from GUI.mnpbem.materials import EpsDrude, EpsConst

    eps_au = EpsDrude(eps0 = 10.0, wp = 9.0, gammad = 0.07, name = 'Au')
    eps_vac = EpsConst(1.0)
    inout = np.array([[1, 2]])
    p = trisphere(144, 10.0)
    return ComParticleMirror([eps_vac, eps_au], [p], inout, sym = sym)


def _build_mirror_dimer(sym: str = 'x'):

    from GUI.mnpbem.geometry import trisphere, ComParticleMirror
    from GUI.mnpbem.materials import EpsDrude, EpsConst

    eps_au = EpsDrude(eps0 = 10.0, wp = 9.0, gammad = 0.07, name = 'Au')
    eps_vac = EpsConst(1.0)
    inout = np.array([[1, 2]])
    p_right = trisphere(144, 10.0).shift([15.0, 0, 0])
    return ComParticleMirror([eps_vac, eps_au], [p_right], inout, sym = sym)


def _force_gpu_mode(use_gpu: bool) -> None:
    os.environ['MNPBEM_GPU'] = '1' if use_gpu else '0'
    os.environ['MNPBEM_GPU_THRESHOLD'] = '100'


def _solve_ret_mirror(mp, enei: float = 600.0):

    from GUI.mnpbem.bem.bem_ret_mirror import BEMRetMirror
    from GUI.mnpbem.simulation.planewave_ret_mirror import PlaneWaveRetMirror

    bem = BEMRetMirror(mp)
    exc = PlaneWaveRetMirror([[1, 0, 0]], [[0, 0, 1]])
    sig, _ = bem.solve(exc(mp, enei))
    sca, _ = exc.scattering(sig)
    ext = exc.extinction(sig)
    return float(sca), float(ext)


def _solve_stat_mirror(mp, enei: float = 600.0):

    from GUI.mnpbem.bem.bem_stat_mirror import BEMStatMirror
    from GUI.mnpbem.simulation.planewave_stat_mirror import PlaneWaveStatMirror

    bem = BEMStatMirror(mp)
    exc = PlaneWaveStatMirror([[1, 0, 0]])
    sig, _ = bem.solve(exc(mp, enei))
    return float(exc.scattering(sig)), float(exc.extinction(sig))


def _solve_eig_mirror(mp, nev: int = 30, enei: float = 600.0):

    from GUI.mnpbem.bem.bem_stat_eig_mirror import BEMStatEigMirror
    from GUI.mnpbem.simulation.planewave_stat_mirror import PlaneWaveStatMirror

    bem = BEMStatEigMirror(mp, nev = nev)
    exc = PlaneWaveStatMirror([[1, 0, 0]])
    sig, _ = bem.solve(exc(mp, enei))
    return float(exc.scattering(sig)), float(exc.extinction(sig))


# ---------------------------------------------------------------------------
# Bug 3: BEMLayerMirror must raise NotImplementedError (parity with MATLAB).
# ---------------------------------------------------------------------------

def test_bem_layer_mirror_not_implemented():
    from GUI.mnpbem.bem.bem_layer_mirror import BEMLayerMirror
    with pytest.raises(NotImplementedError):
        BEMLayerMirror(None)


# ---------------------------------------------------------------------------
# BEMRetMirror — smoke + CPU vs GPU equivalence.
# ---------------------------------------------------------------------------

def test_bem_ret_mirror_cpu_smoke():
    _force_gpu_mode(False)
    mp = _build_mirror_sphere('x')
    sca, ext = _solve_ret_mirror(mp)
    assert np.isfinite(sca) and np.isfinite(ext)
    assert sca > 0.0


@cupy_required
def test_bem_ret_mirror_gpu_matches_cpu_sphere_x():
    _force_gpu_mode(False)
    sca_c, ext_c = _solve_ret_mirror(_build_mirror_sphere('x'))
    _force_gpu_mode(True)
    sca_g, ext_g = _solve_ret_mirror(_build_mirror_sphere('x'))
    assert np.isclose(sca_c, sca_g, rtol = 1e-10)
    assert np.isclose(ext_c, ext_g, rtol = 1e-10)


@cupy_required
@pytest.mark.parametrize('sym', ['x', 'y', 'xy'])
def test_bem_ret_mirror_gpu_all_planes(sym: str):
    _force_gpu_mode(False)
    sca_c, ext_c = _solve_ret_mirror(_build_mirror_sphere(sym))
    _force_gpu_mode(True)
    sca_g, ext_g = _solve_ret_mirror(_build_mirror_sphere(sym))
    assert np.isclose(sca_c, sca_g, rtol = 1e-10)
    assert np.isclose(ext_c, ext_g, rtol = 1e-10)


@cupy_required
def test_bem_ret_mirror_gpu_dimer_x():
    _force_gpu_mode(False)
    sca_c, ext_c = _solve_ret_mirror(_build_mirror_dimer('x'))
    _force_gpu_mode(True)
    sca_g, ext_g = _solve_ret_mirror(_build_mirror_dimer('x'))
    assert np.isclose(sca_c, sca_g, rtol = 1e-10)
    assert np.isclose(ext_c, ext_g, rtol = 1e-10)


# ---------------------------------------------------------------------------
# BEMStatMirror — smoke + CPU vs GPU equivalence.
# ---------------------------------------------------------------------------

def test_bem_stat_mirror_cpu_smoke():
    _force_gpu_mode(False)
    mp = _build_mirror_sphere('x')
    sca, ext = _solve_stat_mirror(mp)
    assert np.isfinite(sca) and np.isfinite(ext)


@cupy_required
def test_bem_stat_mirror_gpu_matches_cpu_sphere_x():
    _force_gpu_mode(False)
    sca_c, ext_c = _solve_stat_mirror(_build_mirror_sphere('x'))
    _force_gpu_mode(True)
    sca_g, ext_g = _solve_stat_mirror(_build_mirror_sphere('x'))
    assert np.isclose(sca_c, sca_g, rtol = 1e-10)
    assert np.isclose(ext_c, ext_g, rtol = 1e-10)


# ---------------------------------------------------------------------------
# BEMStatEigMirror — half-particle index fix + GPU equivalence.
# ---------------------------------------------------------------------------

def test_bem_stat_eig_mirror_cpu_smoke_half_index():
    # Regression for Bug 2 (v1.7 A4): ``p.index_func(ip + 1)`` previously
    # returned full-particle indices >= nfaces_half, which crashed the
    # eigenvector slicing.  The new path builds half-particle ranges from
    # ``self.p.p`` and must succeed.
    _force_gpu_mode(False)
    mp = _build_mirror_sphere('x')
    sca, ext = _solve_eig_mirror(mp, nev = 30)
    assert np.isfinite(sca) and np.isfinite(ext)


@cupy_required
def test_bem_stat_eig_mirror_gpu_matches_cpu():
    _force_gpu_mode(False)
    sca_c, ext_c = _solve_eig_mirror(_build_mirror_sphere('x'), nev = 30)
    _force_gpu_mode(True)
    sca_g, ext_g = _solve_eig_mirror(_build_mirror_sphere('x'), nev = 30)
    assert np.isclose(sca_c, sca_g, rtol = 1e-10)
    assert np.isclose(ext_c, ext_g, rtol = 1e-10)


# ---------------------------------------------------------------------------
# Mirror eval must produce numpy outputs (host promotion).
# ---------------------------------------------------------------------------

@cupy_required
def test_mirror_eval_host_promotes_cupy():
    # Bug 1 root cause: CompGreenRetMirror.eval returned [0, 0] when the
    # base eval was cupy.  The new host-promoting helper inside
    # ``bem_ret_mirror`` must always populate non-trivial blocks under
    # MNPBEM_GPU=1.
    _force_gpu_mode(True)
    mp = _build_mirror_sphere('x')
    from GUI.mnpbem.bem.bem_ret_mirror import _mirror_eval_host
    g = mp  # not used directly, helper takes the CompGreenRetMirror
    from GUI.mnpbem.greenfun.compgreen_ret_mirror import CompGreenRetMirror
    cgrm = CompGreenRetMirror(mp)
    blocks = _mirror_eval_host(cgrm, 0, 0, 'G', 600.0)
    assert all(isinstance(b, np.ndarray) for b in blocks)
    assert all(b.shape == (mp.nfaces, mp.nfaces) for b in blocks)


@cupy_required
def test_mirror_stat_eval_host_promotes_cupy():
    _force_gpu_mode(True)
    mp = _build_mirror_sphere('x')
    from GUI.mnpbem.bem.bem_stat_mirror import _mirror_stat_eval_host
    from GUI.mnpbem.greenfun.compgreen_stat_mirror import CompGreenStatMirror
    cgsm = CompGreenStatMirror(mp)
    blocks = _mirror_stat_eval_host(cgsm, 'F')
    assert all(isinstance(b, np.ndarray) for b in blocks)
    assert all(b.shape == (mp.nfaces, mp.nfaces) for b in blocks)


# ---------------------------------------------------------------------------
# GPU memory: no accumulation across repeated solves.
# ---------------------------------------------------------------------------

@cupy_required
def test_bem_ret_mirror_gpu_no_memory_growth():
    _force_gpu_mode(True)
    mp = _build_mirror_sphere('x')
    from GUI.mnpbem.bem.bem_ret_mirror import BEMRetMirror
    from GUI.mnpbem.simulation.planewave_ret_mirror import PlaneWaveRetMirror

    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()

    peaks = []
    for i in range(3):
        bem = BEMRetMirror(mp)
        exc = PlaneWaveRetMirror([[1, 0, 0]], [[0, 0, 1]])
        sig, _ = bem.solve(exc(mp, 550.0 + 25.0 * i))
        peaks.append(int(pool.used_bytes()))
        del bem, exc, sig
        pool.free_all_blocks()

    # Pool size after each iteration should be flat (within tolerance);
    # we tolerate the first warm-up but expect 2 .. n to be identical.
    assert peaks[1] == peaks[2], (
        '[error] GPU pool grew between iterations: {!r}'.format(peaks))
