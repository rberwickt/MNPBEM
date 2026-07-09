"""A5 — excitation runners cupy/numpy backend-mix audit.

These tests verify that excitation result-conversion paths (extinction /
scattering / absorption / decayrate / loss) work correctly when the BEM
solver returns surface-charge CompStruct members as cupy ndarrays.

Strategy: run the BEM solve on CPU to keep the harness independent of any
in-flight changes to bem/, then inject cupy copies of the surface charges
into a fresh CompStruct and confirm that:

  1. each conversion routine accepts the cupy-backed CompStruct without
     raising on numpy/cupy operand mixes;
  2. the returned cross sections / loss probabilities / decay rates are
     host (NumPy / float / scalar) — no cupy leakage to user code;
  3. the values match the CPU-only baseline within float precision.
"""

import os

import numpy as np
import pytest


try:
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    cp = None  # type: ignore
    _HAS_CUPY = False


cupy_required = pytest.mark.skipif(not _HAS_CUPY, reason = 'cupy not installed')


@pytest.fixture(autouse = True)
def _isolate_gpu_env(monkeypatch):
    """Other A1/A3/A4 GPU test suites flip MNPBEM_GPU=1; ensure this file
    always builds the BEM solver on CPU so the sig surface charges are
    NumPy arrays before injection."""
    monkeypatch.delenv('MNPBEM_GPU', raising = False)
    monkeypatch.delenv('MNPBEM_GPU_NATIVE', raising = False)
    yield


# -- common fixture: tiny sphere + BEM -------------------------------------

def _build_sphere_problem(sim):
    from GUI.mnpbem import EpsDrude, EpsConst, trisphere, ComParticle, BEMStat, BEMRet

    eps_in = EpsDrude(9.5, 138.0, 138.0)
    eps_out = EpsConst(1.0)
    p = ComParticle([eps_out, eps_in], [trisphere(144, 10.0)], [[2, 1]])

    if sim == 'stat':
        bem = BEMStat(p)
    else:
        bem = BEMRet(p)
    return p, bem


def _assert_close(cpu, gpu, tol = 1e-7, ctx = ''):
    cpu_a = np.squeeze(np.asarray(cpu))
    gpu_a = np.squeeze(np.asarray(gpu))
    diff = np.linalg.norm(cpu_a - gpu_a) / max(np.linalg.norm(cpu_a), 1e-30)
    assert diff < tol, '{}: GPU/CPU diverge rel={:.3e} cpu={} gpu={}'.format(
        ctx, diff, cpu_a, gpu_a)


def _cupy_clone(arr):
    if isinstance(arr, cp.ndarray):
        return arr.copy()
    return cp.asarray(np.asarray(arr))


def _to_cupy_compstruct_stat(sig_cpu):
    from GUI.mnpbem.greenfun import CompStruct
    return CompStruct(sig_cpu.p, sig_cpu.enei, sig = _cupy_clone(sig_cpu.sig))


def _to_cupy_compstruct_ret(sig_cpu):
    from GUI.mnpbem.greenfun import CompStruct
    return CompStruct(sig_cpu.p, sig_cpu.enei,
        sig1 = _cupy_clone(sig_cpu.sig1),
        sig2 = _cupy_clone(sig_cpu.sig2),
        h1 = _cupy_clone(sig_cpu.h1),
        h2 = _cupy_clone(sig_cpu.h2))


def _assert_host_finite(val, ctx = ''):
    arr = np.asarray(val)
    # cupy ndarrays survive np.asarray as object dtype with shape (), which
    # is what we want to forbid. Numpy scalars and arrays have numeric kind.
    assert arr.dtype.kind in 'fcui', '{}: not numeric/host (dtype={})'.format(
        ctx, arr.dtype)
    assert np.all(np.isfinite(arr)), '{}: contains non-finite'.format(ctx)


# -- PlaneWaveStat ---------------------------------------------------------

@cupy_required
@pytest.mark.parametrize('wl', [520.0, 600.0, 700.0])
def test_planewave_stat_cupy_sig_matches_cpu(wl):
    from GUI.mnpbem.simulation import PlaneWaveStat

    p, bem = _build_sphere_problem('stat')
    exc = PlaneWaveStat(np.array([1.0, 0.0, 0.0]))
    sig_cpu, _ = bem.solve(exc(p, wl))
    sig_cpu.enei = wl

    ext_cpu = exc.extinction(sig_cpu)
    sca_cpu = exc.scattering(sig_cpu)
    abs_cpu = exc.absorption(sig_cpu)

    sig_gpu = _to_cupy_compstruct_stat(sig_cpu)
    ext_gpu = exc.extinction(sig_gpu)
    sca_gpu = exc.scattering(sig_gpu)
    abs_gpu = exc.absorption(sig_gpu)

    _assert_host_finite(ext_gpu, 'planewavestat ext')
    _assert_host_finite(sca_gpu, 'planewavestat sca')
    _assert_host_finite(abs_gpu, 'planewavestat abs')
    _assert_close(ext_cpu, ext_gpu, ctx = 'planewavestat ext')
    _assert_close(sca_cpu, sca_gpu, ctx = 'planewavestat sca')
    _assert_close(abs_cpu, abs_gpu, ctx = 'planewavestat abs')


# -- PlaneWaveRet ----------------------------------------------------------

@cupy_required
@pytest.mark.parametrize('wl', [520.0, 600.0, 700.0])
def test_planewave_ret_cupy_sig_matches_cpu(wl):
    from GUI.mnpbem.simulation import PlaneWaveRet

    p, bem = _build_sphere_problem('ret')
    exc = PlaneWaveRet(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    sig_cpu, _ = bem.solve(exc(p, wl))
    sig_cpu.enei = wl

    ext_cpu = exc.extinction(sig_cpu)
    sca_cpu, _ = exc.scattering(sig_cpu)
    abs_cpu = exc.absorption(sig_cpu)

    sig_gpu = _to_cupy_compstruct_ret(sig_cpu)
    ext_gpu = exc.extinction(sig_gpu)
    sca_gpu, _ = exc.scattering(sig_gpu)
    abs_gpu = exc.absorption(sig_gpu)

    _assert_host_finite(ext_gpu, 'planewaveret ext')
    _assert_host_finite(sca_gpu, 'planewaveret sca')
    _assert_host_finite(abs_gpu, 'planewaveret abs')
    _assert_close(ext_cpu, ext_gpu, tol = 1e-9, ctx = 'planewaveret ext')
    _assert_close(sca_cpu, sca_gpu, tol = 1e-9, ctx = 'planewaveret sca')
    _assert_close(abs_cpu, abs_gpu, tol = 1e-9, ctx = 'planewaveret abs')


# -- DipoleStat ------------------------------------------------------------

@cupy_required
def test_dipole_stat_cupy_sig_matches_cpu():
    from GUI.mnpbem.simulation import DipoleStat
    from GUI.mnpbem import ComPoint

    p, bem = _build_sphere_problem('stat')
    pt = ComPoint(p, np.array([[0.0, 0.0, 15.0]]))
    exc = DipoleStat(pt, np.array([0.0, 0.0, 1.0]))
    sig_cpu, _ = bem.solve(exc(p, 600.0))
    sig_cpu.enei = 600.0

    tot_cpu, _, _ = exc.decayrate(sig_cpu)

    sig_gpu = _to_cupy_compstruct_stat(sig_cpu)
    tot_gpu, _, _ = exc.decayrate(sig_gpu)

    _assert_host_finite(tot_gpu, 'dipolestat decayrate')
    _assert_close(tot_cpu, tot_gpu, ctx = 'dipolestat decayrate')


# -- DipoleRet -------------------------------------------------------------

@cupy_required
def test_dipole_ret_cupy_sig_matches_cpu():
    from GUI.mnpbem.simulation import DipoleRet
    from GUI.mnpbem import ComPoint

    p, bem = _build_sphere_problem('ret')
    pt = ComPoint(p, np.array([[0.0, 0.0, 15.0]]))
    exc = DipoleRet(pt, np.array([0.0, 0.0, 1.0]))
    sig_cpu, _ = bem.solve(exc(p, 600.0))
    sig_cpu.enei = 600.0

    tot_cpu, _, _ = exc.decayrate(sig_cpu)

    # decayrate mutates sig.sig1/h1 to host copies; clone before second call.
    sig_cpu2, _ = bem.solve(exc(p, 600.0))
    sig_cpu2.enei = 600.0
    sig_gpu = _to_cupy_compstruct_ret(sig_cpu2)
    tot_gpu, _, _ = exc.decayrate(sig_gpu)

    _assert_host_finite(tot_gpu, 'dipoleret decayrate')
    _assert_close(tot_cpu, tot_gpu, ctx = 'dipoleret decayrate')


# -- EELSStat --------------------------------------------------------------

@cupy_required
def test_eels_stat_cupy_sig_matches_cpu():
    from GUI.mnpbem.simulation import EELSStat

    p, bem = _build_sphere_problem('stat')
    impact = np.array([[12.0, 0.0]])
    exc = EELSStat(p, impact, 0.5, EELSStat.ene2vel(200e3))
    sig_cpu, _ = bem.solve(exc(p, 600.0))
    sig_cpu.enei = 600.0

    psurf_cpu, _ = exc.loss(sig_cpu)

    sig_gpu = _to_cupy_compstruct_stat(sig_cpu)
    psurf_gpu, _ = exc.loss(sig_gpu)

    _assert_host_finite(psurf_gpu, 'eelsstat psurf')
    _assert_close(psurf_cpu, psurf_gpu, ctx = 'eelsstat psurf')


# -- EELSRet ---------------------------------------------------------------

@cupy_required
def test_eels_ret_cupy_sig_matches_cpu():
    from GUI.mnpbem.simulation import EELSRet

    p, bem = _build_sphere_problem('ret')
    impact = np.array([[12.0, 0.0]])
    exc = EELSRet(p, impact, 0.5, EELSRet.ene2vel(200e3))
    sig_cpu, _ = bem.solve(exc(p, 600.0))
    sig_cpu.enei = 600.0

    psurf_cpu, _ = exc.loss(sig_cpu)

    sig_gpu = _to_cupy_compstruct_ret(sig_cpu)
    psurf_gpu, _ = exc.loss(sig_gpu)

    _assert_host_finite(psurf_gpu, 'eelsret psurf')
    _assert_close(psurf_cpu, psurf_gpu, ctx = 'eelsret psurf')
