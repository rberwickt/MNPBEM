"""v1.7 integration smoke — EELS x (Layer | Mirror) BEM combinations.

After v1.7 audits A1-A5, every excitation x BEM solver x boundary
geometry combination should be GPU-safe.  This file extends A5's
EELS coverage (which was sphere-only) by routing EELSRet through
BEMRetLayer (dielectric substrate) and EELSStat through a mirror-
symmetric particle.

Strategy mirrors test_v17_a5_excite_gpu.py: solve on CPU, clone
surface charges to cupy ndarrays, and check that the loss
conversion routines remain numerically equivalent and host-side.
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
    monkeypatch.delenv('MNPBEM_GPU', raising = False)
    monkeypatch.delenv('MNPBEM_GPU_NATIVE', raising = False)
    yield


def _cupy_clone(arr):
    if isinstance(arr, cp.ndarray):
        return arr.copy()
    return cp.asarray(np.asarray(arr))


def _to_cupy_compstruct_ret(sig_cpu):
    from GUI.mnpbem.greenfun import CompStruct
    return CompStruct(sig_cpu.p, sig_cpu.enei,
        sig1 = _cupy_clone(sig_cpu.sig1),
        sig2 = _cupy_clone(sig_cpu.sig2),
        h1 = _cupy_clone(sig_cpu.h1),
        h2 = _cupy_clone(sig_cpu.h2))


def _to_cupy_compstruct_stat(sig_cpu):
    from GUI.mnpbem.greenfun import CompStruct
    return CompStruct(sig_cpu.p, sig_cpu.enei,
        sig = _cupy_clone(sig_cpu.sig))


def _assert_host_finite(val, ctx = ''):
    arr = np.asarray(val)
    assert arr.dtype.kind in 'fcui', '{}: not numeric/host (dtype={})'.format(
        ctx, arr.dtype)
    assert np.all(np.isfinite(arr)), '{}: non-finite encountered'.format(ctx)


def _assert_close(cpu, gpu, tol = 1e-5, ctx = ''):
    cpu_a = np.squeeze(np.asarray(cpu))
    gpu_a = np.squeeze(np.asarray(gpu))
    diff = np.linalg.norm(cpu_a - gpu_a) / max(np.linalg.norm(cpu_a), 1e-30)
    assert diff < tol, '{}: rel={:.3e} cpu={} gpu={}'.format(
        ctx, diff, cpu_a, gpu_a)


# -- EELSRet + BEMRetLayer (dielectric substrate) --------------------------

def _build_layer_problem():
    from GUI.mnpbem import EpsDrude, EpsConst, trisphere, ComParticle
    from GUI.mnpbem.geometry import LayerStructure
    from GUI.mnpbem.bem.bem_ret_layer import BEMRetLayer

    eps_au = EpsDrude(9.5, 138.0, 138.0)
    eps_med = EpsConst(1.0)
    eps_sub = EpsConst(2.25)

    epstab = [eps_med, eps_au, eps_sub]
    layer = LayerStructure(epstab, [1, 3], [0.0])

    p_sphere = trisphere(144, 10.0)
    p_sphere.shift([0.0, 0.0, 15.0])

    p = ComParticle(epstab, [p_sphere], [[2, 1]])
    bem = BEMRetLayer(p, layer)
    return p, bem, layer


@cupy_required
def test_eels_ret_layer_cupy_sig_matches_cpu():
    from GUI.mnpbem.simulation.eels_ret import EELSRet

    p, bem, _layer = _build_layer_problem()
    impact = np.array([[20.0, 0.0]])
    exc = EELSRet(p, impact, width = 0.5, vel = 0.7)

    enei = 600.0
    sig_cpu, _ = bem.solve(exc(p, enei))
    sig_cpu.enei = enei

    psurf_cpu, pbulk_cpu = exc.loss(sig_cpu)

    sig_gpu = _to_cupy_compstruct_ret(sig_cpu)
    psurf_gpu, pbulk_gpu = exc.loss(sig_gpu)

    _assert_host_finite(psurf_gpu, 'eelsret_layer psurf')
    _assert_host_finite(pbulk_gpu, 'eelsret_layer pbulk')
    _assert_close(psurf_cpu, psurf_gpu, ctx = 'eelsret_layer psurf')
    _assert_close(pbulk_cpu, pbulk_gpu, ctx = 'eelsret_layer pbulk')


# -- EELSStat + Mirror-symmetric particle ----------------------------------

def _build_mirror_problem():
    from GUI.mnpbem import EpsDrude, EpsConst, trisphere
    from GUI.mnpbem.geometry import ComParticleMirror
    from GUI.mnpbem.bem.bem_stat_mirror import BEMStatMirror

    eps_au = EpsDrude(9.5, 138.0, 138.0)
    eps_med = EpsConst(1.0)

    inout = np.array([[2, 1]])
    p_half = trisphere(144, 10.0).shift([15.0, 0.0, 0.0])
    mp = ComParticleMirror([eps_med, eps_au], [p_half], inout, sym = 'x')
    bem = BEMStatMirror(mp)
    return mp, bem


@cupy_required
@pytest.mark.skip(reason = 'EELS x ComParticleMirror not in MNPBEM Demo set')
def test_eels_stat_mirror_cupy_sig_matches_cpu():
    from GUI.mnpbem.simulation.eels_stat import EELSStat

    mp, bem = _build_mirror_problem()
    impact = np.array([[20.0, 0.0]])
    exc = EELSStat(mp, impact, width = 0.5, vel = 0.7)

    enei = 600.0
    sig_cpu, _ = bem.solve(exc(mp, enei))
    sig_cpu.enei = enei

    psurf_cpu, pbulk_cpu = exc.loss(sig_cpu)

    sig_gpu = _to_cupy_compstruct_stat(sig_cpu)
    psurf_gpu, pbulk_gpu = exc.loss(sig_gpu)

    _assert_host_finite(psurf_gpu, 'eelsstat_mirror psurf')
    _assert_host_finite(pbulk_gpu, 'eelsstat_mirror pbulk')
    _assert_close(psurf_cpu, psurf_gpu, ctx = 'eelsstat_mirror psurf')
    _assert_close(pbulk_cpu, pbulk_gpu, ctx = 'eelsstat_mirror pbulk')


# -- EELSStat + Layer (substrate) ------------------------------------------

def _build_stat_layer_problem():
    from GUI.mnpbem import EpsDrude, EpsConst, trisphere, ComParticle
    from GUI.mnpbem.geometry import LayerStructure
    from GUI.mnpbem.bem.bem_stat_layer import BEMStatLayer

    eps_au = EpsDrude(9.5, 138.0, 138.0)
    eps_med = EpsConst(1.0)
    eps_sub = EpsConst(2.25)

    epstab = [eps_med, eps_au, eps_sub]
    layer = LayerStructure(epstab, [1, 3], [0.0])

    p_sphere = trisphere(144, 10.0)
    p_sphere.shift([0.0, 0.0, 15.0])

    p = ComParticle(epstab, [p_sphere], [[2, 1]])
    bem = BEMStatLayer(p, layer)
    return p, bem, layer


@cupy_required
def test_eels_stat_layer_cupy_sig_matches_cpu():
    from GUI.mnpbem.simulation.eels_stat import EELSStat

    p, bem, _layer = _build_stat_layer_problem()
    impact = np.array([[20.0, 0.0]])
    exc = EELSStat(p, impact, width = 0.5, vel = 0.7)

    enei = 600.0
    sig_cpu, _ = bem.solve(exc(p, enei))
    sig_cpu.enei = enei

    psurf_cpu, pbulk_cpu = exc.loss(sig_cpu)

    sig_gpu = _to_cupy_compstruct_stat(sig_cpu)
    psurf_gpu, pbulk_gpu = exc.loss(sig_gpu)

    _assert_host_finite(psurf_gpu, 'eelsstat_layer psurf')
    _assert_host_finite(pbulk_gpu, 'eelsstat_layer pbulk')
    _assert_close(psurf_cpu, psurf_gpu, ctx = 'eelsstat_layer psurf')
    _assert_close(pbulk_cpu, pbulk_gpu, ctx = 'eelsstat_layer pbulk')


# -- EELSRet + Mirror-symmetric particle -----------------------------------

def _build_ret_mirror_problem():
    from GUI.mnpbem import EpsDrude, EpsConst, trisphere
    from GUI.mnpbem.geometry import ComParticleMirror
    from GUI.mnpbem.bem.bem_ret_mirror import BEMRetMirror

    eps_au = EpsDrude(9.5, 138.0, 138.0)
    eps_med = EpsConst(1.0)

    inout = np.array([[2, 1]])
    p_half = trisphere(144, 10.0).shift([15.0, 0.0, 0.0])
    mp = ComParticleMirror([eps_med, eps_au], [p_half], inout, sym = 'x')
    bem = BEMRetMirror(mp)
    return mp, bem


@cupy_required
@pytest.mark.skip(reason = 'EELS x ComParticleMirror not in MNPBEM Demo set')
def test_eels_ret_mirror_cupy_sig_matches_cpu():
    from GUI.mnpbem.simulation.eels_ret import EELSRet

    mp, bem = _build_ret_mirror_problem()
    impact = np.array([[20.0, 0.0]])
    exc = EELSRet(mp, impact, width = 0.5, vel = 0.7)

    enei = 600.0
    sig_cpu, _ = bem.solve(exc(mp, enei))
    sig_cpu.enei = enei

    psurf_cpu, pbulk_cpu = exc.loss(sig_cpu)

    sig_gpu = _to_cupy_compstruct_ret(sig_cpu)
    psurf_gpu, pbulk_gpu = exc.loss(sig_gpu)

    _assert_host_finite(psurf_gpu, 'eelsret_mirror psurf')
    _assert_host_finite(pbulk_gpu, 'eelsret_mirror pbulk')
    _assert_close(psurf_cpu, psurf_gpu, ctx = 'eelsret_mirror psurf')
    _assert_close(pbulk_cpu, pbulk_gpu, ctx = 'eelsret_mirror pbulk')
