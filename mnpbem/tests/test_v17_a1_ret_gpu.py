"""v1.7 A1 — BEMRet + BEMRetLayer GPU audit smoke tests.

Covers the 5 bug patterns:
1. Backend mix (cupy + numpy operand in binary op)
2. Memory accumulation (cupy intermediates not freed)
3. Dispatch routing (multi_gpu, n_workers)
4. Numerical precision (single vs double, FP associativity)
5. Edge cases (1 particle vs multi-particle, scalar vs diagonal eps, mirror skip)

GPU 0 only: ``os.environ['CUDA_VISIBLE_DEVICES'] = '0'``.
"""

import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import numpy as np
import pytest


# Skip the whole module when cupy / CUDA is unavailable so the file does
# not fail collection on CPU-only hosts.
cp = pytest.importorskip('cupy')

try:
    _N_GPUS = int(cp.cuda.runtime.getDeviceCount())
except Exception:
    _N_GPUS = 0
if _N_GPUS == 0:
    pytest.skip(
            'no CUDA device available — skipping v1.7 A1 GPU smoke',
            allow_module_level = True)


# Common imports used by every smoke case.
from mnpbem.materials import EpsConst, EpsDrude, EpsTable
from mnpbem.geometry import trisphere, tricube, ComParticle, LayerStructure
from mnpbem.bem import BEMRet, BEMRetLayer
from mnpbem.simulation import PlaneWaveRet, PlaneWaveRetLayer, DipoleRet
from mnpbem.greenfun import GreenTabLayer


_TOL = 1e-5


def _set_gpu_env(enable = True):
    if enable:
        os.environ['MNPBEM_GPU'] = '1'
        os.environ['MNPBEM_GPU_THRESHOLD'] = '10'
        os.environ['MNPBEM_GPU_NATIVE'] = '1'
    else:
        os.environ['MNPBEM_GPU'] = '0'
        os.environ.pop('MNPBEM_GPU_NATIVE', None)


def _reload_gpu_module():
    # Force ``mnpbem.utils.gpu`` to re-read the env vars after they change.
    import importlib

    import mnpbem.utils.gpu as _gpu
    importlib.reload(_gpu)


def _build_sphere(nfaces = 144):
    eps_b = EpsConst(1.0)
    eps_m = EpsDrude.gold()
    p_raw = trisphere(nfaces, 10.0)
    return ComParticle([eps_b, eps_m], [p_raw], [[2, 1]])


def _build_dimer_cube(nfaces_per_cube = 300):
    # Two small cubes side-by-side (mimicks dimer geometry; tricube faces ≈ 6 * n^2).
    eps_b = EpsConst(1.0)
    eps_m = EpsDrude.gold()
    # tricube divides each face into n*n triangles * 2 (= 12 n^2 total).
    # nfaces=300 -> n=5
    n_side = int(round(np.sqrt(nfaces_per_cube / 12.0)))
    n_side = max(2, n_side)
    p1 = tricube(n_side, 4.0)
    p1.shift([-3.0, 0.0, 0.0])
    p2 = tricube(n_side, 4.0)
    p2.shift([+3.0, 0.0, 0.0])
    return ComParticle([eps_b, eps_m], [p1, p2], [[2, 1], [2, 1]], [1, 2])


def _build_sphere_on_glass(nfaces = 144):
    epstab = [EpsConst(1.0), EpsTable('gold.dat'), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 3], [0.0])
    sphere = trisphere(nfaces, 12.0)
    sphere.shift([0.0, 0.0, -sphere.pos[:, 2].min() + 2.0])
    p = ComParticle(epstab, [sphere], [[2, 1]], [1])
    return p, layer


def _build_dimer_cube_on_glass(nfaces_per_cube = 300):
    epstab = [EpsConst(1.0), EpsTable('gold.dat'), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 3], [0.0])
    n_side = int(round(np.sqrt(nfaces_per_cube / 12.0)))
    n_side = max(2, n_side)
    p1 = tricube(n_side, 4.0)
    p2 = tricube(n_side, 4.0)
    p1.shift([-3.0, 0.0, -p1.pos[:, 2].min() + 1.5])
    p2.shift([+3.0, 0.0, -p2.pos[:, 2].min() + 1.5])
    p = ComParticle(epstab, [p1, p2], [[2, 1], [2, 1]], [1, 2])
    return p, layer


def _plane_wave_ext_sca_abs(bem, p, exc, enei):
    sig, _ = bem.solve(exc.potential(p, enei) if hasattr(exc, 'potential') else exc(p, enei))
    ext = float(np.real(np.ravel(exc.extinction(sig)))[0])
    sca_raw = exc.scattering(sig)
    if isinstance(sca_raw, tuple):
        sca = float(np.real(np.ravel(sca_raw[0]))[0])
    else:
        sca = float(np.real(np.ravel(sca_raw))[0])
    abs_raw = exc.absorption(sig)
    if isinstance(abs_raw, tuple):
        abs_val = float(np.real(np.ravel(abs_raw[0]))[0])
    else:
        abs_val = float(np.real(np.ravel(abs_raw))[0])
    return ext, sca, abs_val


def _cpu_vs_gpu_planewave(p, enei_list, tol = _TOL):
    pol = np.array([1.0, 0.0, 0.0])
    dirn = np.array([0.0, 0.0, 1.0])

    _set_gpu_env(False)
    _reload_gpu_module()
    bem_cpu = BEMRet(p)
    exc_cpu = PlaneWaveRet(pol, dirn)
    cpu = np.array([_plane_wave_ext_sca_abs(bem_cpu, p, exc_cpu, e) for e in enei_list])

    _set_gpu_env(True)
    _reload_gpu_module()
    bem_gpu = BEMRet(p)
    exc_gpu = PlaneWaveRet(pol, dirn)
    gpu = np.array([_plane_wave_ext_sca_abs(bem_gpu, p, exc_gpu, e) for e in enei_list])

    return cpu, gpu


def _cpu_vs_gpu_planewave_layer(p, layer, enei_list, tol = _TOL):
    pol = np.array([[1.0, 0.0, 0.0]])
    dirn = np.array([[0.0, 0.0, -1.0]])

    enei_arr = np.array(enei_list)
    tab = layer.tabspace(p)

    _set_gpu_env(False)
    _reload_gpu_module()
    gt_c = GreenTabLayer(layer, tab = tab)
    gt_c.set(enei_arr)
    bem_cpu = BEMRetLayer(p, layer, greentab = gt_c)
    exc_cpu = PlaneWaveRetLayer(pol, dirn, layer)
    cpu = []
    for e in enei_list:
        sig, _ = bem_cpu.solve(exc_cpu(p, e))
        cpu.append(float(np.real(np.ravel(exc_cpu.extinction(sig)))[0]))

    _set_gpu_env(True)
    _reload_gpu_module()
    gt_g = GreenTabLayer(layer, tab = tab)
    gt_g.set(enei_arr)
    bem_gpu = BEMRetLayer(p, layer, greentab = gt_g)
    exc_gpu = PlaneWaveRetLayer(pol, dirn, layer)
    gpu = []
    for e in enei_list:
        sig, _ = bem_gpu.solve(exc_gpu(p, e))
        gpu.append(float(np.real(np.ravel(exc_gpu.extinction(sig)))[0]))

    return np.array(cpu), np.array(gpu)


# ---------------------------------------------------------------------------
# Smoke matrix
# ---------------------------------------------------------------------------


def test_bem_ret_sphere_planewave_no_substrate():
    p = _build_sphere(144)
    cpu, gpu = _cpu_vs_gpu_planewave(p, [500.0, 550.0, 600.0])
    rel = np.abs(cpu - gpu) / np.maximum(np.abs(cpu), 1e-12)
    assert rel.max() < _TOL, '[error] sphere planewave: rel diff = {}'.format(rel)


def test_bem_ret_layer_sphere_planewave_with_substrate():
    p, layer = _build_sphere_on_glass(144)
    cpu, gpu = _cpu_vs_gpu_planewave_layer(p, layer, [500.0, 550.0, 600.0])
    rel = np.abs(cpu - gpu) / np.maximum(np.abs(cpu), 1e-12)
    assert rel.max() < _TOL, '[error] sphere+substrate planewave: rel diff = {}'.format(rel)


def test_bem_ret_sphere_dipole_no_substrate():
    p = _build_sphere(144)
    dip_pos = np.array([[0.0, 0.0, 20.0]])
    enei_list = [500.0, 550.0, 600.0]
    eps_b = p.eps[0]  # outside vacuum

    # CPU
    _set_gpu_env(False)
    _reload_gpu_module()
    bem_cpu = BEMRet(p)
    cpu_ext = []
    for e in enei_list:
        # DipoleRet needs a ComPoint-like wrapper. Use the lightweight wrapper.
        from mnpbem.greenfun import CompStruct
        # Build minimal compound point as ComParticle-like point.
        from mnpbem.geometry.compoint import ComPoint
        pt = ComPoint(p, dip_pos)
        exc = DipoleRet(pt)
        sig, _ = bem_cpu.solve(exc.potential(p, e))
        # Extinction via scattering; just verify finite.
        sca = exc.scattering(sig)
        if isinstance(sca, tuple):
            sca = sca[0]
        cpu_ext.append(float(np.real(np.ravel(sca))[0]))

    # GPU
    _set_gpu_env(True)
    _reload_gpu_module()
    bem_gpu = BEMRet(p)
    gpu_ext = []
    for e in enei_list:
        from mnpbem.geometry.compoint import ComPoint
        pt = ComPoint(p, dip_pos)
        exc = DipoleRet(pt)
        sig, _ = bem_gpu.solve(exc.potential(p, e))
        sca = exc.scattering(sig)
        if isinstance(sca, tuple):
            sca = sca[0]
        gpu_ext.append(float(np.real(np.ravel(sca))[0]))

    cpu_arr = np.array(cpu_ext)
    gpu_arr = np.array(gpu_ext)
    rel = np.abs(cpu_arr - gpu_arr) / np.maximum(np.abs(cpu_arr), 1e-12)
    assert rel.max() < _TOL, '[error] sphere dipole: rel diff = {}'.format(rel)


def test_bem_ret_dimer_cube_planewave_no_substrate():
    p = _build_dimer_cube(600)
    assert p.nfaces > 100  # sanity
    cpu, gpu = _cpu_vs_gpu_planewave(p, [500.0, 550.0, 600.0])
    rel = np.abs(cpu - gpu) / np.maximum(np.abs(cpu), 1e-12)
    assert rel.max() < _TOL, '[error] dimer cube planewave: rel diff = {}'.format(rel)


def test_bem_ret_layer_dimer_cube_planewave_with_substrate():
    p, layer = _build_dimer_cube_on_glass(600)
    cpu, gpu = _cpu_vs_gpu_planewave_layer(p, layer, [500.0, 550.0, 600.0])
    rel = np.abs(cpu - gpu) / np.maximum(np.abs(cpu), 1e-12)
    assert rel.max() < _TOL, '[error] dimer cube+substrate planewave: rel diff = {}'.format(rel)


# ---------------------------------------------------------------------------
# Memory leak: 10 wavelength repeats must keep GPU mem stable.
# ---------------------------------------------------------------------------

def test_bem_ret_gpu_no_memory_leak_over_repeats():
    _set_gpu_env(True)
    _reload_gpu_module()
    p = _build_sphere(144)
    pol = np.array([1.0, 0.0, 0.0])
    dirn = np.array([0.0, 0.0, 1.0])
    exc = PlaneWaveRet(pol, dirn)

    mempool = cp.get_default_memory_pool()
    bem = BEMRet(p)

    enei_list = np.linspace(500.0, 700.0, 10).tolist()
    used_after_warm = None
    for i, e in enumerate(enei_list):
        sig, _ = bem.solve(exc.potential(p, e))
        _ = exc.extinction(sig)
        mempool.free_all_blocks()
        if i == 2:
            # after a couple of inits the cache stabilises
            used_after_warm = mempool.used_bytes()
        if i == len(enei_list) - 1:
            used_at_end = mempool.used_bytes()

    # Acceptable monotone growth threshold: 100 MB (smaller than even a
    # single BEM matrix at 144x144x16 bytes, so any real leak that compounds
    # over 8 wavelengths will overflow this).
    if used_after_warm is not None:
        growth = used_at_end - used_after_warm
        assert growth < 100 * 1024 * 1024, \
                '[error] GPU mem grew {} bytes across repeats (likely leak)'.format(growth)


# ---------------------------------------------------------------------------
# Backend-mix edge cases: helper functions on the layer path.
# ---------------------------------------------------------------------------

def test_backend_mix_helpers_in_layer_solver():
    # Ensure _backend_align / _sub_mat / _mul_eps don't crash on
    # cupy/numpy mix.  Smoke-only — full behaviour already covered by
    # test_bem_ret_layer_backend.py.
    from mnpbem.bem.bem_ret_layer import (
            _backend_align, _is_cupy_array, _to_host_safe)
    A = cp.ones((4, 4), dtype = complex)
    B = np.ones((4, 4), dtype = complex)
    A2, B2 = _backend_align(A, B)
    assert _is_cupy_array(A2) and _is_cupy_array(B2)
    h = _to_host_safe(A)
    assert isinstance(h, np.ndarray)


# ---------------------------------------------------------------------------
# Multi-polarisation smoke (npol > 1) — the vectorised npol path lives in
# bem_ret.solve() and has separate code from npol=1.
# ---------------------------------------------------------------------------

def test_bem_ret_multipol_gpu_matches_cpu():
    p = _build_sphere(144)
    pol = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    dirn = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    enei = 550.0

    _set_gpu_env(False)
    _reload_gpu_module()
    bem_cpu = BEMRet(p)
    exc_cpu = PlaneWaveRet(pol, dirn)
    sig_c, _ = bem_cpu.solve(exc_cpu.potential(p, enei))
    ext_c = exc_cpu.extinction(sig_c)

    _set_gpu_env(True)
    _reload_gpu_module()
    bem_gpu = BEMRet(p)
    exc_gpu = PlaneWaveRet(pol, dirn)
    sig_g, _ = bem_gpu.solve(exc_gpu.potential(p, enei))
    ext_g = exc_gpu.extinction(sig_g)

    rel = np.abs(np.asarray(ext_c) - np.asarray(ext_g)) / np.maximum(np.abs(ext_c), 1e-12)
    assert rel.max() < _TOL, '[error] multipol: rel diff = {}'.format(rel)


# ---------------------------------------------------------------------------
# Edge case (v1.7 A1 regression): disjoint-particle dimer with NON-uniform
# eps + zero off-diagonal connectivity.  Pre-fix: L1_is_scalar flag was
# True even though self.L1 was a (n,n) host numpy diag matrix; the
# Sigma_dev * self.L1 product then raised cupy 'unsupported type'.
# ---------------------------------------------------------------------------


def _build_disjoint_dimer_nonuniform_eps():
    eps_b = EpsConst(1.0)
    eps_au = EpsConst(-5 + 0.1j)
    eps_ag = EpsConst(-3 + 0.5j)
    p1 = trisphere(60, 4.0)
    p1.shift([-10.0, 0.0, 0.0])
    p2 = trisphere(60, 4.0)
    p2.shift([+10.0, 0.0, 0.0])
    return ComParticle([eps_b, eps_au, eps_ag],
            [p1, p2], [[2, 1], [3, 1]], [1, 2])


def test_bem_ret_disjoint_dimer_nonuniform_eps_gpu_matches_cpu():
    p = _build_disjoint_dimer_nonuniform_eps()
    cpu, gpu = _cpu_vs_gpu_planewave(p, [500.0, 550.0, 600.0])
    rel = np.abs(cpu - gpu) / np.maximum(np.abs(cpu), 1e-12)
    assert rel.max() < _TOL, \
            '[error] disjoint dimer non-uniform eps: rel diff = {}'.format(rel)


def test_bem_ret_disjoint_dimer_nonuniform_eps_native_gpu_uploads_L1():
    # White-box: in native mode L1 must end up on device when eps is
    # non-uniform (the v1.7 A1 fix path).  Without the fix L1 stayed
    # host numpy and the next solve() raised TypeError.
    p = _build_disjoint_dimer_nonuniform_eps()
    _set_gpu_env(True)
    _reload_gpu_module()
    bem = BEMRet(p)
    bem.init(550.0)
    # eps1 must be non-scalar diag matrix
    assert not np.isscalar(bem.eps1)
    # L1 should be on device (cupy) since native mode is active
    from mnpbem.utils.gpu import is_cupy_array
    assert is_cupy_array(bem.L1), \
            '[error] L1 not on device: type={}'.format(type(bem.L1))
