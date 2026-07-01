"""v1.7 A2 — BEMRetIter + BEMRetLayerIter GPU audit smoke + regression.

5 bug pattern 점검 (이미 v1.6.3/4/5 에서 fix):
  1. backend mix (numpy/cupy)
  2. complex128 dtype 일관성
  3. HMatrix matvec 메모리 누적 (5 wl 반복)
  4. GMRES iterate backend 일관성
  5. precond GPU pipeline OOM 회피

GPU 1 만 사용 — module top-level 에서 CUDA_VISIBLE_DEVICES 설정.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import gc

import numpy as np
import pytest

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, tricube, ComParticle, LayerStructure
from mnpbem.bem import BEMRetIter, BEMRetLayer
from mnpbem.bem.bem_ret_layer_iter import BEMRetLayerIter
from mnpbem.simulation import PlaneWaveRet, PlaneWaveRetLayer
from mnpbem.greenfun import GreenTabLayer


try:
    import cupy as _cp
    _CUPY_OK = True
except ImportError:
    _CUPY_OK = False


_POL = np.array([1.0, 0.0, 0.0])
_DIR = np.array([0.0, 0.0, 1.0])
_POL_LAYER = np.array([[1.0, 0.0, 0.0]])
_DIR_LAYER = np.array([[0.0, 0.0, -1.0]])

_pytestmark_gpu = pytest.mark.skipif(not _CUPY_OK,
        reason = 'cupy unavailable; v1.7 A2 GPU audit requires CUDA runtime')


# ---------------------------------------------------------------------------
# Geometry fixtures (small, fast).
# ---------------------------------------------------------------------------

def _au_sphere_144():
    epstab = [EpsConst(1.0), EpsTable('gold.dat')]
    sphere = trisphere(144, 8.0)
    p = ComParticle(epstab, [sphere], [[2, 1]], [1])
    return p, epstab


def _au_sphere_on_glass_144():
    epstab = [EpsConst(1.0), EpsTable('gold.dat'), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 3], [0.0])
    sphere = trisphere(144, 8.0)
    sphere.shift([0.0, 0.0, -sphere.pos[:, 2].min() + 1.0])
    p = ComParticle(epstab, [sphere], [[2, 1]], [1])
    return p, layer, epstab


def _au_dimer_cube_1176():
    # tricube produces ~6*N^2 + 8*N + 2 vertices, but face count differs.
    # Aim ~588/cube to total 1176; tricube(10) gives a typical 588-face cube.
    epstab = [EpsConst(1.0), EpsTable('gold.dat')]
    p1 = tricube(10, 10.0)
    p2 = tricube(10, 10.0)
    p1.shift([-6.0, 0.0, 0.0])
    p2.shift([+6.0, 0.0, 0.0])
    p = ComParticle(epstab, [p1, p2], [[2, 1], [2, 1]], [1, 2])
    return p, epstab


def _au_dimer_cube_on_glass_1176():
    epstab = [EpsConst(1.0), EpsTable('gold.dat'), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 3], [0.0])
    p1 = tricube(10, 10.0)
    p2 = tricube(10, 10.0)
    p1.shift([-6.0, 0.0, 0.0])
    p2.shift([+6.0, 0.0, 0.0])
    z_min = min(p1.pos[:, 2].min(), p2.pos[:, 2].min())
    p1.shift([0.0, 0.0, -z_min + 1.0])
    p2.shift([0.0, 0.0, -z_min + 1.0])
    p = ComParticle(epstab, [p1, p2], [[2, 1], [2, 1]], [1, 2])
    return p, layer, epstab


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpu_mem_free_gb():
    free, _total = _cp.cuda.runtime.memGetInfo()
    return free / (1024 ** 3)


def _drain_pool():
    if _CUPY_OK:
        try:
            _cp.get_default_memory_pool().free_all_blocks()
            _cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
        gc.collect()


def _gpu_env(active = True):
    # context: enables MNPBEM_GPU + cupy path
    prev = os.environ.get('MNPBEM_GPU')
    if active:
        os.environ['MNPBEM_GPU'] = '1'
    else:
        os.environ.pop('MNPBEM_GPU', None)
    return prev


def _gpu_env_restore(prev):
    if prev is None:
        os.environ.pop('MNPBEM_GPU', None)
    else:
        os.environ['MNPBEM_GPU'] = prev


# ---------------------------------------------------------------------------
# Smoke matrix — sphere 144
# ---------------------------------------------------------------------------

@_pytestmark_gpu
def test_sphere_144_iter_gpu_smoke():
    """Sphere 144 face, no substrate, GPU path."""
    p, _ = _au_sphere_144()
    exc = PlaneWaveRet(_POL, _DIR)

    prev = _gpu_env(active = True)
    try:
        _drain_pool()
        bem = BEMRetIter(p, tol = 1e-6, maxit = 200)
        for enei in (540.0, 600.0, 660.0):
            sig, _bem = bem.solve(exc.potential(p, enei))
            ext = float(np.real(np.ravel(exc.extinction(sig))[0]))
            assert np.isfinite(ext) and ext > 0, \
                '[error] sphere144 ext bad at enei={}: {}'.format(enei, ext)
    finally:
        _gpu_env_restore(prev)
        _drain_pool()


@_pytestmark_gpu
def test_sphere_144_iter_layer_gpu_smoke():
    """Sphere 144 face, glass substrate, layer iter GPU path."""
    p, layer, _ = _au_sphere_on_glass_144()
    enei_arr = np.array([540.0, 600.0, 660.0])
    tab = layer.tabspace(p)
    gt = GreenTabLayer(layer, tab = tab)
    gt.set(enei_arr)
    exc = PlaneWaveRetLayer(_POL_LAYER, _DIR_LAYER, layer)

    prev = _gpu_env(active = True)
    try:
        _drain_pool()
        bem = BEMRetLayerIter(p, layer, greentab = gt, tol = 1e-6, maxit = 200)
        for enei in enei_arr:
            sig, _bem = bem.solve(exc(p, float(enei)))
            ext = float(np.real(np.ravel(exc.extinction(sig)))[0])
            assert np.isfinite(ext) and ext > 0, \
                '[error] sphere144+layer ext bad at enei={}: {}'.format(enei, ext)
    finally:
        _gpu_env_restore(prev)
        _drain_pool()


# ---------------------------------------------------------------------------
# Bug pattern 1: backend mix — _afun + _mfun must return numpy
# ---------------------------------------------------------------------------

@_pytestmark_gpu
def test_bug1_afun_mfun_returns_numpy_iter():
    """BEMRetIter._afun and _mfun (with GPU LU) must return numpy."""
    p, _ = _au_sphere_144()
    prev = _gpu_env(active = True)
    try:
        _drain_pool()
        bem = BEMRetIter(p, enei = 600.0, tol = 1e-6, maxit = 200)
        n = p.n
        vec = (np.random.randn(8 * n) + 1j * np.random.randn(8 * n)).astype(complex)
        out_a = bem._afun(vec)
        out_m = bem._mfun(vec)
        assert isinstance(out_a, np.ndarray) and not (
                hasattr(out_a, 'get') and not isinstance(out_a, np.ndarray)), \
                '[error] _afun returned cupy ndarray'
        assert out_a.dtype == np.complex128, \
                '[error] _afun dtype regression: {}'.format(out_a.dtype)
        assert isinstance(out_m, np.ndarray) and not (
                hasattr(out_m, 'get') and not isinstance(out_m, np.ndarray)), \
                '[error] _mfun returned cupy ndarray'
        assert out_m.dtype == np.complex128, \
                '[error] _mfun dtype regression: {}'.format(out_m.dtype)
    finally:
        _gpu_env_restore(prev)
        _drain_pool()


@_pytestmark_gpu
def test_bug1_afun_mfun_returns_numpy_layer_iter():
    """BEMRetLayerIter._afun and _mfun (with GPU LU in precond) must return numpy."""
    p, layer, _ = _au_sphere_on_glass_144()
    enei = 600.0
    tab = layer.tabspace(p)
    gt = GreenTabLayer(layer, tab = tab)
    gt.set(np.array([enei]))

    prev = _gpu_env(active = True)
    try:
        _drain_pool()
        bem = BEMRetLayerIter(p, layer, greentab = gt, enei = enei,
                tol = 1e-6, maxit = 200)
        n = p.n
        vec = (np.random.randn(8 * n) + 1j * np.random.randn(8 * n)).astype(complex)
        out_a = bem._afun(vec)
        out_m = bem._mfun(vec)
        for name, out in (('_afun', out_a), ('_mfun', out_m)):
            assert isinstance(out, np.ndarray) and not (
                    hasattr(out, 'get') and not isinstance(out, np.ndarray)), \
                    '[error] Layer {} returned cupy ndarray'.format(name)
            assert out.dtype == np.complex128, \
                    '[error] Layer {} dtype regression: {}'.format(name, out.dtype)
    finally:
        _gpu_env_restore(prev)
        _drain_pool()


# ---------------------------------------------------------------------------
# Bug pattern 3: memory accumulation over 5-wavelength sweep
#   — peak free-memory after each iterate must not collapse (no leak).
# ---------------------------------------------------------------------------

@_pytestmark_gpu
def test_bug3_5wl_no_memory_leak_iter_sphere():
    """5 wavelength sweep — memory must not drop monotonically (leak indicator)."""
    p, _ = _au_sphere_144()
    exc = PlaneWaveRet(_POL, _DIR)
    enei_arr = np.linspace(540.0, 660.0, 5)

    prev = _gpu_env(active = True)
    try:
        _drain_pool()
        baseline_free = _gpu_mem_free_gb()

        bem = BEMRetIter(p, tol = 1e-6, maxit = 200)
        free_per_wl = []
        for enei in enei_arr:
            sig, _bem = bem.solve(exc.potential(p, float(enei)))
            _drain_pool()
            free_per_wl.append(_gpu_mem_free_gb())
        # The free memory should not collapse over 5 wavelengths.  We tolerate
        # a small dip (<= 1 GB) for fragmented blocks held by cupy pool.
        worst_drop_gb = baseline_free - min(free_per_wl)
        assert worst_drop_gb < 1.0, \
                '[error] memory leak suspected: baseline={:.2f} GB, ' \
                'worst free over sweep={:.2f} GB, drop={:.2f} GB'.format(
                        baseline_free, min(free_per_wl), worst_drop_gb)
    finally:
        _gpu_env_restore(prev)
        _drain_pool()


# ---------------------------------------------------------------------------
# Bug pattern 5: 12672-face dimer GPU path completes without OOM.
#   — guard behind env flag MNPBEM_RUN_LARGE so the suite stays fast by
#     default; the parent script can opt in.
# ---------------------------------------------------------------------------

def _build_au_ag_dimer_12672():
    epstab = [EpsConst(1.77), EpsTable('gold.dat'), EpsTable('silver.dat')]
    core_d = 24.0
    shell_t = 2.0
    outer_d = core_d + 2.0 * shell_t
    gap = 2.0
    half = (outer_d + gap) / 2.0
    # trisphere with ~3168 faces per shell -> 4 shells x 3168 = 12672
    n_face = 3168
    p1_shell = trisphere(n_face, outer_d); p1_core = trisphere(n_face, core_d)
    p1_shell.shift([-half, 0.0, 0.0]); p1_core.shift([-half, 0.0, 0.0])
    p2_shell = trisphere(n_face, outer_d); p2_core = trisphere(n_face, core_d)
    p2_shell.shift([+half, 0.0, 0.0]); p2_core.shift([+half, 0.0, 0.0])
    inds = [[3, 1], [2, 3], [3, 1], [2, 3]]
    p = ComParticle(epstab, [p1_shell, p1_core, p2_shell, p2_core],
            inds, 1, 2, interp = 'curv')
    return p, epstab


@_pytestmark_gpu
@pytest.mark.skipif(os.environ.get('MNPBEM_RUN_LARGE', '0') != '1',
        reason = '12672-face case gated by MNPBEM_RUN_LARGE=1')
def test_bug5_12672_no_substrate_completes():
    p, _ = _build_au_ag_dimer_12672()
    assert p.n == 12672, \
            '[error] expected 12672-face dimer, got {}'.format(p.n)
    exc = PlaneWaveRet(_POL, _DIR)

    prev = _gpu_env(active = True)
    try:
        _drain_pool()
        baseline_free = _gpu_mem_free_gb()
        bem = BEMRetIter(p, hmatrix = True, htol = 1e-6, tol = 1e-6,
                maxit = 200, preconditioner = 'auto')
        sig, _bem = bem.solve(exc.potential(p, 600.0))
        ext = float(np.real(np.ravel(exc.extinction(sig))[0]))
        assert np.isfinite(ext) and ext > 0, \
                '[error] 12672 ext bad: {}'.format(ext)
        _drain_pool()
        peak_drop_gb = baseline_free - _gpu_mem_free_gb()
        assert peak_drop_gb < 30.0, \
                '[error] 12672 face GPU mem peak drop > 30 GB: {:.2f}'.format(
                        peak_drop_gb)
    finally:
        _gpu_env_restore(prev)
        _drain_pool()
