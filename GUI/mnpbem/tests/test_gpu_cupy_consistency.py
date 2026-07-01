import os
import sys

from typing import Any

import numpy as np
import pytest


# Try import cupy; tests are conditionally skipped when unavailable.
try:
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    cp = None  # type: ignore
    _HAS_CUPY = False


cupy_required = pytest.mark.skipif(not _HAS_CUPY, reason = 'cupy not installed')


# ---------------------------------------------------------------------------
# Bug 1 / 2: gpu helpers.  These must hold even on CPU-only hosts.
# ---------------------------------------------------------------------------

def test_eye_like_lu_cpu_returns_numpy():

    from mnpbem.utils.gpu import eye_like_lu

    fake_cpu_lu = ('cpu', None, None)
    eye_h = eye_like_lu(fake_cpu_lu, 5)
    assert isinstance(eye_h, np.ndarray)
    assert eye_h.shape == (5, 5)


def test_to_host_passthrough_numpy():

    from mnpbem.utils.gpu import to_host

    a = np.arange(6).reshape(2, 3)
    b = to_host(a)
    assert isinstance(b, np.ndarray)
    assert np.array_equal(a, b)


def test_is_cupy_array_false_for_numpy():

    from mnpbem.utils.gpu import is_cupy_array

    assert is_cupy_array(np.zeros(3)) is False
    assert is_cupy_array([1, 2, 3]) is False


@cupy_required
def test_eye_like_lu_gpu_returns_cupy():

    from mnpbem.utils.gpu import eye_like_lu

    fake_gpu_lu = ('gpu', cp.zeros((4, 4)), cp.zeros(4, dtype = cp.int32))
    eye_d = eye_like_lu(fake_gpu_lu, 4)
    assert isinstance(eye_d, cp.ndarray)


@cupy_required
def test_to_host_brings_cupy_to_numpy():

    from mnpbem.utils.gpu import to_host

    a = cp.arange(6).reshape(2, 3)
    h = to_host(a)
    assert isinstance(h, np.ndarray)
    assert h.shape == (2, 3)


@cupy_required
def test_is_cupy_array_true_for_cupy():

    from mnpbem.utils.gpu import is_cupy_array

    assert is_cupy_array(cp.zeros(3)) is True


@cupy_required
def test_lu_solve_native_keeps_cupy_when_b_is_cupy():
    # Bug 1 root cause: lu_solve_dispatch round-tripped cupy → numpy.
    # lu_solve_native should keep a cupy result when b is cupy.
    from mnpbem.utils.gpu import lu_factor_dispatch, lu_solve_native

    n = 64
    A = np.eye(n) + 0.01 * np.random.randn(n, n)
    A_pkg = lu_factor_dispatch(A)  # CPU LU below threshold
    b_cp = cp.eye(n)
    x = lu_solve_native(A_pkg, b_cp)
    # CPU LU with cupy b: result is numpy (lu_solve_native brings b to host).
    assert isinstance(x, np.ndarray)


@cupy_required
def test_lu_solve_native_keeps_cupy_on_gpu_lu():
    # v1.7 Phase 1.3 audit: confirm GPU LU + cupy b returns cupy (no host
    # round-trip).  Callers in bem_ret.py rely on this when MNPBEM_GPU_NATIVE
    # is active so downstream cupy broadcast ops do not mix host/device.
    import os
    from mnpbem.utils.gpu import lu_factor_dispatch, lu_solve_native

    old_gpu = os.environ.get('MNPBEM_GPU')
    old_threshold = os.environ.get('MNPBEM_GPU_THRESHOLD')
    os.environ['MNPBEM_GPU'] = '1'
    os.environ['MNPBEM_GPU_THRESHOLD'] = '10'
    try:
        # Force re-detection of USE_GPU / GPU_THRESHOLD by reloading
        import importlib
        from mnpbem.utils import gpu as gpu_mod
        importlib.reload(gpu_mod)

        n = 64
        A = np.eye(n) + 0.01 * np.random.randn(n, n)
        A_pkg = gpu_mod.lu_factor_dispatch(A)
        assert A_pkg[0] == 'gpu', 'expected GPU LU, got tag={}'.format(A_pkg[0])
        b_cp = cp.eye(n)
        x = gpu_mod.lu_solve_native(A_pkg, b_cp)
        assert isinstance(x, cp.ndarray), 'GPU LU + cupy b must return cupy'
    finally:
        if old_gpu is None:
            os.environ.pop('MNPBEM_GPU', None)
        else:
            os.environ['MNPBEM_GPU'] = old_gpu
        if old_threshold is None:
            os.environ.pop('MNPBEM_GPU_THRESHOLD', None)
        else:
            os.environ['MNPBEM_GPU_THRESHOLD'] = old_threshold
        import importlib
        from mnpbem.utils import gpu as gpu_mod
        importlib.reload(gpu_mod)


# ---------------------------------------------------------------------------
# Bug 1: BEMRet CPU init path consistency under MNPBEM_GPU=0.
# ---------------------------------------------------------------------------

def _build_small_sphere(nfaces_target: int = 144) -> Any:

    from mnpbem.geometry import ComParticle, trisphere
    from mnpbem.materials import EpsConst, EpsDrude

    eps_b = EpsConst(1.0)
    eps_m = EpsDrude.gold()
    p_raw = trisphere(nfaces_target, 10.0)
    p = ComParticle([eps_b, eps_m], [p_raw], [[2, 1]])
    return p


def test_bemret_cpu_init_smoke():
    # Bug 1 / 2 regression: BEMRet on a tiny sphere must initialize and
    # solve without raising the cupy/numpy mix error.  Runs on CPU only
    # (MNPBEM_GPU not set / 0) so it is portable.
    from mnpbem.bem import BEMRet

    p = _build_small_sphere()
    bem = BEMRet(p)
    bem.init(550.0)
    assert bem.G1_lu is not None
    assert bem.G2_lu is not None
    assert bem.Sigma_lu is not None


# ---------------------------------------------------------------------------
# Bug 3: hmatrix._aca_block must tolerate cupy index inputs.
# ---------------------------------------------------------------------------

def test_aca_block_accepts_numpy_indices():

    from mnpbem.greenfun.hmatrix import HMatrix

    # Use a synthetic low-rank matrix so ACA terminates quickly.
    rank = 3
    m_full, n_full = 32, 28
    rng = np.random.default_rng(0)
    U_true = rng.standard_normal((m_full, rank))
    V_true = rng.standard_normal((n_full, rank))
    A_full = U_true @ V_true.T

    def fun(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return A_full[np.asarray(rows, dtype = np.int64),
                np.asarray(cols, dtype = np.int64)]

    h = HMatrix.__new__(HMatrix)  # bypass __init__
    rows = np.arange(m_full, dtype = np.int64)
    cols = np.arange(n_full, dtype = np.int64)
    U, V = h._aca_block(fun, rows, cols, htol = 1e-10, kmax = rank * 2)
    err = np.linalg.norm(A_full - U @ V.T) / np.linalg.norm(A_full)
    assert err < 1e-8, '[error] ACA reconstruction failed: err={:.2e}'.format(err)


@cupy_required
def test_aca_block_accepts_cupy_indices():
    # Bug 3 regression: pass cupy index arrays.  Coercion inside
    # _aca_block must succeed without raising the implicit-numpy index
    # error and reconstruction must remain bit-equivalent.
    from mnpbem.greenfun.hmatrix import HMatrix

    rank = 2
    m_full, n_full = 24, 20
    rng = np.random.default_rng(1)
    U_true = rng.standard_normal((m_full, rank))
    V_true = rng.standard_normal((n_full, rank))
    A_full = U_true @ V_true.T

    def fun(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return A_full[np.asarray(rows, dtype = np.int64),
                np.asarray(cols, dtype = np.int64)]

    h = HMatrix.__new__(HMatrix)
    rows_g = cp.arange(m_full, dtype = cp.int64)
    cols_g = cp.arange(n_full, dtype = cp.int64)
    U, V = h._aca_block(fun, rows_g, cols_g, htol = 1e-10, kmax = rank * 2)
    err = np.linalg.norm(A_full - U @ V.T) / np.linalg.norm(A_full)
    assert err < 1e-8, '[error] ACA (cupy idx) failed: err={:.2e}'.format(err)


# ---------------------------------------------------------------------------
# Bug 4: solve_spectrum_multi_gpu accepts bem_class.
# ---------------------------------------------------------------------------

def test_resolve_bem_class_strings():

    from mnpbem.utils.multi_gpu import _resolve_bem_class
    from mnpbem.bem import BEMRet, BEMRetIter, BEMRetLayer, BEMRetLayerIter

    assert _resolve_bem_class(None) is BEMRet
    assert _resolve_bem_class('BEMRet') is BEMRet
    assert _resolve_bem_class('BEMRetIter') is BEMRetIter
    assert _resolve_bem_class('BEMRetLayer') is BEMRetLayer
    assert _resolve_bem_class('BEMRetLayerIter') is BEMRetLayerIter


def test_resolve_bem_class_invalid_raises():

    from mnpbem.utils.multi_gpu import _resolve_bem_class

    with pytest.raises(ValueError, match = '\\[error\\]'):
        _resolve_bem_class('NoSuchSolver')


def test_solve_spectrum_multi_gpu_signature_accepts_bem_class():
    # Bug 4 regression: signature must accept bem_class kwarg without
    # touching CUDA.  Run with n_gpus=1 and a 0-length wavelength array
    # so the dispatcher exits before forking workers.
    import inspect

    from mnpbem.utils.multi_gpu import solve_spectrum_multi_gpu

    sig = inspect.signature(solve_spectrum_multi_gpu)
    assert 'bem_class' in sig.parameters, \
        '[error] solve_spectrum_multi_gpu must accept <bem_class>'


def test_solve_spectrum_multi_gpu_bem_class_name_passthrough():
    # Stub the spawn process so we can verify bem_class_name actually
    # reaches the worker arg list.
    import multiprocessing as mp

    from mnpbem.utils import multi_gpu as mg
    from mnpbem.bem import BEMRetIter

    captured = []

    class _DummyProc(object):

        def __init__(self,
                target: Any = None,
                args: tuple = ()) -> None:

            captured.append(args)

        def start(self) -> None:
            pass

        def join(self,
                timeout: Any = None) -> None:
            pass

    class _DummyCtx(object):

        @staticmethod
        def Process(target: Any = None,
                args: tuple = ()) -> _DummyProc:
            return _DummyProc(target, args)

        @staticmethod
        def Queue() -> Any:

            class _Q(object):

                def __init__(self) -> None:
                    self._items = []

                def put(self, item: Any) -> None:
                    self._items.append(item)

                def get(self) -> Any:
                    # Return a no-op success record so the driver returns.
                    return {
                        'gpu_idx': 0,
                        'wl_indices': [0],
                        'ext': np.zeros((1, 1)),
                        'sca': np.zeros((1, 1)),
                        'wall_s': 0.0,
                        'ok': True,
                    }

            return _Q()

    orig_get_context = mp.get_context
    mp.get_context = lambda *a, **k: _DummyCtx()
    try:
        mg.solve_spectrum_multi_gpu(
            particle_factory = lambda: None,
            enei = [500.0],
            pol_dirs = [[1, 0, 0]],
            prop_dirs = [[0, 0, 1]],
            n_gpus = 1,
            bem_class = BEMRetIter,
        )
    finally:
        mp.get_context = orig_get_context

    assert len(captured) == 1
    args = captured[0]
    # _worker signature: (gpu_idx, wl_indices, enei_chunk, particle_factory,
    #   pol_dirs, prop_dirs, queue, bem_kwargs, bem_class_name)
    assert args[-1] == 'BEMRetIter', \
        '[error] bem_class_name was not propagated to the worker'
