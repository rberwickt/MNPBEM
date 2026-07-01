"""Unit tests for VRAM-share env var auto-wiring (v1.6.2).

Validates that ``lu_factor_dispatch`` reads ``MNPBEM_VRAM_SHARE_GPUS``,
``MNPBEM_VRAM_SHARE_BACKEND``, and ``MNPBEM_VRAM_SHARE_DEVICE_IDS`` when
no explicit kwargs are passed, and that explicit kwargs always win.

The tests do NOT require CUDA / cupy. Instead, they monkeypatch
``mnpbem.utils.multi_gpu_lu.factor_multi_gpu`` and
``cusolvermg_available`` so the dispatch path is exercised purely on
CPU and the captured arguments can be asserted.
"""

from __future__ import annotations

import os
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubMultiGPULU(object):
    """Minimal stand-in for the cuSolverMg-backed handle."""

    def __init__(self, A, n_gpus, backend, device_ids):
        self.A = A
        self.n_gpus = n_gpus
        self.backend = backend
        self.device_ids = device_ids

    def solve(self, b, trans='N'):
        # Identity-like passthrough; the tests only check dispatch routing.
        return np.asarray(b)


@pytest.fixture
def patch_multi_gpu(monkeypatch):
    """Replace cuSolverMg backend with a CPU-friendly stub."""
    captured = {}

    def fake_factor(A, n_gpus, backend, device_ids):
        captured['A_shape'] = A.shape
        captured['n_gpus'] = n_gpus
        captured['backend'] = backend
        captured['device_ids'] = device_ids
        return _StubMultiGPULU(A, n_gpus, backend, device_ids)

    def fake_available():
        return True

    def fake_warn(msg):
        captured.setdefault('warnings', []).append(msg)

    import GUI.mnpbem.utils.multi_gpu_lu as mgl
    monkeypatch.setattr(mgl, 'factor_multi_gpu', fake_factor)
    monkeypatch.setattr(mgl, 'cusolvermg_available', fake_available)
    monkeypatch.setattr(mgl, 'warn_fallback', fake_warn)
    return captured


@pytest.fixture
def clean_env(monkeypatch):
    """Remove all MNPBEM_VRAM_SHARE_* env vars for a deterministic baseline."""
    for key in (
        'MNPBEM_VRAM_SHARE',
        'MNPBEM_VRAM_SHARE_GPUS',
        'MNPBEM_VRAM_SHARE_BACKEND',
        'MNPBEM_VRAM_SHARE_DEVICE_IDS',
    ):
        monkeypatch.delenv(key, raising=False)
    return None


# ---------------------------------------------------------------------------
# Env-var helper itself
# ---------------------------------------------------------------------------


def test_env_defaults_unset_returns_none(clean_env):
    from GUI.mnpbem.utils.gpu import _vram_share_env_defaults
    n, b, d = _vram_share_env_defaults()
    assert n is None and b is None and d is None


def test_env_defaults_n_only(monkeypatch, clean_env):
    monkeypatch.setenv('MNPBEM_VRAM_SHARE_GPUS', '4')
    from GUI.mnpbem.utils.gpu import _vram_share_env_defaults
    n, b, d = _vram_share_env_defaults()
    assert n == 4
    assert b == 'cusolvermg'
    assert d is None


def test_env_defaults_full_set(monkeypatch, clean_env):
    monkeypatch.setenv('MNPBEM_VRAM_SHARE_GPUS', '2')
    monkeypatch.setenv('MNPBEM_VRAM_SHARE_BACKEND', 'cusolvermg')
    monkeypatch.setenv('MNPBEM_VRAM_SHARE_DEVICE_IDS', '0,2,3,5')
    from GUI.mnpbem.utils.gpu import _vram_share_env_defaults
    n, b, d = _vram_share_env_defaults()
    assert n == 2
    assert b == 'cusolvermg'
    assert d == [0, 2, 3, 5]


def test_env_defaults_master_off_disables(monkeypatch, clean_env):
    monkeypatch.setenv('MNPBEM_VRAM_SHARE', '0')
    monkeypatch.setenv('MNPBEM_VRAM_SHARE_GPUS', '4')
    from GUI.mnpbem.utils.gpu import _vram_share_env_defaults
    n, b, d = _vram_share_env_defaults()
    assert n is None and b is None and d is None


def test_env_defaults_n1_disables(monkeypatch, clean_env):
    monkeypatch.setenv('MNPBEM_VRAM_SHARE_GPUS', '1')
    from GUI.mnpbem.utils.gpu import _vram_share_env_defaults
    n, b, d = _vram_share_env_defaults()
    assert n is None


def test_env_defaults_invalid_value_safe(monkeypatch, clean_env):
    monkeypatch.setenv('MNPBEM_VRAM_SHARE_GPUS', 'not_a_number')
    from GUI.mnpbem.utils.gpu import _vram_share_env_defaults
    n, _, _ = _vram_share_env_defaults()
    assert n is None


# ---------------------------------------------------------------------------
# lu_factor_dispatch routing via env var
# ---------------------------------------------------------------------------


def test_dispatch_no_env_no_kwarg_takes_cpu(clean_env, patch_multi_gpu):
    from GUI.mnpbem.utils.gpu import lu_factor_dispatch
    A = np.eye(8, dtype=np.complex128)
    pkg = lu_factor_dispatch(A)
    assert pkg[0] == 'cpu'
    assert 'n_gpus' not in patch_multi_gpu


def test_dispatch_env_only_routes_mgpu(monkeypatch, clean_env, patch_multi_gpu):
    monkeypatch.setenv('MNPBEM_VRAM_SHARE_GPUS', '4')
    from GUI.mnpbem.utils.gpu import lu_factor_dispatch
    A = np.eye(8, dtype=np.complex128)
    pkg = lu_factor_dispatch(A)
    assert pkg[0] == 'mgpu'
    assert patch_multi_gpu['n_gpus'] == 4
    assert patch_multi_gpu['backend'] == 'cusolvermg'
    assert patch_multi_gpu['device_ids'] is None


def test_dispatch_env_with_backend(monkeypatch, clean_env, patch_multi_gpu):
    monkeypatch.setenv('MNPBEM_VRAM_SHARE_GPUS', '3')
    monkeypatch.setenv('MNPBEM_VRAM_SHARE_BACKEND', 'cusolvermg')
    monkeypatch.setenv('MNPBEM_VRAM_SHARE_DEVICE_IDS', '0,1,2')
    from GUI.mnpbem.utils.gpu import lu_factor_dispatch
    A = np.eye(4, dtype=np.complex128)
    pkg = lu_factor_dispatch(A)
    assert pkg[0] == 'mgpu'
    assert patch_multi_gpu['n_gpus'] == 3
    assert patch_multi_gpu['device_ids'] == [0, 1, 2]


def test_dispatch_kwarg_overrides_env(monkeypatch, clean_env, patch_multi_gpu):
    monkeypatch.setenv('MNPBEM_VRAM_SHARE_GPUS', '4')
    from GUI.mnpbem.utils.gpu import lu_factor_dispatch
    A = np.eye(4, dtype=np.complex128)
    pkg = lu_factor_dispatch(A, n_gpus=2)
    assert pkg[0] == 'mgpu'
    # Explicit kwarg wins over env (4 -> 2).
    assert patch_multi_gpu['n_gpus'] == 2


def test_dispatch_kwarg_n1_overrides_env_off(monkeypatch, clean_env, patch_multi_gpu):
    monkeypatch.setenv('MNPBEM_VRAM_SHARE_GPUS', '4')
    from GUI.mnpbem.utils.gpu import lu_factor_dispatch
    A = np.eye(4, dtype=np.complex128)
    pkg = lu_factor_dispatch(A, n_gpus=1)
    # Explicit n_gpus=1 should disable mgpu path even when env says 4.
    assert pkg[0] == 'cpu'


def test_dispatch_master_off(monkeypatch, clean_env, patch_multi_gpu):
    monkeypatch.setenv('MNPBEM_VRAM_SHARE', '0')
    monkeypatch.setenv('MNPBEM_VRAM_SHARE_GPUS', '4')
    from GUI.mnpbem.utils.gpu import lu_factor_dispatch
    A = np.eye(4, dtype=np.complex128)
    pkg = lu_factor_dispatch(A)
    assert pkg[0] == 'cpu'


# ---------------------------------------------------------------------------
# lu_solve_dispatch end-to-end via env-routed factor
# ---------------------------------------------------------------------------


def test_solve_after_env_factor_returns_numpy(monkeypatch, clean_env, patch_multi_gpu):
    monkeypatch.setenv('MNPBEM_VRAM_SHARE_GPUS', '2')
    from GUI.mnpbem.utils.gpu import lu_factor_dispatch, lu_solve_dispatch
    A = np.eye(4, dtype=np.complex128)
    b = np.ones(4, dtype=np.complex128)
    pkg = lu_factor_dispatch(A)
    assert pkg[0] == 'mgpu'
    x = lu_solve_dispatch(pkg, b)
    assert isinstance(x, np.ndarray)


# ---------------------------------------------------------------------------
# solve_dispatch one-shot path
# ---------------------------------------------------------------------------


def test_solve_dispatch_env_routes_through_mgpu(monkeypatch, clean_env, patch_multi_gpu):
    monkeypatch.setenv('MNPBEM_VRAM_SHARE_GPUS', '2')
    from GUI.mnpbem.utils.gpu import solve_dispatch
    A = np.eye(4, dtype=np.complex128)
    b = np.ones(4, dtype=np.complex128)
    x = solve_dispatch(A, b)
    assert isinstance(x, np.ndarray)
    # The fake mgpu solve returns b unchanged (identity stub), and the
    # captured n_gpus must reflect the env var.
    assert patch_multi_gpu['n_gpus'] == 2


def test_solve_dispatch_no_env_takes_cpu(clean_env, patch_multi_gpu):
    from GUI.mnpbem.utils.gpu import solve_dispatch
    A = np.eye(4, dtype=np.complex128) * 2.0
    b = np.ones(4, dtype=np.complex128) * 2.0
    x = solve_dispatch(A, b)
    np.testing.assert_allclose(x, np.ones(4, dtype=np.complex128))
    assert 'n_gpus' not in patch_multi_gpu
