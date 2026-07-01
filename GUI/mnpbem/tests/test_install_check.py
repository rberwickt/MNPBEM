import os
import sys
import warnings
from typing import Any

import pytest

import mnpbem.utils.gpu as gpu_mod
from mnpbem.utils.gpu import (
    get_install_hint,
    has_gpu_capability,
    require_gpu_or_raise,
)


def test_get_install_hint_lists_all_extras():
    hint = get_install_hint()
    assert isinstance(hint, str)
    assert 'mnpbem[gpu]' in hint
    assert 'mnpbem[mpi]' in hint
    assert 'mnpbem[fmm]' in hint
    assert 'mnpbem[all]' in hint
    assert 'docs/INSTALL.md' in hint


def test_has_gpu_capability_returns_bool():
    result = has_gpu_capability(verbose=False)
    assert isinstance(result, bool)


def test_has_gpu_capability_no_cupy_returns_false(monkeypatch):
    monkeypatch.setattr(gpu_mod, '_CUPY_OK', False)
    monkeypatch.setattr(gpu_mod, '_CUPY_IMPORT_ERROR', "ImportError('no cupy')")
    monkeypatch.setattr(gpu_mod, '_cp', None)
    assert has_gpu_capability(verbose=False) is False


def test_has_gpu_capability_no_cupy_emits_warning(monkeypatch):
    monkeypatch.setattr(gpu_mod, '_CUPY_OK', False)
    monkeypatch.setattr(gpu_mod, '_CUPY_IMPORT_ERROR', "ImportError('no cupy')")
    monkeypatch.setattr(gpu_mod, '_cp', None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = has_gpu_capability(verbose=True)
    assert result is False
    assert len(caught) >= 1
    msgs = [str(w.message) for w in caught]
    assert any('cupy is not importable' in m for m in msgs)
    assert any('mnpbem[gpu]' in m for m in msgs)


def test_has_gpu_capability_runtime_check_failure(monkeypatch):

    class _BrokenRuntime(object):

        @staticmethod
        def getDeviceCount() -> int:
            raise RuntimeError('cuda runtime borked')


    class _BrokenCuda(object):

        runtime = _BrokenRuntime()


    class _BrokenCupy(object):

        cuda = _BrokenCuda()

    monkeypatch.setattr(gpu_mod, '_CUPY_OK', True)
    monkeypatch.setattr(gpu_mod, '_cp', _BrokenCupy())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = has_gpu_capability(verbose=True)
    assert result is False
    assert any('CUDA runtime check failed' in str(w.message) for w in caught)


def test_has_gpu_capability_zero_devices(monkeypatch):

    class _ZeroRuntime(object):

        @staticmethod
        def getDeviceCount() -> int:
            return 0


    class _ZeroCuda(object):

        runtime = _ZeroRuntime()


    class _ZeroCupy(object):

        cuda = _ZeroCuda()

    monkeypatch.setattr(gpu_mod, '_CUPY_OK', True)
    monkeypatch.setattr(gpu_mod, '_cp', _ZeroCupy())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = has_gpu_capability(verbose=True)
    assert result is False
    assert any('no CUDA devices' in str(w.message) for w in caught)


def test_require_gpu_or_raise_no_op_when_disabled(monkeypatch):
    monkeypatch.setattr(gpu_mod, 'USE_GPU', False)
    monkeypatch.setattr(gpu_mod, '_CUPY_OK', False)
    require_gpu_or_raise()


def test_require_gpu_or_raise_no_op_when_cupy_present(monkeypatch):
    monkeypatch.setattr(gpu_mod, 'USE_GPU', True)
    monkeypatch.setattr(gpu_mod, '_CUPY_OK', True)
    require_gpu_or_raise()


def test_require_gpu_or_raise_raises_with_install_hint(monkeypatch):
    monkeypatch.setattr(gpu_mod, 'USE_GPU', True)
    monkeypatch.setattr(gpu_mod, '_CUPY_OK', False)
    monkeypatch.setattr(gpu_mod, '_CUPY_IMPORT_ERROR', "ImportError('no cupy')")
    with pytest.raises(RuntimeError) as excinfo:
        require_gpu_or_raise()
    msg = str(excinfo.value)
    assert 'MNPBEM_GPU=1' in msg
    assert 'mnpbem[gpu]' in msg
    assert 'docs/INSTALL.md' in msg


def test_extras_documented_in_pyproject():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    pyproject = (root / 'pyproject.toml').read_text(encoding='utf-8')
    assert 'gpu = [' in pyproject
    assert 'mpi = [' in pyproject
    assert 'fmm = [' in pyproject
    assert 'all = [' in pyproject
    assert 'dev = [' in pyproject
    assert 'test = [' in pyproject
    assert 'docs = [' in pyproject
