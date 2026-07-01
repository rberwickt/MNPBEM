"""Regression test for v1.6.5: BEMRetLayer cupy/numpy mix safety."""

import numpy as np
import pytest


def test_backend_align_numpy_only():
    from GUI.mnpbem.bem.bem_ret_layer import _backend_align
    A = np.zeros((4, 4), dtype = complex)
    B = np.ones((4, 4), dtype = complex)
    A2, B2 = _backend_align(A, B)
    assert A2 is A
    assert B2 is B


def test_backend_align_scalars():
    from GUI.mnpbem.bem.bem_ret_layer import _backend_align
    A2, B2 = _backend_align(0, 0)
    assert A2 == 0 and B2 == 0
    A2, B2 = _backend_align(1.5, 0)
    assert A2 == 1.5 and B2 == 0


def test_backend_align_cupy_numpy_mix():
    cp = pytest.importorskip('cupy')
    from GUI.mnpbem.bem.bem_ret_layer import _backend_align
    A_gpu = cp.zeros((4, 4), dtype = complex)
    B_cpu = np.ones((4, 4), dtype = complex)
    A2, B2 = _backend_align(A_gpu, B_cpu)
    assert isinstance(A2, cp.ndarray)
    assert isinstance(B2, cp.ndarray)
    # Symmetric case
    A2, B2 = _backend_align(B_cpu, A_gpu)
    assert isinstance(A2, cp.ndarray)
    assert isinstance(B2, cp.ndarray)


def test_sub_mat_cupy_numpy_mix():
    cp = pytest.importorskip('cupy')
    from GUI.mnpbem.bem.bem_ret_layer import BEMRetLayer
    obj = BEMRetLayer.__new__(BEMRetLayer)
    A_gpu = cp.ones((3, 3), dtype = complex)
    B_cpu = np.ones((3, 3), dtype = complex) * 0.25
    out = obj._sub_mat(A_gpu, B_cpu)
    # _to_host_safe brings result to host so downstream lu_factor_dispatch works.
    assert isinstance(out, np.ndarray)
    expected = np.ones((3, 3), dtype = complex) * 0.75
    assert np.allclose(out, expected)


def test_mul_eps_cupy_numpy_mix():
    cp = pytest.importorskip('cupy')
    from GUI.mnpbem.bem.bem_ret_layer import BEMRetLayer
    obj = BEMRetLayer.__new__(BEMRetLayer)
    eps_cpu = np.diag(np.full(3, 2.0 + 0j))
    M_gpu = cp.ones((3, 3), dtype = complex) * 0.5
    out = obj._mul_eps(eps_cpu, M_gpu)
    # _to_host_safe brings result to host so downstream lu_factor_dispatch works.
    assert isinstance(out, np.ndarray)
    expected = np.ones((3, 3), dtype = complex)
    assert np.allclose(out, expected)
