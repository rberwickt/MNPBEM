import os
import sys

from typing import Any, Tuple

import numpy as np
import pytest


# Try import cupy; tests that need GPU are conditionally skipped.
try:
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    cp = None  # type: ignore
    _HAS_CUPY = False


cupy_required = pytest.mark.skipif(not _HAS_CUPY, reason = 'cupy not installed')


# ---------------------------------------------------------------------------
# Helpers — tiny synthetic 3D problem with a known dense reference matrix.
# ---------------------------------------------------------------------------

def _two_cluster_points(n1: int = 16, n2: int = 16,
        offset: float = 60.0) -> np.ndarray:

    rng = np.random.RandomState(0)
    p1 = rng.standard_normal((n1, 3)) * 5.0
    p2 = rng.standard_normal((n2, 3)) * 5.0
    p2[:, 0] += offset
    pos = np.empty((n1 + n2, 3), dtype = np.float64)
    pos[:n1] = p1
    pos[n1:] = p2
    return pos


def _build_cpu_hmatrix(pos: np.ndarray,
        complex_kernel: bool = True) -> Tuple[Any, np.ndarray]:

    from GUI.mnpbem.greenfun.clustertree import ClusterTree
    from GUI.mnpbem.greenfun.hmatrix import HMatrix

    tree = ClusterTree(pos, cleaf = 8)

    if complex_kernel:
        def fun(row: np.ndarray, col: np.ndarray) -> np.ndarray:
            d = np.linalg.norm(pos[row] - pos[col], axis = 1)
            d = np.maximum(d, 1e-10)
            k = 0.05  # small wavenumber, mildly oscillatory
            return np.exp(1j * k * d) / (4.0 * np.pi * d)
    else:
        def fun(row: np.ndarray, col: np.ndarray) -> np.ndarray:
            d = np.linalg.norm(pos[row] - pos[col], axis = 1)
            d = np.maximum(d, 1e-10)
            return 1.0 / (4.0 * np.pi * d)

    h_cpu = HMatrix(tree = tree, htol = 1e-10, kmax = 30)
    h_cpu.aca(fun)

    # Reference dense matrix in particle ordering.
    n = pos.shape[0]
    ref = np.empty((n, n), dtype = np.complex128 if complex_kernel else np.float64)
    rows = np.arange(n, dtype = np.int64)
    for i in range(n):
        cols = np.full(n, i, dtype = np.int64)
        ref[:, i] = fun(rows, cols)
    return h_cpu, ref


# ---------------------------------------------------------------------------
# Bug 5: HMatrix.full() must return numpy on CPU and cupy on GPU.
# ---------------------------------------------------------------------------

def test_full_cpu_matches_reference_complex():
    # Sanity: numpy backend reproduces the dense reference within ACA tol.
    pos = _two_cluster_points()
    h_cpu, ref = _build_cpu_hmatrix(pos, complex_kernel = True)

    full_np = h_cpu.full()
    assert isinstance(full_np, np.ndarray)
    err = np.linalg.norm(full_np - ref) / np.linalg.norm(ref)
    assert err < 1e-6, '[error] HMatrix.full() drifts: rel={:.2e}'.format(err)


def test_full_cpu_matches_reference_real():
    pos = _two_cluster_points()
    h_cpu, ref = _build_cpu_hmatrix(pos, complex_kernel = False)

    full_np = h_cpu.full()
    assert isinstance(full_np, np.ndarray)
    err = np.linalg.norm(full_np - ref) / np.linalg.norm(ref)
    assert err < 1e-6, '[error] HMatrix.full() (real) drifts: rel={:.2e}'.format(err)


@cupy_required
def test_full_gpu_blocks_autoreturn_host():
    # Bug 5 root cause: when val/lhs/rhs hold cupy arrays the v1.5.1
    # implementation allocated a numpy host buffer and tried to slice-assign
    # cupy ndarrays into it -> TypeError.  After the v1.5.2 fix, full()
    # auto-detects GPU blocks, fills the result on GPU (no TypeError) and
    # — to keep the working set within the 49 GB single-GPU cap on
    # Tier-3-class problems — pulls the matrix to host before the
    # final cluster->particle permutation (which would otherwise allocate
    # a second NxN buffer on device).  Caller can override with
    # ``xp=cupy`` to keep the result on device.
    pos = _two_cluster_points()
    h_cpu, ref = _build_cpu_hmatrix(pos, complex_kernel = True)

    # Promote dense + low-rank blocks to cupy.
    h_gpu = h_cpu._copy()
    h_gpu.val = [cp.asarray(v) if v is not None else None for v in h_gpu.val]
    h_gpu.lhs = [cp.asarray(l) if l is not None else None for l in h_gpu.lhs]
    h_gpu.rhs = [cp.asarray(r) if r is not None else None for r in h_gpu.rhs]

    full_auto = h_gpu.full()  # auto-detect: cupy fill, host return
    assert isinstance(full_auto, np.ndarray), \
        '[error] auto full() on GPU blocks should return host (memory-safe)'
    err = np.linalg.norm(full_auto - ref) / np.linalg.norm(ref)
    assert err < 1e-6, '[error] auto GPU full() drifts: rel={:.2e}'.format(err)

    # Forced xp=cupy keeps result on device.
    full_gpu = h_gpu.full(xp = cp)
    assert isinstance(full_gpu, cp.ndarray)
    err = np.linalg.norm(cp.asnumpy(full_gpu) - ref) / np.linalg.norm(ref)
    assert err < 1e-6, '[error] forced GPU full() drifts: rel={:.2e}'.format(err)


@cupy_required
def test_full_gpu_xp_force_numpy_returns_host():
    # Caller can override the backend via the new ``xp`` argument; cupy
    # blocks are pulled to host with .get() per block.
    pos = _two_cluster_points()
    h_cpu, ref = _build_cpu_hmatrix(pos, complex_kernel = True)

    h_gpu = h_cpu._copy()
    h_gpu.val = [cp.asarray(v) if v is not None else None for v in h_gpu.val]
    h_gpu.lhs = [cp.asarray(l) if l is not None else None for l in h_gpu.lhs]
    h_gpu.rhs = [cp.asarray(r) if r is not None else None for r in h_gpu.rhs]

    full_np = h_gpu.full(xp = np)
    assert isinstance(full_np, np.ndarray)
    err = np.linalg.norm(full_np - ref) / np.linalg.norm(ref)
    assert err < 1e-6, '[error] full(xp=np) on GPU blocks drifts: rel={:.2e}'.format(err)


@cupy_required
def test_full_cpu_xp_force_cupy_promotes_numpy():
    # Reverse direction: numpy blocks + xp=cupy should promote each block
    # via cupy.asarray and produce a cupy result equal to the CPU one.
    pos = _two_cluster_points()
    h_cpu, ref = _build_cpu_hmatrix(pos, complex_kernel = True)

    full_gpu = h_cpu.full(xp = cp)
    assert isinstance(full_gpu, cp.ndarray)
    err = np.linalg.norm(cp.asnumpy(full_gpu) - ref) / np.linalg.norm(ref)
    assert err < 1e-6, '[error] full(xp=cp) on numpy blocks drifts: rel={:.2e}'.format(err)


@cupy_required
def test_full_mixed_blocks_cupy_dominates():
    # Even a single cupy block triggers the auto-detected GPU backend; the
    # remaining numpy blocks are promoted on the fly.  This exercises the
    # code path BEMRetIter._init_precond → _compress → hmat.full()
    # encounters when ACA runs on GPU but a few dense leaves stayed on
    # host (e.g. tiny diagonal blocks below the GPU threshold).
    pos = _two_cluster_points()
    h_cpu, ref = _build_cpu_hmatrix(pos, complex_kernel = True)

    h_mix = h_cpu._copy()
    if len(h_mix.val) > 0 and h_mix.val[0] is not None:
        h_mix.val[0] = cp.asarray(h_mix.val[0])

    full_mix = h_mix.full()  # auto-detected GPU; returns host (memory-safe)
    assert isinstance(full_mix, np.ndarray)
    err = np.linalg.norm(full_mix - ref) / np.linalg.norm(ref)
    assert err < 1e-6, '[error] mixed full() drifts: rel={:.2e}'.format(err)


# ---------------------------------------------------------------------------
# Integration: BEMRetIter._init_precond on GPU must complete (Bug 5 → fix).
# Runs only when cupy is present AND MNPBEM_GPU=1.
# ---------------------------------------------------------------------------

@cupy_required
def test_full_with_aca_built_cupy_dense_blocks():
    # Realistic scenario: ACA dense blocks are cupy (CompGreenRet returns
    # cupy under MNPBEM_GPU_NATIVE=1), low-rank lhs/rhs are numpy (the CPU
    # ACA loop in HMatrix._aca_block coerces them back to host).  full()
    # must allocate the result on GPU without TypeError and reproduce the
    # CPU reference within ACA tol.
    pos = _two_cluster_points()
    h_cpu, ref = _build_cpu_hmatrix(pos, complex_kernel = True)

    h_real = h_cpu._copy()
    # Mirror production GPU layout: dense blocks on GPU, low-rank on host.
    h_real.val = [cp.asarray(v) if v is not None else None for v in h_real.val]
    h_real.lhs = [l for l in h_real.lhs]  # numpy untouched
    h_real.rhs = [r for r in h_real.rhs]

    full_real = h_real.full()  # auto: GPU fill, host return (memory-safe)
    assert isinstance(full_real, np.ndarray)
    err = np.linalg.norm(full_real - ref) / np.linalg.norm(ref)
    assert err < 1e-6, '[error] mixed dense-gpu / lr-cpu drifts: rel={:.2e}'.format(err)


@cupy_required
@pytest.mark.skipif(os.environ.get('MNPBEM_GPU', '0') != '1',
        reason = 'MNPBEM_GPU=1 not set')
def test_bemretiter_init_precond_gpu_completes():
    # Drives the actual production trigger of Bug 5: with
    # MNPBEM_GPU_NATIVE=1 the underlying CompGreenRet returns cupy dense
    # matrices once the mesh exceeds the GPU threshold; ACACompGreenRet
    # then stores cupy slices in HMatrix.val[i].  v1.5.1's full() raised
    # TypeError on the implicit numpy slice assignment, blocking the
    # BEMRetIter dense-LU preconditioner build.  This test must clear
    # _init_matrices end-to-end on a mesh large enough to activate the
    # native GPU path (~2884 face trisphere @ default GPU thresholds).
    os.environ.setdefault('MNPBEM_GPU_NATIVE', '1')

    from GUI.mnpbem.geometry import ComParticle, trisphere
    from GUI.mnpbem.materials import EpsConst, EpsDrude
    from GUI.mnpbem.bem import BEMRetIter

    eps_b = EpsConst(1.0)
    eps_m = EpsDrude.gold()
    p_raw = trisphere(2000, 10.0)
    p = ComParticle([eps_b, eps_m], [p_raw], [[2, 1]])

    bem = BEMRetIter(p, hmatrix = True, htol = 1e-3, precond = 'lu_dense')
    bem._init_matrices(550.0)
    assert bem._G1 is not None and bem._G2 is not None, \
        '[error] BEMRetIter._init_matrices did not populate G blocks'
