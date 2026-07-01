"""GPU-accelerated Adaptive Cross Approximation (ACA) primitives.

This module provides a cupy-backed implementation of the partially-pivoted
ACA algorithm used to compress admissible H-matrix blocks.  The intent is
to keep the public function signature aligned with
:func:`mnpbem.greenfun.hmatrix.HMatrix._aca_block` so that the existing
H-matrix machinery (cluster tree, admissibility, block bookkeeping, full
reconstruction, mtimes, LU, ...) can stay on the CPU while only the
*per-block ACA fill* — the dominant cost for large meshes — is offloaded
to the GPU.

Lane E (M4 GPU Phase 1) prototype.
==================================

Design constraints
------------------
* Optional dependency.  Importing this module must not require cupy at
  module import time; cupy is only loaded when GPU functions are
  actually invoked.  This keeps the regular CPU pipeline unaffected.
* Drop-in compatible.  ``aca_block_gpu(fun, rows, cols, htol, kmax)``
  returns ``(U, V)`` numpy arrays just like the CPU version, so the
  caller does not have to know about cupy.
* Numerically equivalent.  We follow the same partial-pivoting strategy
  (pivot on max |residual| in the unused row/column) and the same
  Frobenius-norm convergence test.  Differences w.r.t. the CPU path are
  bounded by FP rounding (cuBLAS vs MKL).

Why a hand-rolled cupy ACA rather than an external library?
-----------------------------------------------------------
The M4-H1 reconnaissance (see ``docs/PERFORMANCE_STRATEGY.md`` and the
``m4-h1`` branch) considered three options:

1. ``scipy.sparse.linalg.LinearOperator`` — only an interface, no
   compression algorithm; CPU only.
2. ``h2tools`` / ``pyhml2d`` / ``hlibpro`` python bindings — either
   unmaintained, MPI-bound, or LGPL/commercial; none of them ship a
   cupy code path.  Forcing the kernel evaluation through their Python
   callbacks would also kill GPU throughput because each block calls
   the user kernel ``O(rank)`` times.
3. Roll our own on top of the existing ``HMatrix`` — small (~150 LoC),
   matches the partially-pivoted CPU ACA byte-for-byte at zero
   tolerance, and makes the kernel function the natural batching unit.

We chose option (3).  The kernel function ``fun(rows, cols)`` is the
*only* required GPU primitive; if the caller can supply a cupy-aware
``fun`` (e.g. one that reads from a cupy ``ndarray`` of precomputed
Green-function values, or that recomputes the 1/r kernel on the GPU)
the entire ACA loop runs without round-tripping to host memory.  The
default behaviour transparently up-/down-loads numpy buffers so existing
CPU kernels keep working.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional, Tuple


# ---------------------------------------------------------------------------
# cupy bootstrap — defer the import so the module is safe to import on
# CPU-only machines.  ``_cupy()`` raises a clear error if cupy is missing.
# ---------------------------------------------------------------------------
_cp = None


def _cupy():
    """Return the cupy module, importing it lazily."""
    global _cp
    if _cp is None:
        try:
            import cupy as cp  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                '[error] aca_gpu requires cupy; install with '
                '`pip install cupy-cuda12x` (or matching CUDA toolkit)'
            ) from exc
        _cp = cp
    return _cp


def is_available() -> bool:
    """Return True iff cupy and a CUDA device are usable."""
    try:
        cp = _cupy()
        return bool(cp.cuda.is_available())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Kernel adapter.  When the caller passes a numpy-only ``fun(row, col)`` we
# wrap it so it returns cupy arrays and accepts cupy index arrays.
# ---------------------------------------------------------------------------
def _wrap_kernel_to_gpu(fun: Callable) -> Callable:
    cp = _cupy()

    def fun_gpu(row, col):
        if hasattr(row, 'device'):  # cupy array
            row_h = cp.asnumpy(row)
            col_h = cp.asnumpy(col)
        else:
            row_h = np.ascontiguousarray(row)
            col_h = np.ascontiguousarray(col)
        vals = fun(row_h, col_h)
        return cp.asarray(vals)

    return fun_gpu


# ---------------------------------------------------------------------------
# GPU ACA per-block.  Mirrors HMatrix._aca_block but uses cupy ops.
# ---------------------------------------------------------------------------
def aca_block_gpu(fun: Callable,
                  rows: np.ndarray,
                  cols: np.ndarray,
                  htol: float = 1e-6,
                  kmax: int = 100,
                  return_gpu: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Partially-pivoted ACA for a single admissible block, on GPU.

    Parameters
    ----------
    fun : callable
        ``fun(row_idx, col_idx) -> values``.  ``row_idx`` / ``col_idx``
        are 1-D arrays of (cluster-local) global indices; the returned
        values may be numpy or cupy.
    rows, cols : np.ndarray
        Row and column indices of the block (1-D, in *cluster* ordering
        as used by the parent HMatrix).
    htol : float
        Relative Frobenius-norm tolerance for early termination.
    kmax : int
        Hard upper bound on the rank.
    return_gpu : bool
        If True, leave U/V on the GPU as cupy arrays.  Default False.

    Returns
    -------
    U, V : ndarray
        ``A_block ~= U @ V.T`` with shapes ``(m, rank)`` and ``(n, rank)``.
    """
    cp = _cupy()

    rows = np.ascontiguousarray(rows, dtype=np.int64)
    cols = np.ascontiguousarray(cols, dtype=np.int64)
    m, n = rows.size, cols.size
    max_rank = int(min(m, n, kmax))

    fun_g = _wrap_kernel_to_gpu(fun)

    # Probe dtype
    probe = fun_g(rows[:1], cols[:1])
    out_dtype = cp.complex128 if cp.iscomplexobj(probe) else cp.float64

    # Move index arrays once to device
    rows_g = cp.asarray(rows)
    cols_g = cp.asarray(cols)

    U = cp.empty((m, max_rank), dtype=out_dtype)
    V = cp.empty((n, max_rank), dtype=out_dtype)

    used_row = cp.zeros(m, dtype=cp.bool_)
    used_col = cp.zeros(n, dtype=cp.bool_)

    pivot_row = 0
    rank = 0
    frob_sq = 0.0

    for _ in range(max_rank):
        # Residual row at pivot_row.
        row_global = int(rows[pivot_row])
        r_idx = cp.full(n, row_global, dtype=cp.int64)
        row_vals = cp.asarray(fun_g(r_idx, cols_g), dtype=out_dtype)

        if rank > 0:
            # row_vals -= V[:, :rank] @ U[pivot_row, :rank]
            row_vals = row_vals - V[:, :rank] @ U[pivot_row, :rank]

        abs_row = cp.abs(row_vals)
        abs_row = cp.where(used_col, cp.asarray(0.0, dtype=abs_row.dtype), abs_row)
        pivot_col = int(cp.argmax(abs_row))
        pivot_val = row_vals[pivot_col]

        if float(cp.abs(pivot_val)) < 1e-15:
            break

        # Residual column at pivot_col.
        col_global = int(cols[pivot_col])
        c_idx = cp.full(m, col_global, dtype=cp.int64)
        col_vals = cp.asarray(fun_g(rows_g, c_idx), dtype=out_dtype)
        if rank > 0:
            col_vals = col_vals - U[:, :rank] @ V[pivot_col, :rank]

        u_new = col_vals / pivot_val
        v_new = row_vals  # use directly (no further mutation)

        U[:, rank] = u_new
        V[:, rank] = v_new
        rank += 1

        used_row[pivot_row] = True
        used_col[pivot_col] = True

        # Frobenius-norm convergence (Hermitian for complex).
        if out_dtype == cp.complex128:
            u_norm_sq = float(cp.vdot(u_new, u_new).real)
            v_norm_sq = float(cp.vdot(v_new, v_new).real)
        else:
            u_norm_sq = float(cp.dot(u_new, u_new))
            v_norm_sq = float(cp.dot(v_new, v_new))
        new_term_sq = u_norm_sq * v_norm_sq

        if rank > 1:
            if out_dtype == cp.complex128:
                dot_u = cp.conj(U[:, :rank - 1].T) @ u_new
                dot_v = cp.conj(V[:, :rank - 1].T) @ v_new
                cross = 2.0 * float(cp.real(cp.dot(dot_u, dot_v)))
            else:
                cross = 2.0 * float(cp.dot(U[:, :rank - 1].T @ u_new,
                                           V[:, :rank - 1].T @ v_new))
        else:
            cross = 0.0
        frob_sq += new_term_sq + cross

        if frob_sq > 0 and np.sqrt(new_term_sq) < htol * np.sqrt(abs(frob_sq)):
            break

        # Next pivot row: max |u_new| over unused rows.
        abs_u = cp.abs(u_new)
        abs_u = cp.where(used_row, cp.asarray(0.0, dtype=abs_u.dtype), abs_u)
        pivot_row = int(cp.argmax(abs_u))

    if rank == 0:
        zero_u = cp.zeros((m, 1), dtype=out_dtype)
        zero_v = cp.zeros((n, 1), dtype=out_dtype)
        if return_gpu:
            return zero_u, zero_v
        return cp.asnumpy(zero_u), cp.asnumpy(zero_v)

    U_out = U[:, :rank].copy()
    V_out = V[:, :rank].copy()
    if return_gpu:
        return U_out, V_out
    return cp.asnumpy(U_out), cp.asnumpy(V_out)
