"""GPU-accelerated H-matrix front-end (ACA fill on cupy).

This is a *prototype* — Lane E of the M4 GPU Phase 1 plan — that wires
the existing :class:`mnpbem.greenfun.hmatrix.HMatrix` (cluster tree,
admissibility, dense + low-rank block bookkeeping) together with the
cupy ACA primitives in :mod:`mnpbem.greenfun.aca_gpu`.  Only the per-block
ACA loop is offloaded to the GPU; the global tree, the dense diagonal
blocks, the truncate/inverse/LU paths and the public ``mtimes_vec`` /
``full`` API stay on the CPU and remain bit-compatible with the existing
H-matrix code.

Usage
-----
>>> from mnpbem.greenfun.h_matrix_gpu import HMatrixGPU
>>> hmat = HMatrixGPU.from_func(tree, kernel_fun, htol=1e-6, kmax=100)
>>> y = hmat @ x          # CPU mtimes (re-uses HMatrix path)
>>> stats = hmat.stat()   # compression report

Design notes
------------
* The GPU H-matrix is *not* a different class hierarchy.  We subclass
  :class:`HMatrix` and override only ``aca`` so the rest of the BEM
  stack does not need to know whether the fill happened on CPU or GPU.
* If cupy is unavailable (no CUDA device, missing wheel, ...) we fall
  back to the CPU ``HMatrix.aca`` path with a single warning.  This
  mirrors the convention used by other GPU-accelerated paths in the
  repo (see ``mnpbem/bem/_gpu.py``).
* For dimer (6336 faces) the per-block transfer overhead dominates; the
  prototype is expected to be *slower* than the CPU/numba path for that
  mesh.  The win materialises around N >= 20-30k faces; see
  ``docs/H_MATRIX_GPU.md`` for the projected break-even curve.
"""

from __future__ import annotations

import os
import warnings
from typing import Callable, Optional

import numpy as np

from .hmatrix import HMatrix
from .clustertree import ClusterTree
from . import aca_gpu


def _gpu_available() -> bool:
    if os.environ.get('MNPBEM_DISABLE_GPU', '').strip() in ('1', 'true', 'TRUE'):
        return False
    return aca_gpu.is_available()


class HMatrixGPU(HMatrix):
    """H-matrix whose admissible blocks are filled by GPU ACA.

    Parameters
    ----------
    tree : ClusterTree
    htol : float
        ACA Frobenius tolerance.
    kmax : int
        Max rank per low-rank block.
    fadmiss : callable, optional
        Admissibility predicate.  See :func:`HMatrix`.
    force_cpu : bool, optional
        Skip GPU and use the CPU ACA path even when cupy is available.
        Mostly useful for benchmarking and tests.
    """

    def __init__(self,
                 tree: Optional[ClusterTree] = None,
                 htol: float = 1e-6,
                 kmax: int = 100,
                 fadmiss: Optional[Callable] = None,
                 force_cpu: bool = False):
        super().__init__(tree=tree, htol=htol, kmax=kmax, fadmiss=fadmiss)
        self._force_cpu = bool(force_cpu)
        self._used_gpu = False  # set during aca()

    # ------------------------------------------------------------------
    # GPU-aware ACA.
    # ------------------------------------------------------------------
    def aca(self, fun: Callable) -> 'HMatrixGPU':
        """Fill dense + low-rank blocks; low-rank fill via GPU when possible."""
        if self._force_cpu or not _gpu_available():
            if not self._force_cpu:
                warnings.warn(
                    '[warn] HMatrixGPU: cupy/CUDA unavailable, falling '
                    'back to CPU ACA',
                    RuntimeWarning,
                    stacklevel=2,
                )
            self._used_gpu = False
            return super().aca(fun)

        self._used_gpu = True

        tree = self.tree
        ind_c2p = tree.ind[:, 0]

        def fun_c(row_c: np.ndarray, col_c: np.ndarray) -> np.ndarray:
            return fun(ind_c2p[row_c], ind_c2p[col_c])

        # Dense blocks — CPU path is already fast (small blocks).
        for i in range(len(self.row1)):
            indr = tree.cind[self.row1[i]]
            indc = tree.cind[self.col1[i]]
            rows = np.arange(indr[0], indr[1] + 1, dtype=np.int64)
            cols = np.arange(indc[0], indc[1] + 1, dtype=np.int64)
            row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')
            self.val[i] = fun_c(row_grid.ravel(),
                                col_grid.ravel()).reshape(row_grid.shape)

        # Low-rank blocks — GPU ACA.
        for i in range(len(self.row2)):
            indr = tree.cind[self.row2[i]]
            indc = tree.cind[self.col2[i]]
            rows = np.arange(indr[0], indr[1] + 1, dtype=np.int64)
            cols = np.arange(indc[0], indc[1] + 1, dtype=np.int64)

            U, V = aca_gpu.aca_block_gpu(
                fun_c, rows, cols, htol=self.htol, kmax=self.kmax,
                return_gpu=False,
            )
            self.lhs[i] = U
            self.rhs[i] = V

        return self

    # ------------------------------------------------------------------
    # Convenience constructor mirroring HMatrix.from_func.
    # ------------------------------------------------------------------
    @staticmethod
    def from_func(tree: ClusterTree,
                  fun: Callable,
                  htol: float = 1e-6,
                  kmax: int = 100,
                  fadmiss: Optional[Callable] = None,
                  force_cpu: bool = False) -> 'HMatrixGPU':
        h = HMatrixGPU(tree=tree, htol=htol, kmax=kmax,
                       fadmiss=fadmiss, force_cpu=force_cpu)
        h.aca(fun)
        return h

    # ------------------------------------------------------------------
    # Reporting.
    # ------------------------------------------------------------------
    @property
    def used_gpu(self) -> bool:
        """True iff the most recent aca() ran on the GPU."""
        return self._used_gpu


# ---------------------------------------------------------------------------
# Self-test (synthetic 1/r kernel).  Invoke via:
#   python -m mnpbem.greenfun.h_matrix_gpu
# ---------------------------------------------------------------------------
def _selftest(n: int = 1024, htol: float = 1e-6, seed: int = 0) -> dict:
    """Run a small round-trip accuracy + memory test.

    Builds a random point cloud, defines the 1/r kernel between
    well-separated halves, fills the H-matrix on the GPU, and compares
    the reconstructed dense matrix against the analytic dense matrix.
    Returns a dict with the metrics.
    """
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, 3))
    # Push the second half far away to ensure the off-diagonal block is
    # admissible (low rank).
    pos[n // 2:, 0] += 50.0

    eps = 1e-3

    def kernel(row_idx, col_idx):
        # 1/(|r_i - r_j| + eps), avoiding the singularity for diagonal.
        diff = pos[row_idx] - pos[col_idx]
        d = np.linalg.norm(diff, axis=-1)
        return 1.0 / (d + eps)

    tree = ClusterTree(pos, cleaf=64)

    # Reference dense matrix in particle ordering.
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    A_ref = kernel(ii.ravel(), jj.ravel()).reshape(n, n)

    metrics: dict = {'n': n, 'htol': htol, 'gpu_available': _gpu_available()}

    # GPU fill.
    if _gpu_available():
        hmat_gpu = HMatrixGPU.from_func(tree, kernel, htol=htol, kmax=80)
        A_gpu = hmat_gpu.full()
        err_gpu = float(np.linalg.norm(A_ref - A_gpu) / np.linalg.norm(A_ref))
        stats_gpu = hmat_gpu.stat()
        metrics['gpu_rel_fro_error'] = err_gpu
        metrics['gpu_compression_ratio'] = stats_gpu['compression_ratio']
        metrics['gpu_memory_mb'] = stats_gpu['memory_mb']
        metrics['gpu_mean_rank'] = stats_gpu['mean_rank']
        metrics['gpu_max_rank'] = stats_gpu['max_rank']
        metrics['used_gpu'] = hmat_gpu.used_gpu

    # CPU baseline for comparison.
    hmat_cpu = HMatrix.from_func(tree, kernel, htol=htol, kmax=80)
    A_cpu = hmat_cpu.full()
    err_cpu = float(np.linalg.norm(A_ref - A_cpu) / np.linalg.norm(A_ref))
    stats_cpu = hmat_cpu.stat()
    metrics['cpu_rel_fro_error'] = err_cpu
    metrics['cpu_compression_ratio'] = stats_cpu['compression_ratio']
    metrics['cpu_memory_mb'] = stats_cpu['memory_mb']

    metrics['dense_memory_mb'] = (n * n * 8) / (1024.0 * 1024.0)

    return metrics


if __name__ == '__main__':  # pragma: no cover
    import json
    out = _selftest(n=1024, htol=1e-6)
    print(json.dumps(out, indent=2, default=str))
