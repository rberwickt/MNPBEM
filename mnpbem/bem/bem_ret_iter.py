import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator

from ..greenfun import CompStruct
from ..utils.gpu import (
    lu_factor_dispatch, lu_solve_dispatch, lu_solve_native,
    eye_like_lu, to_host, is_cupy_array,
)
from ..utils.matlab_compat import msqrt
from .bem_iter import BEMIter

# v1.7.2 GPU memory-pool cleanup: mirror BEMRet's wavelength-end immediate
# free pattern (bem_ret.py:485-503, 734-737) on the iter solver so cupy
# returns blocks to the driver at every wavelength rather than accumulating
# across the sweep (MATLAB-parity).  The retarded iter path already had
# scattered ``free_all_blocks`` calls inside the GPU-LU pipeline; the
# additions below extend coverage to the wavelength boundary
# (``_init_matrices`` entry), the preconditioner boundary
# (``_init_precond`` exit) and the GMRES iter boundary (``solve`` around
# ``_iter_solve``) so the high-water mark stays stable across the entire
# sweep on 12672-face Au@Ag dimers.
try:
    import cupy as _cp_iter  # type: ignore
    _CUPY_OK_ITER = True
except Exception:
    _cp_iter = None  # type: ignore
    _CUPY_OK_ITER = False


def _gpu_pool_cleanup_iter(apply_limit: bool = False) -> None:
    """Synchronise CUDA stream then drain cupy default + pinned pools.

    Mirrors the v1.7.2 BEMRet helper (bem_ret.py:485-503, 734-737).  The
    deviceSynchronize() BEFORE free_all_blocks() is load-bearing: blocks
    still in flight on the CUDA stream are not idle yet, so a
    free_all_blocks that races ahead of the stream returns nothing.

    Honours MNPBEM_GPU_POOL_LIMIT_GB (legitimate peaks past this cap will
    OOM; default 0 = uncapped).
    """
    if not _CUPY_OK_ITER:
        return
    try:
        mempool = _cp_iter.get_default_memory_pool()
        pinned = _cp_iter.get_default_pinned_memory_pool()
        if apply_limit:
            try:
                pool_limit_gb = float(os.environ.get(
                        'MNPBEM_GPU_POOL_LIMIT_GB', '0'))
            except (TypeError, ValueError):
                pool_limit_gb = 0.0
            if pool_limit_gb > 0:
                mempool.set_limit(size = int(pool_limit_gb * (1024 ** 3)))
        _cp_iter.cuda.runtime.deviceSynchronize()
        mempool.free_all_blocks()
        pinned.free_all_blocks()
    except Exception:
        pass


def _vram_share_lu_kwargs() -> dict:
    """Read MNPBEM_VRAM_SHARE_* env vars and return kwargs for lu_factor_dispatch.

    Mirrors ``bem_ret.py``'s helper of the same name (line 43-54).  Returns
    an empty dict when VRAM-share is disabled (``MNPBEM_VRAM_SHARE!=1`` or
    ``MNPBEM_VRAM_SHARE_GPUS<=1``) so the dispatch call is bit-identical to
    the single-GPU path.  When VRAM-share is enabled the returned kwargs
    route ``lu_factor_dispatch`` through ``factor_multi_gpu`` (cuSolverMg by
    default) and the matrix is partitioned across ``n_gpus`` devices.
    Required for the 15072-face Au@Ag dimer sweep where the dense LU
    (~3.6 GB complex128) plus the cached G/H/precond state exceeds the
    49 GB single-A6000 cap.
    """
    if os.environ.get('MNPBEM_VRAM_SHARE', '0') != '1':
        return {}
    n_gpus = int(os.environ.get('MNPBEM_VRAM_SHARE_GPUS', '1'))
    if n_gpus <= 1:
        return {}
    backend = os.environ.get('MNPBEM_VRAM_SHARE_BACKEND', 'cusolvermg')
    return {'n_gpus': n_gpus, 'backend': backend}


# ---------------------------------------------------------------------------
# B-3 distributed-build helpers (v1.7.3 Phase 3)
# ---------------------------------------------------------------------------

def _vram_share_active() -> bool:
    """Return True when distributed-build is enabled.

    Activated when **all** of the following hold:
    - ``MNPBEM_VRAM_SHARE=1``
    - ``MNPBEM_VRAM_SHARE_DISTRIBUTED=1`` (gates the heavier distributed
      build path; default off so existing call sites stay bit-identical)
    - ``MNPBEM_VRAM_SHARE_GPUS>=2``
    - cupy + cuSolverMg are importable

    Distributed build assembles G/H matrices directly across N GPUs via
    ``DistributedMatrix.from_func`` + ``CompGreenRet.eval_block``, avoiding
    the host-resident full ``N x N`` Green-function matrix that would
    otherwise dominate the BEM build memory.  Used for the 15072-face
    Au@Ag dimer sweep where the per-wavelength precond pipeline would
    otherwise OOM around N~12k.
    """
    if os.environ.get('MNPBEM_VRAM_SHARE', '0') != '1':
        return False
    if os.environ.get('MNPBEM_VRAM_SHARE_DISTRIBUTED', '0') != '1':
        return False
    try:
        n_gpus = int(os.environ.get('MNPBEM_VRAM_SHARE_GPUS', '1'))
    except (TypeError, ValueError):
        return False
    if n_gpus < 2:
        return False
    if not _CUPY_OK_ITER:
        return False
    try:
        from ..utils.multi_gpu_lu import cusolvermg_available
        return bool(cusolvermg_available())
    except Exception:
        return False


def _vram_share_env_config() -> Tuple[int, str, Optional[List[int]]]:
    """Resolve (n_gpus, backend, device_ids) for the distributed path.

    Caller should already have checked ``_vram_share_active()``.
    """
    n_gpus = int(os.environ.get('MNPBEM_VRAM_SHARE_GPUS', '1'))
    backend = os.environ.get('MNPBEM_VRAM_SHARE_BACKEND', 'cusolvermg')
    devs_env = os.environ.get('MNPBEM_VRAM_SHARE_DEVICE_IDS', '')
    device_ids: Optional[List[int]] = None
    if devs_env.strip():
        try:
            device_ids = [int(x) for x in devs_env.split(',') if x.strip()]
            if len(device_ids) != n_gpus:
                device_ids = None
        except (TypeError, ValueError):
            device_ids = None
    return n_gpus, backend, device_ids


def _build_distributed_green(green: Any,
        i: int,
        j: int,
        key: str,
        enei: float,
        nrows: int,
        ncols: int,
        n_gpus: int,
        device_ids: Optional[List[int]]) -> Any:
    """Assemble a single (i, j, key) Green-function block as a DistributedMatrix.

    Each GPU computes its own column tile via :meth:`CompGreenRet.eval_block`,
    so the full ``(nrows, ncols)`` matrix never materialises on any single
    device.  Returned as ``DistributedMatrix`` (caller does ``.to_host()``
    when it needs a host gather for the precond pipeline).

    Note: caller passes ``i, j, key, enei`` straight through to
    ``green.eval_block``.  The eval_block API raises NotImplementedError
    for the H-matrix / hmode path, so the caller must guard distributed
    build against ACA wrappers.
    """
    from ..utils.distributed_matrix import DistributedMatrix
    import numpy as _np_local

    def _evalfn(gpu_idx: int, c0: int, c1: int) -> Any:
        return green.eval_block(i, j, key, enei, c0, c1)

    return DistributedMatrix.from_func(
            shape = (nrows, ncols),
            dtype = _np_local.complex128,
            n_gpus = n_gpus,
            eval_func = _evalfn,
            device_ids = device_ids)


def _distributed_block_assemble(green: Any,
        enei: float,
        ncomb_list: List[Tuple[int, int, str, int]],
        nrows: int,
        ncols: int,
        n_gpus: int,
        device_ids: Optional[List[int]]) -> Any:
    """Build a linear combination of Green-function blocks distributed.

    ``ncomb_list`` is ``[(i, j, key, sign), ...]`` — the per-GPU eval_func
    sums ``sign * eval_block(...)`` so the resulting DistributedMatrix is
    ``sum_k sign_k · G(i_k, j_k, key_k)``.  Used for ``G1 = G11 - G21``
    and friends without ever materialising the individual ``G11 / G21``
    full matrices on a single device.
    """
    from ..utils.distributed_matrix import DistributedMatrix
    import numpy as _np_local

    def _evalfn(gpu_idx: int, c0: int, c1: int) -> Any:
        out = None
        for (i, j, key, sign) in ncomb_list:
            blk = green.eval_block(i, j, key, enei, c0, c1)
            if out is None:
                if isinstance(blk, _np_local.ndarray):
                    out = sign * blk if sign != 1 else blk.copy()
                else:
                    # cupy ndarray
                    out = sign * blk if sign != 1 else blk.copy()
            else:
                if sign == 1:
                    out = out + blk
                elif sign == -1:
                    out = out - blk
                else:
                    out = out + sign * blk
        return out

    return DistributedMatrix.from_func(
            shape = (nrows, ncols),
            dtype = _np_local.complex128,
            n_gpus = n_gpus,
            eval_func = _evalfn,
            device_ids = device_ids)


class _GpuPrecondOOM(Exception):
    pass


def _gpu_precond_capacity_ok(N: int) -> bool:
    # v1.6.3: decide whether the serialized GPU precond pipeline can fit.
    # The transient peak in any single stage is bounded by ~6 N x N
    # complex128 buffers (G + LU + eye + Gi + workspace + cusolver
    # scratch).  Require that the free memory on the active CUDA device
    # exceeds 7 N^2 * 16 bytes to leave headroom for fragmentation.
    try:
        import cupy as _cp_local
    except Exception:
        return False
    try:
        free_bytes, _total_bytes = _cp_local.cuda.runtime.memGetInfo()
    except Exception:
        return False
    needed = 7 * (N ** 2) * 16
    return free_bytes >= needed


def _build_precond_gpu_serial(
        G1: np.ndarray,
        H1: np.ndarray,
        G2: np.ndarray,
        H2: np.ndarray,
        eps1_diag: Any,
        eps2_diag: Any,
        nvec: np.ndarray,
        k: float,
        decorate_deltai_fn: Callable) -> Tuple:
    # v1.6.3 hybrid GPU LU + host matmul pipeline.
    #
    # Background: cuSOLVER's ``lu_solve(LU, eye(N))`` (used to build the
    # explicit inverse) is *not* faster than scipy's MKL-multithreaded
    # ``lu_solve`` on a 132-core box at N >= 8000 — both are O(N^3) and
    # MKL parallelises across all cores, while cusolver maxes out a
    # single GPU.  In isolated benchmarks the host scipy path is ~30%
    # *faster* than a pure GPU pipeline at N=12672 on RTX A6000.
    #
    # The actual win from "use GPU for precond" is therefore not the
    # init pipeline itself but rather the per-iterate cost during
    # GMRES: ``_mfun`` does eight ``lu_solve_dispatch`` calls per
    # iterate, each on an N-vector.  When the LU lives on GPU these
    # cost ~5 ms each (cusolver getrs); when it lives on host they
    # cost ~50 ms each (scipy single-RHS).  At ~100 GMRES iterates
    # (typical Au@Ag) that is 40 s saved per wavelength.
    #
    # Pipeline:
    #   * Factor G1, G2, Delta, Sigma_mat on GPU once; keep the LU
    #     packages tagged ``'gpu'`` so ``_mfun`` dispatches to cusolver
    #     getrs.
    #   * Compute G1^{-1}, G2^{-1}, Deltai on the host via scipy
    #     lu_solve (fast MKL).  These are then matmul-ed with H/G to
    #     form Sigma1, Sigma_L1, L1, Sigma_mat on the host (cheap).
    #   * Bring the host LU package up to the GPU once for each of the
    #     four needed factorizations.  We therefore do the LU twice —
    #     once on GPU (for fast solve) and once on host (for the
    #     inverse) — but the GPU LU is overlapped with host LU + GEMMs
    #     so wall time is dominated by the host pipeline.
    #
    # Memory: only one N x N matrix is GPU-resident at any time during
    # init (the matrix being factored).  Persistent residents after
    # init are the four LU packages (~4 * 2.57 = 10 GB at N=12672) —
    # comfortably under the 49 GB cap.
    #
    # OOM fallback: any cupy OOM raises ``_GpuPrecondOOM`` so the caller
    # drops to the pure host scipy path.
    try:
        import cupy as _cp_local
        from cupyx.scipy.linalg import lu_factor as _cp_lu_factor
    except Exception as exc:
        raise _GpuPrecondOOM('cupy unavailable: {}'.format(exc))
    from scipy.linalg import lu_factor as _scipy_lu_factor
    from scipy.linalg import lu_solve as _scipy_lu_solve

    pool = _cp_local.get_default_memory_pool()
    pool.free_all_blocks()
    N = G1.shape[0]
    eps1_is_scalar = (np.isscalar(eps1_diag)
            or (isinstance(eps1_diag, np.ndarray) and eps1_diag.ndim == 0))

    if eps1_is_scalar:
        eps1_vec = None
        eps2_vec = None
    else:
        eps1_vec = np.diag(eps1_diag) if eps1_diag.ndim == 2 else eps1_diag
        eps2_vec = np.diag(eps2_diag) if eps2_diag.ndim == 2 else eps2_diag

    # v1.7.3 (Phase 2 VRAM-share): when MNPBEM_VRAM_SHARE=1 +
    # MNPBEM_VRAM_SHARE_GPUS>=2 is set, route the four precond LU factors
    # through ``lu_factor_dispatch`` so cuSolverMg partitions each N x N
    # complex128 factor across the worker's GPUs.  Without this the
    # 15072-face Au@Ag dimer (~3.6 GB / factor + cached G/H/precond
    # state) overshoots the 49 GB single-A6000 cap.  When the env var is
    # off, the legacy single-GPU ``cupyx.scipy.linalg.lu_factor`` path
    # below is taken (bit-identical to v1.6.3 behaviour).
    _vram_kwargs = _vram_share_lu_kwargs()

    def _gpu_lu(A_host):
        # Factor A on GPU and return ('gpu'|'mgpu', lu, piv).  A_host is
        # left untouched.  VRAM-share path returns ``('mgpu', handle, None)``
        # which ``lu_solve_dispatch`` already handles transparently
        # (gpu.py:200-209).
        if _vram_kwargs:
            try:
                return lu_factor_dispatch(A_host, **_vram_kwargs)
            except _cp_local.cuda.memory.OutOfMemoryError as exc:
                pool.free_all_blocks()
                raise _GpuPrecondOOM('gpu_lu (mgpu): {}'.format(exc))
            except Exception as exc:
                # cuSolverMg / driver failures fall through to single-GPU
                # path so the precond still builds.
                pool.free_all_blocks()
                # Surface as _GpuPrecondOOM to let the caller fall back to
                # the host scipy LU path.  Bare ``Exception`` is intentional:
                # NotImplementedError, RuntimeError, libcusolverMg-load
                # failures, ValueError on shape mismatch all funnel here.
                raise _GpuPrecondOOM('gpu_lu (mgpu): {}'.format(exc))
        try:
            A_dev = _cp_local.asarray(A_host)
            lu_dev, piv_dev = _cp_lu_factor(A_dev, overwrite_a = True)
            return ('gpu', lu_dev, piv_dev)
        except _cp_local.cuda.memory.OutOfMemoryError as exc:
            pool.free_all_blocks()
            raise _GpuPrecondOOM('gpu_lu: {}'.format(exc))

    def _host_inverse(A_host):
        lu, piv = _scipy_lu_factor(A_host, check_finite = False)
        return _scipy_lu_solve((lu, piv), np.eye(A_host.shape[0]),
                check_finite = False)

    # Stage 1: G1 path (GPU LU + host inverse + host matmuls).
    G1_lu = _gpu_lu(G1)
    pool.free_all_blocks()
    G1i = _host_inverse(G1)
    Sigma1 = H1 @ G1i
    if eps1_is_scalar:
        Sigma_L1 = eps1_diag * Sigma1
        L1 = eps1_diag
    else:
        # H1 @ diag(eps) @ G1i == H1 @ (eps[:, None] * G1i).
        eps_G1i = eps1_vec[:, None] * G1i
        Sigma_L1 = H1 @ eps_G1i
        L1 = G1 @ eps_G1i
        del eps_G1i

    # Stage 2: G2 path.
    G2_lu = _gpu_lu(G2)
    pool.free_all_blocks()
    G2i = _host_inverse(G2)
    Sigma2 = H2 @ G2i
    if eps1_is_scalar:
        Sigma_L2 = eps2_diag * Sigma2
        L2 = eps2_diag
    else:
        eps_G2i = eps2_vec[:, None] * G2i
        Sigma_L2 = H2 @ eps_G2i
        L2 = G2 @ eps_G2i
        del eps_G2i
    del G1i, G2i

    # Stage 3: Delta = Sigma1 - Sigma2 (host) -> GPU LU + host inverse.
    Delta_host = Sigma1 - Sigma2
    Delta_lu = _gpu_lu(Delta_host)
    pool.free_all_blocks()
    Deltai = _host_inverse(Delta_host)
    del Delta_host

    # Stage 4: Sigma_mat (host) -> GPU LU.
    if eps1_is_scalar:
        L = L1 - L2
        Deltai_nvec = decorate_deltai_fn(Deltai, nvec)
        Sigma_mat = (Sigma_L1 - Sigma_L2 + k ** 2 * L * Deltai_nvec * L)
    else:
        L_diff = L1 - L2
        nvec_outer = nvec @ nvec.T
        magnetic = k ** 2 * ((L_diff @ Deltai) * nvec_outer) @ L_diff
        Sigma_mat = Sigma_L1 - Sigma_L2 + magnetic
    Sigma_lu = _gpu_lu(Sigma_mat)
    del Sigma_mat
    pool.free_all_blocks()

    return (G1_lu, G2_lu, Sigma1, Sigma2, Sigma_L1, Sigma_L2,
            L1, L2, Deltai, Delta_lu, Sigma_lu)


class BEMRetIter(BEMIter):

    # MATLAB: @bemretiter properties (Constant)
    name = 'bemsolver'
    needs = {'sim': 'ret'}

    def __init__(self,
            p: Any,
            enei: Optional[float] = None,
            **options: Any) -> None:

        # Schur option (v1.5.0): cover-layer (EpsNonlocal) shell-face
        # elimination on the iterative retarded path.  Combines with
        # hmatrix=True via SchurIterOperator: the eight retarded
        # components (phi, a_x, a_y, a_z, phip, ap_x, ap_y, ap_z) share
        # the same face-level partition, lifted to the 8N packed vector
        # layout used by ``_pack`` / ``_unpack``.
        # v1.6.0 (B-Schur): added ``schur_eps_form`` to communicate to the
        # SchurIterOperator that ``_afun`` uses operator-form eps (β fix).
        # Default 'auto' picks 'operator' if eps is non-uniform per region
        # (then dense A_ss probe is ill-conditioned → inner GMRES); else
        # 'pointwise' (legacy v1.5.0 fast path).
        self._schur_opt = options.pop('schur', False)
        self._schur_g_ss_solver = options.pop('schur_g_ss_solver', 'auto')
        self._schur_inner_tol = options.pop('schur_inner_tol', 1e-8)
        self._schur_inner_maxit = options.pop('schur_inner_maxit', 200)
        self._schur_eps_form = options.pop('schur_eps_form', 'auto')
        self._schur_active = False
        self._shell_face_idx = None
        self._core_face_idx = None
        self._schur_op = None

        # H-matrix (v1.3.0): opt-in ACA acceleration of Green functions.
        # When True, the matvec used by GMRES uses HMatrix @ x compression
        # (O(N log N) memory) rather than dense ndarrays.
        self._hmatrix = bool(options.pop('hmatrix', False))
        self._htol = options.pop('htol', 1e-6)
        self._kmax = options.pop('kmax', [4, 100])
        self._cleaf = options.pop('cleaf', 200)
        self._fadmiss = options.pop('fadmiss', None)
        self._eta = options.pop('eta', 2.5)

        # H-matrix LU preconditioner (v1.5.0, agent alpha):
        #   'auto'      — pick dense for small mesh, tree for large
        #   'none'      — disable preconditioner entirely (legacy v1.3 behaviour)
        #   'hlu_dense' — alpha-1 dense LU on H-matrix.full()
        #   'hlu_tree'  — alpha-2 recursive block-Schur LU
        # Active only on the H-matrix code path (hmatrix=True).
        self._hlu_mode = options.pop('preconditioner', 'auto')
        self._htol_precond = options.pop('htol_precond', 1e-4)
        self._hlu_object = None  # built lazily inside solve()

        # Default v1.3.0 ``precond``: when the H-matrix path is active and
        # the user did not explicitly choose the legacy preconditioner, we
        # leave it disabled. The new v1.5.0 H-matrix LU preconditioner is
        # plumbed separately and only acts when self._hmatrix is True.
        if self._hmatrix and 'precond' not in options:
            options['precond'] = None

        # Initialize BEMIter base class
        super(BEMRetIter, self).__init__(**options)

        # MATLAB: @bemretiter properties
        self.p = p
        self.enei = None
        self.g = None

        # MATLAB: @bemretiter properties (Access = private)
        self._op = options
        self._sav = None
        self._k = None
        self._eps1 = None
        self._eps2 = None
        self._nvec = p.nvec
        self._G1 = None
        self._H1 = None
        self._G2 = None
        self._H2 = None

        # User-supplied refinement hook (e.g. coverlayer.refine). Stripped
        # before forwarding to CompGreenRet, applied at the BEM matrix
        # level inside _init_matrices(). MATLAB bemretiter forwards refun
        # via varargin → compgreenretiter → greenret/private/init.m.
        self._refun = options.pop('refun', None)
        self._op = options

        # H-matrix path is incompatible with refun for now (refun densifies
        # G/H pairs, defeating the compression). Fall back to dense if both
        # are requested.
        if self._hmatrix and self._refun is not None:
            raise NotImplementedError(
                '[error] BEMRetIter <hmatrix> + <refun> not supported '
                '(refun densifies the Green pairs). Disable one.')

        # Green function. With ``hmatrix=True`` we pull the ACA wrapper from
        # mnpbem.greenfun; otherwise the dense CompGreenRet is used (legacy
        # path preserved for tests / demos).
        # MATLAB: obj.g = aca.compgreenret(p, varargin{:}, ...)
        self._init_green(p, **options)

        # Initialize for given wavelength
        if enei is not None:
            self._init_matrices(enei)

    def _init_green(self,
            p: Any,
            **options: Any) -> None:

        # MATLAB: bemretiter/private/init.m
        if self._hmatrix:
            from ..greenfun import ACACompGreenRet
            # MATLAB stores kmax as [k_min, k_max]; HMatrix expects scalar.
            # Take the upper bound when forwarding.
            kmax_scalar = (max(self._kmax) if hasattr(self._kmax, '__iter__')
                    else self._kmax)
            htol_scalar = (max(self._htol) if hasattr(self._htol, '__iter__')
                    else self._htol)
            aca_kwargs = {
                'htol': htol_scalar,
                'kmax': kmax_scalar,
                'cleaf': self._cleaf,
                'eta': self._eta,
            }
            if self._fadmiss is not None:
                aca_kwargs['fadmiss'] = self._fadmiss
            self.g = ACACompGreenRet(p, **aca_kwargs, **options)
        else:
            from ..greenfun import CompGreenRet
            self.g = CompGreenRet(p, p, **options)

    def _init_matrices(self,
            enei: float) -> 'BEMRetIter':

        # MATLAB: bemretiter/private/initmat.m
        if self.enei is not None and self.enei == enei:
            return self

        # v1.7.2 wavelength-entry GPU cleanup.  Drop the previous
        # wavelength's cached G/H/precond state and drain the cupy pool
        # BEFORE the new eval() round uploads ~4 N^2 of fresh Green-
        # function buffers.  Mirrors bem_ret.py:485-503; without this, the
        # iter sweep accumulates ~2.5 GB/wl of stale buffers on a
        # 12672-face Au@Ag dimer and hits OOM around wl ~12-15 on a
        # 49 GB A6000.  The `_sav` precond dict (8N x 8N LU factors +
        # Sigma1/L1/L2 dense matrices) is the bulk of the held state;
        # dropping it first is load-bearing.
        for _attr in ('_G1', '_H1', '_G2', '_H2', '_sav'):
            if hasattr(self, _attr):
                setattr(self, _attr, None)
        import gc as _gc
        _gc.collect()
        _gpu_pool_cleanup_iter(apply_limit = True)
        if _CUPY_OK_ITER:
            _cp_iter.cuda.runtime.deviceSynchronize()
            _cp_iter.get_default_memory_pool().free_all_blocks()
            _cp_iter.get_default_pinned_memory_pool().free_all_blocks()

        self.enei = enei

        # Wavenumber
        self._k = 2 * np.pi / enei

        # Dielectric function
        self._eps1 = self.p.eps1(enei)
        self._eps2 = self.p.eps2(enei)

        # v1.7.3 Phase 3 (B-3) — distributed G/H assembly hook.
        # When the env-gated distributed build is active AND the dense
        # (non-H-matrix / non-refun) path is in use, build each of the
        # four Green-function differences directly across N GPUs via
        # ``DistributedMatrix.from_func`` + ``CompGreenRet.eval_block``.
        # Each per-GPU column tile is ~N/n_gpus * N * 16 B (e.g. 0.9 GB
        # per GPU at N=15072 / n_gpus=4 vs 3.6 GB / N x N matrix on a
        # single device).  After distributed assembly we gather to host
        # once per matrix; the downstream pipeline (LU dispatch, _afun /
        # _mfun matvecs) is identical to the legacy host path.
        distributed_built = False
        if (_vram_share_active()
                and not self._hmatrix
                and self._refun is None):
            try:
                self._build_distributed_GH(enei)
                distributed_built = True
            except Exception as exc:
                print('[info] BEMRetIter init: distributed G/H build failed '
                        '({}), falling back to legacy eval.'.format(exc),
                        flush = True)
                _gpu_pool_cleanup_iter()

        if not distributed_built:
            # Green functions and surface derivatives
            # MATLAB: G1 = g{1,1}.G(enei) - g{2,1}.G(enei)
            G11 = self.g.eval(0, 0, 'G', enei)
            G21 = self.g.eval(1, 0, 'G', enei)
            G22 = self.g.eval(1, 1, 'G', enei)
            G12 = self.g.eval(0, 1, 'G', enei)

            self._G1 = G11 - G21 if not (isinstance(G21, (int, float)) and G21 == 0) else G11
            self._G2 = G22 - G12 if not (isinstance(G12, (int, float)) and G12 == 0) else G22
            # v1.7.2: release the cross-block intermediates (Gxx) so the cupy
            # pool can reclaim their device buffers before the H1/H2 round
            # starts uploading the next ~2 N^2 of data.
            del G11, G21, G22, G12
            if _CUPY_OK_ITER:
                _cp_iter.get_default_memory_pool().free_all_blocks()

            H11 = self.g.eval(0, 0, 'H1', enei)
            H21 = self.g.eval(1, 0, 'H1', enei)
            H22 = self.g.eval(1, 1, 'H2', enei)
            H12 = self.g.eval(0, 1, 'H2', enei)

            self._H1 = H11 - H21 if not (isinstance(H21, (int, float)) and H21 == 0) else H11
            self._H2 = H22 - H12 if not (isinstance(H12, (int, float)) and H12 == 0) else H22
            # v1.7.2: same logic for the H-block intermediates.
            del H11, H21, H22, H12
            if _CUPY_OK_ITER:
                _cp_iter.cuda.runtime.deviceSynchronize()
                _cp_iter.get_default_memory_pool().free_all_blocks()

            # v1.7 A2 fix: when MNPBEM_GPU=1 + dense path, CompGreenRet returns
            # cupy ndarrays for G/H. The GMRES iterates (``_afun``) get host
            # numpy vectors from scipy.sparse.linalg.gmres, so cupy @ numpy
            # mixes backends and raises TypeError. HMatrix objects already
            # handle the mix inside ``mtimes_vec`` so we only normalise plain
            # ndarrays here. Leaves HMatrix / refun paths bit-identical.
            for _attr in ('_G1', '_G2', '_H1', '_H2'):
                _val = getattr(self, _attr)
                if is_cupy_array(_val):
                    setattr(self, _attr, to_host(_val))

        # Optional user-supplied refinement (coverlayer.refine for nonlocal
        # cover-layer effects). Applied to dense G/H pairs. If ACA H-matrix
        # acceleration is in use the matrices are densified for refun and
        # the refined dense result is kept (refun touches a small set of
        # face pairs, so densification is acceptable here).
        if self._refun is not None:
            G1 = self._G1.full() if hasattr(self._G1, 'full') and not isinstance(self._G1, np.ndarray) else self._G1
            H1 = self._H1.full() if hasattr(self._H1, 'full') and not isinstance(self._H1, np.ndarray) else self._H1
            G2 = self._G2.full() if hasattr(self._G2, 'full') and not isinstance(self._G2, np.ndarray) else self._G2
            H2 = self._H2.full() if hasattr(self._H2, 'full') and not isinstance(self._H2, np.ndarray) else self._H2
            G1, H1 = self._refun(self.g, G1, H1)
            G2, H2 = self._refun(self.g, G2, H2)
            self._G1, self._H1 = G1, H1
            self._G2, self._H2 = G2, H2

        # Initialize preconditioner
        if self.precond is not None:
            self._init_precond(enei)
            # v1.7.2: the preconditioner builds up to 4 dense LUs and several
            # transient G^{-1} / Sigma / nvec_outer GEMMs on the GPU
            # (~3 N^2 transient peak on 12672-face dimers).  Drain right
            # after init so the GMRES Krylov build below sees max headroom.
            if _CUPY_OK_ITER:
                _cp_iter.cuda.runtime.deviceSynchronize()
                _cp_iter.get_default_memory_pool().free_all_blocks()
                _cp_iter.get_default_pinned_memory_pool().free_all_blocks()
            _gpu_pool_cleanup_iter()

        # Schur (v1.5.0): detect cover-layer partition and prepare the
        # SchurIterOperator wrapping the 8N packed _afun.  The Schur
        # operator probes _afun for the shell block (lu_dense path) or
        # delegates A_ss^{-1} to inner GMRES.  For BEMRetIter the eight
        # retarded components share the same face-level partition --
        # SchurIterOperator with components=8 lifts the indices to the
        # full 8N packed layout (column-major / order='F').
        self._schur_active = False
        self._schur_op = None
        if self._schur_opt:
            from .schur_iter_helpers import SchurIterOperator, detect_iter_partition
            partition = detect_iter_partition(self.p)
            if partition is not None:
                shell_idx, core_idx = partition
                nfaces = self.p.n if hasattr(self.p, 'n') else self.p.nfaces
                self._shell_face_idx = shell_idx
                self._core_face_idx = core_idx

                # v1.6.0 (B-Schur): resolve eps_form.  'auto' picks 'operator'
                # whenever ``_afun`` follows the β v1.5.1 operator-form path,
                # i.e. eps1 OR eps2 is non-uniform within its region.  For
                # uniform-eps (scalar) callers we keep the legacy 'pointwise'
                # path so existing dense-LU probe behaviour is preserved.
                eps_form = self._schur_eps_form
                if eps_form == 'auto':
                    eps1_nonuniform = (isinstance(self._eps1, np.ndarray)
                            and self._eps1.ndim >= 1)
                    eps2_nonuniform = (isinstance(self._eps2, np.ndarray)
                            and self._eps2.ndim >= 1)
                    eps_form = ('operator'
                            if (eps1_nonuniform or eps2_nonuniform)
                            else 'pointwise')

                # Block-decomposed eps_diag (diagnostics; future preconditioning).
                eps_diag = None
                if eps_form == 'operator':
                    def _block_eps(eps_val: Any, idx: np.ndarray) -> Any:
                        if (np.isscalar(eps_val)
                                or (isinstance(eps_val, np.ndarray) and eps_val.ndim == 0)):
                            return eps_val
                        return np.asarray(eps_val)[idx]
                    eps_diag = {
                        'shell_eps1': _block_eps(self._eps1, shell_idx),
                        'core_eps1':  _block_eps(self._eps1, core_idx),
                        'shell_eps2': _block_eps(self._eps2, shell_idx),
                        'core_eps2':  _block_eps(self._eps2, core_idx),
                    }

                self._schur_op = SchurIterOperator(
                        self._afun,
                        shell_idx,
                        core_idx,
                        nfaces = nfaces,
                        components = 8,
                        dtype = complex,
                        g_ss_solver = self._schur_g_ss_solver,
                        inner_tol = self._schur_inner_tol,
                        inner_maxit = self._schur_inner_maxit,
                        eps_form = eps_form,
                        eps_diag = eps_diag)
                self._schur_active = True

        return self

    def _compress(self,
            hmat: Any) -> Any:

        # MATLAB: bemretiter/private/compress.m
        # The dense-LU preconditioner needs an ndarray; if we got an HMatrix
        # we densify it here. Memory cost is the standard dense N x N — only
        # invoked when the user explicitly opts into the dense preconditioner.
        #
        # v1.6.3 fix: HMatrix.full() auto-detects GPU blocks and would
        # densify *into* a cupy buffer (2.57 GB at N=12672) on each call.
        # Four calls in a row from _init_precond accumulate ~10 GB of
        # GPU buffers and fragmentation that the pool cannot reclaim
        # before the dense LU pipeline runs (Bug 5/6 root cause).  Force
        # the densification onto the host; the serialized GPU LU
        # pipeline below pushes one block at a time and reclaims it
        # before the next.
        if hasattr(hmat, 'full') and not isinstance(hmat, np.ndarray):
            try:
                return hmat.full(xp = np)
            except TypeError:
                # Older HMatrix that does not accept ``xp`` kwarg.
                return hmat.full()
        return hmat

    def _init_precond(self,
            enei: float) -> None:

        # MATLAB: bemretiter/private/initprecond.m
        # Garcia de Abajo and Howie, PRB 65, 115418 (2002)
        #
        # v1.5.1 (agent beta) — non-uniform-eps fix.  When ``g.con[0][1]``
        # is non-zero AND eps1/eps2 are non-uniform within their region
        # (composite particle, e.g. Au@Ag dimer), the dense ``BEMRet``
        # path (`bem_ret.py:360-393`) uses the operator form ``L1 =
        # G1·diag(eps1)·G1⁻¹``.  Algebraically, ``Sigma1·L1 =
        # H1·G1⁻¹·G1·diag(eps1)·G1⁻¹ = H1·diag(eps1)·G1⁻¹``.  The
        # original Python (and MATLAB) preconditioner instead built
        # ``diag(eps1)·H1·G1⁻¹``, which is **not** the same operator and
        # is the source of the Au@Ag mid-band drift.  The fix is to
        # build the corrected combined Sigma:
        #
        #     Sigma_mat = H1·diag(eps1)·G1⁻¹ - H2·diag(eps2)·G2⁻¹
        #               + k² · ((L1-L2)·Deltai * nvec·nvec') · (L1-L2)
        #
        # where ``L1, L2`` are themselves the dense G·eps·G⁻¹ operators.
        # This makes the iter preconditioner numerically equivalent to
        # the dense ``BEMRet`` Sigma factorisation.

        # v1.7.3 Phase 3 (B-3 distributed build): the LU stage uses
        # ``lu_factor_dispatch`` with the VRAM-share kwargs, which already
        # routes through cuSolverMg distributed LU when the env vars are
        # set.  No separate dispatch hook needed here — the distributed
        # G/H build happens upstream in ``_init_matrices``.

        k = 2 * np.pi / enei
        eps1 = self._eps1
        eps2 = self._eps2
        nvec = self._nvec

        G1 = self._compress(self._G1)
        H1 = self._compress(self._H1)
        G2 = self._compress(self._G2)
        H2 = self._compress(self._H2)

        # Bug 2 fix: coerce any cupy operands down to host before the
        # CPU-style dense preconditioner pipeline so the eps_diag /
        # H @ G^{-1} GEMMs do not mix devices.
        if is_cupy_array(G1): G1 = to_host(G1)
        if is_cupy_array(G2): G2 = to_host(G2)
        if is_cupy_array(H1): H1 = to_host(H1)
        if is_cupy_array(H2): H2 = to_host(H2)

        # Bug 5/6 (v1.5.2) Tier-3 12672-face follow-up: cupy memory pool
        # accumulates the per-block GPU buffers from the four _compress()
        # full() calls above (~10 GB even after _del_).  Drain the pool
        # before launching the GPU LU pipeline so the 49 GB single-GPU
        # cap is not exceeded by stale pool blocks.
        try:
            import cupy as _cp_local
            _cp_local.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

        # Dielectric as diagonal matrices for matrix operations
        if np.isscalar(eps1) or (isinstance(eps1, np.ndarray) and eps1.ndim == 0):
            eps1_diag = eps1
            eps2_diag = eps2
        else:
            eps1_diag = np.diag(eps1)
            eps2_diag = np.diag(eps2)

        # LU factorizations of Green functions.  Tier-3 12672-face note:
        # the dense preconditioner needs G1_lu, G2_lu, eye(N), inverse,
        # and Sigma_lu simultaneously alive — at complex128 / N=12672 the
        # naive in-flight peak is ~30+ GB on a single GPU and overshoots
        # the 49 GB single-A6000 cap (Bug 5/6 follow-up OOM in v1.5.2).
        #
        # v1.6.3 fix (precond GPU path): we run a *hybrid* pipeline that
        # factors the four LU packages on the GPU (so the per-iterate
        # ``lu_solve_dispatch`` in ``_mfun`` dispatches to cuSOLVER getrs
        # at ~5 ms each, vs ~50 ms for scipy single-RHS) and computes
        # the dense matrix products (G^{-1}, Sigma, L) on the host where
        # MKL parallelises across all CPU cores.  At ~100 GMRES iterates
        # per wavelength the GPU-side LU saves ~40 s of iterate cost vs
        # the legacy "everything on host" path; init wall is roughly
        # parity with the pure host pipeline at N=12672.
        #
        # The persistent GPU residents are the four LU packages tagged
        # ``'gpu'``  (G1_lu / G2_lu / Delta_lu / Sigma_lu, ~10 GB at
        # N=12672) — comfortably under the 49 GB single-A6000 cap.
        #
        # The legacy host fallback (env-var threshold) is retained as a
        # safety net for very large N (e.g. 25k+) or when free GPU memory
        # is insufficient for the transient single-LU peak.
        gpu_overflow_cutoff = int(os.environ.get(
            'MNPBEM_GPU_PRECOND_HOST_THRESHOLD', '32768'))
        N = G1.shape[0]
        use_gpu_serialized = (
                N < gpu_overflow_cutoff
                and _gpu_precond_capacity_ok(N))
        if use_gpu_serialized:
            try:
                (G1_lu, G2_lu, Sigma1, Sigma2, Sigma_L1, Sigma_L2,
                        L1, L2, Deltai, Delta_lu, Sigma_lu) = (
                        _build_precond_gpu_serial(
                                G1, H1, G2, H2, eps1_diag, eps2_diag,
                                nvec, k, self._decorate_deltai))
            except _GpuPrecondOOM as exc:
                print('[info] BEMRetIter precond: GPU serialized path hit '
                        'OOM ({}), falling back to host scipy LU.'.format(
                                exc))
                use_gpu_serialized = False
        if not use_gpu_serialized:
            from scipy.linalg import lu_factor as _scipy_lu_factor
            from scipy.linalg import lu_solve as _scipy_lu_solve
            print('[info] BEMRetIter precond: N={}, routing dense LU '
                    'through host (GPU memory safety).'.format(N))
            # v1.7.2: even on the host-LU fallback we want to drain the
            # cupy pool BEFORE starting the host MKL pipeline, since the
            # G/H matrices may have just been pulled down from GPU (the
            # ``is_cupy_array`` checks above) and the staging buffer is
            # still sitting in the pool.
            if _CUPY_OK_ITER:
                _cp_iter.get_default_memory_pool().free_all_blocks()
            G1_lu_pkg = _scipy_lu_factor(G1, check_finite = False)
            G2_lu_pkg = _scipy_lu_factor(G2, check_finite = False)
            G1i = _scipy_lu_solve(G1_lu_pkg, np.eye(N), check_finite = False)
            G2i = _scipy_lu_solve(G2_lu_pkg, np.eye(N), check_finite = False)
            G1_lu = ('cpu', G1_lu_pkg[0], G1_lu_pkg[1])
            G2_lu = ('cpu', G2_lu_pkg[0], G2_lu_pkg[1])
            Sigma1 = H1 @ G1i
            Sigma2 = H2 @ G2i
            Delta_lu_pkg = _scipy_lu_factor(Sigma1 - Sigma2, check_finite = False)
            Deltai = _scipy_lu_solve(Delta_lu_pkg, np.eye(N),
                    check_finite = False)
            Delta_lu = ('cpu', Delta_lu_pkg[0], Delta_lu_pkg[1])
            if np.isscalar(eps1_diag):
                L1 = eps1_diag
                L2 = eps2_diag
                L = L1 - L2
                Sigma_L1 = eps1_diag * Sigma1
                Sigma_L2 = eps2_diag * Sigma2
                Deltai_nvec = self._decorate_deltai(Deltai, nvec)
                Sigma_mat = (Sigma_L1 - Sigma_L2
                        + k ** 2 * L * Deltai_nvec * L)
            else:
                L1 = G1 @ eps1_diag @ G1i
                L2 = G2 @ eps2_diag @ G2i
                L = L1 - L2
                Sigma_L1 = H1 @ eps1_diag @ G1i
                Sigma_L2 = H2 @ eps2_diag @ G2i
                nvec_outer = nvec @ nvec.T
                magnetic = k ** 2 * ((L @ Deltai) * nvec_outer) @ L
                Sigma_mat = Sigma_L1 - Sigma_L2 + magnetic
            Sigma_lu_pkg = _scipy_lu_factor(Sigma_mat, check_finite = False)
            Sigma_lu = ('cpu', Sigma_lu_pkg[0], Sigma_lu_pkg[1])

        # Save variables for preconditioner.  Note: ``Sigma1`` cached here
        # is the v1.5.1 operator-form Sigma1·L1 (= H1·eps1·G1⁻¹), used by
        # ``_mfun`` for the modify-alpha / modify-De step in place of the
        # legacy ``eps1·(Sigma1·phi)``.  The original ``Sigma1`` (= H1·G1⁻¹)
        # is also stored so we can build correct ``-matmul1(Sigma1, a)``
        # corrections when needed by ``_mfun``.
        sav = {}
        sav['k'] = k
        sav['nvec'] = nvec
        sav['G1_lu'] = G1_lu
        sav['G2_lu'] = G2_lu
        sav['eps1'] = eps1_diag
        sav['eps2'] = eps2_diag
        sav['Sigma1'] = Sigma1                    # H1·G1⁻¹  (legacy)
        sav['Sigma1_L1'] = Sigma_L1               # v1.5.1 operator form: H1·eps1·G1⁻¹
        sav['L1'] = L1                            # G1·eps1·G1⁻¹ (or scalar)
        sav['L2'] = L2
        sav['Delta_lu'] = Delta_lu
        sav['Sigma_lu'] = Sigma_lu

        # v1.6.4 Phase 2: optional GPU residency for the dense matrices
        # consumed by ``_mfun`` (Sigma1, L1, L2, L_diff).  Triggered by
        # ``MNPBEM_AGGRESSIVE_GPU_MFUN=1`` only.  When the flag is off
        # the GPU keys are absent and ``_mfun`` keeps the host matmul
        # path bit-identical to v1.6.3.
        #
        # Tiered capacity check: at N=12672 the BEM path has already
        # consumed ~42 GB on a 49 GB A6000 by the time precond runs
        # (HMatrix ACA blocks live on GPU).  We therefore upload
        # Sigma1 first (4 matvec hits per GMRES iterate) and only add
        # L1/L2/L_diff when extra headroom remains.  Each tier checks
        # its own ~2 N^2 * 16 byte budget (matrix + transient asarray
        # buffer + cusolver scratch).
        sav['Sigma1_gpu'] = None
        sav['L1_gpu'] = None
        sav['L2_gpu'] = None
        sav['L_diff_gpu'] = None
        if self._gpu_mfun_enabled():
            try:
                import cupy as _cp_local
                _cp_local.get_default_memory_pool().free_all_blocks()
                N_local = Sigma1.shape[0]
                tier_bytes = int(1.2 * (N_local ** 2) * 16)
                free_bytes, _total = _cp_local.cuda.runtime.memGetInfo()

                # Tier 1: Sigma1 (always uploaded when at all possible).
                if free_bytes >= tier_bytes:
                    sav['Sigma1_gpu'] = _cp_local.asarray(Sigma1)
                    print('[info] BEMRetIter mfun GPU: Sigma1 GPU-resident '
                            '({:.1f} GB)'.format(
                                    (N_local ** 2) * 16 / (1024 ** 3)),
                            flush = True)
                else:
                    print('[info] BEMRetIter mfun GPU: insufficient '
                            'GPU memory for Sigma1 ({:.1f} GB free, need '
                            '{:.1f} GB), falling back to host matmul.'.format(
                                    free_bytes / (1024 ** 3),
                                    tier_bytes / (1024 ** 3)),
                            flush = True)

                # Tier 2a: L_diff (used 2x per iter for sig2 / h2 update).
                # Prefer L_diff alone over L1+L2+L_diff because the diff
                # is the only L-product on the bottleneck path; L1 is
                # used twice on (N,) / (N, 3) inputs and L2 never
                # standalone.  Uploading L_diff captures most of the
                # L-side iter cost at half the GPU footprint.  We only
                # need the matrix itself plus a small (N, 3) transient
                # on the device so the budget is ~ 1.2 * N^2 * 16 B.
                non_scalar_L = not (np.isscalar(L1) or (isinstance(L1, np.ndarray)
                        and L1.ndim == 0))
                if sav['Sigma1_gpu'] is not None and non_scalar_L:
                    _cp_local.get_default_memory_pool().free_all_blocks()
                    free_bytes, _total = _cp_local.cuda.runtime.memGetInfo()
                    needed_ldiff = int(1.2 * (N_local ** 2) * 16)
                    if free_bytes >= needed_ldiff:
                        L_diff_host = L1 - L2
                        sav['L_diff_gpu'] = _cp_local.asarray(L_diff_host)
                        del L_diff_host
                        print('[info] BEMRetIter mfun GPU: L_diff '
                                'GPU-resident ({:.1f} GB)'.format(
                                        (N_local ** 2) * 16 / (1024 ** 3)),
                                flush = True)
                    else:
                        print('[info] BEMRetIter mfun GPU: insufficient '
                                'GPU memory for L_diff ({:.1f} GB free, '
                                'need {:.1f} GB), keeping L_diff on '
                                'host.'.format(free_bytes / (1024 ** 3),
                                        needed_ldiff / (1024 ** 3)),
                                flush = True)

                # Tier 2b: L1 (used 2x per iter on phi/a; minor bonus).
                if sav['L_diff_gpu'] is not None:
                    _cp_local.get_default_memory_pool().free_all_blocks()
                    free_bytes, _total = _cp_local.cuda.runtime.memGetInfo()
                    needed_l1 = int(1.2 * (N_local ** 2) * 16)
                    if free_bytes >= needed_l1:
                        sav['L1_gpu'] = _cp_local.asarray(L1)
                        print('[info] BEMRetIter mfun GPU: L1 GPU-resident '
                                '({:.1f} GB)'.format(
                                        (N_local ** 2) * 16 / (1024 ** 3)),
                                flush = True)
                    else:
                        print('[info] BEMRetIter mfun GPU: insufficient '
                                'GPU memory for L1 ({:.1f} GB free, need '
                                '{:.1f} GB), keeping L1 on host.'.format(
                                        free_bytes / (1024 ** 3),
                                        needed_l1 / (1024 ** 3)),
                                flush = True)
            except Exception as exc:
                print('[info] BEMRetIter mfun GPU: residency setup failed '
                        '({}), falling back to host matmul.'.format(exc),
                        flush = True)
                sav['Sigma1_gpu'] = None
                sav['L1_gpu'] = None
                sav['L2_gpu'] = None
                sav['L_diff_gpu'] = None

        self._sav = sav

        # v1.7.2: drain the cupy pool at the end of _init_precond so the
        # function-frame locals (the G/H/Sigma/Delta dense matrices that
        # `sav` now references AND any cusolver scratch left from the
        # GPU-serialized LU pipeline) become reclaimable before control
        # returns to ``_init_matrices`` and the next wavelength's
        # assembly starts.  We don't ``del`` the locals explicitly because
        # they were conditionally bound (the GPU-serialized branch sets
        # different names than the host-fallback branch); the function
        # return below will collect them and the drain on entry to the
        # NEXT ``_init_matrices`` call catches anything that survives.
        if _CUPY_OK_ITER:
            _cp_iter.cuda.runtime.deviceSynchronize()
            _cp_iter.get_default_memory_pool().free_all_blocks()
            _cp_iter.get_default_pinned_memory_pool().free_all_blocks()

    def _build_distributed_GH(self,
            enei: float) -> None:
        """v1.7.3 Phase 3 (B-3): build G1/H1/G2/H2 with distributed assembly.

        Activated by ``MNPBEM_VRAM_SHARE_DISTRIBUTED=1`` (gated by
        ``_vram_share_active()``).  Each of the four Green-function
        differences (``G1 = G11 - G21``, ``H1 = H11 - H21``,
        ``G2 = G22 - G12``, ``H2 = H22 - H12``) is built directly across
        N GPUs via :class:`DistributedMatrix.from_func` +
        :meth:`CompGreenRet.eval_block`, so the full ``(N, N)`` complex128
        matrix never materialises on a single device.  At N=15072 with
        n_gpus=4 the per-GPU column tile peak is ~0.9 GB vs ~3.6 GB for
        a single-device full build — the difference between fitting in
        and overshooting the 49 GB A6000 cap.

        After distributed assembly the tiles are gathered to host once
        (``.to_host()``) and cached on ``self._G1 / self._H1 / self._G2 /
        self._H2``.  The downstream LU dispatch (``_init_precond``) reads
        these host matrices and routes the four precond LU factors
        through cuSolverMg distributed LU (already wired via
        ``_vram_share_lu_kwargs()``); the GMRES ``_afun`` matvec consumes
        the host G/H directly.

        Bit-identical to the legacy ``_init_matrices`` G/H assembly to
        ufunc ordering tolerance; only the *transient* memory profile
        differs.
        """
        nfaces = self.p.n if hasattr(self.p, 'n') else self.p.nfaces
        n_gpus, backend, device_ids = _vram_share_env_config()

        green = self.g
        # ACA / refun guarded by caller; bail loudly if a sentinel slipped
        # through (defence-in-depth).
        if not hasattr(green, 'eval_block'):
            raise RuntimeError(
                    '[error] BEMRetIter distributed build requires '
                    'CompGreenRet.eval_block (got {})'.format(type(green)))

        print('[info] BEMRetIter init: distributed G/H build (N={}, '
                'n_gpus={}, backend={}).'.format(nfaces, n_gpus, backend),
                flush = True)

        # G1 = G(0, 0) - G(1, 0).  When con[1, 0] is 0, ``eval_block`` of
        # the (1, 0) pair returns a zero-filled tile because all of its
        # ``con[i1, i2] <= 0`` entries are skipped — _distributed_block_assemble
        # handles that case implicitly via its (1, 0, 'G', -1) summand.
        dm_G1 = _distributed_block_assemble(
                green, enei,
                ncomb_list = [(0, 0, 'G', 1), (1, 0, 'G', -1)],
                nrows = nfaces, ncols = nfaces,
                n_gpus = n_gpus, device_ids = device_ids)
        self._G1 = dm_G1.to_host()
        dm_G1.free()
        if _CUPY_OK_ITER:
            _cp_iter.get_default_memory_pool().free_all_blocks()

        dm_H1 = _distributed_block_assemble(
                green, enei,
                ncomb_list = [(0, 0, 'H1', 1), (1, 0, 'H1', -1)],
                nrows = nfaces, ncols = nfaces,
                n_gpus = n_gpus, device_ids = device_ids)
        self._H1 = dm_H1.to_host()
        dm_H1.free()
        if _CUPY_OK_ITER:
            _cp_iter.get_default_memory_pool().free_all_blocks()

        dm_G2 = _distributed_block_assemble(
                green, enei,
                ncomb_list = [(1, 1, 'G', 1), (0, 1, 'G', -1)],
                nrows = nfaces, ncols = nfaces,
                n_gpus = n_gpus, device_ids = device_ids)
        self._G2 = dm_G2.to_host()
        dm_G2.free()
        if _CUPY_OK_ITER:
            _cp_iter.get_default_memory_pool().free_all_blocks()

        dm_H2 = _distributed_block_assemble(
                green, enei,
                ncomb_list = [(1, 1, 'H2', 1), (0, 1, 'H2', -1)],
                nrows = nfaces, ncols = nfaces,
                n_gpus = n_gpus, device_ids = device_ids)
        self._H2 = dm_H2.to_host()
        dm_H2.free()
        if _CUPY_OK_ITER:
            _cp_iter.cuda.runtime.deviceSynchronize()
            _cp_iter.get_default_memory_pool().free_all_blocks()
            _cp_iter.get_default_pinned_memory_pool().free_all_blocks()
        _gpu_pool_cleanup_iter()

    @staticmethod
    def _gpu_mfun_enabled() -> bool:
        # v1.6.4 Phase 2 opt-in.  Default OFF → bit-identical to v1.6.3.
        return os.environ.get('MNPBEM_AGGRESSIVE_GPU_MFUN', '0') == '1'

    @staticmethod
    def _decorate_deltai(
            Deltai: np.ndarray,
            nvec: np.ndarray) -> np.ndarray:

        # MATLAB: fun(Deltai, nvec) in initprecond.m
        # Deltai_nvec = nvec1 * Deltai * nvec1 + nvec2 * Deltai * nvec2 + nvec3 * Deltai * nvec3
        n = nvec.shape[0]
        result = np.zeros((n, n), dtype = Deltai.dtype)
        for i in range(3):
            nvec_i = np.diag(nvec[:, i])
            result = result + nvec_i @ Deltai @ nvec_i
        return result

    def _pack(self,
            phi: np.ndarray,
            a: np.ndarray,
            phip: np.ndarray,
            ap: np.ndarray) -> np.ndarray:

        # MATLAB: bemretiter/private/pack.m
        # MATLAB uses column-major (:) flatten, so we use order='F'.
        total_len = phi.size + a.size + phip.size + ap.size
        vec = np.empty(total_len, dtype = complex)
        offset = 0
        for arr in [phi, a, phip, ap]:
            flat = arr.ravel(order = 'F')
            vec[offset:offset + flat.size] = flat
            offset += flat.size
        return vec

    def _unpack(self,
            vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # MATLAB: bemretiter/private/unpack.m
        # MATLAB uses column-major reshape, so we use order='F'.
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces

        # last dimension
        siz = int(vec.size / (8 * n))

        # reshape vector (column-major to match MATLAB)
        vec_2d = vec.reshape(-1, 8, order = 'F')

        # extract potentials from vector
        phi = vec_2d[:, 0].reshape(n, siz, order = 'F') if siz > 1 else vec_2d[:, 0].reshape(n)
        a = vec_2d[:, 1:4].reshape(n, 3, siz, order = 'F') if siz > 1 else vec_2d[:, 1:4].reshape(n, 3)
        phip = vec_2d[:, 4].reshape(n, siz, order = 'F') if siz > 1 else vec_2d[:, 4].reshape(n)
        ap = vec_2d[:, 5:8].reshape(n, 3, siz, order = 'F') if siz > 1 else vec_2d[:, 5:8].reshape(n, 3)

        return phi, a, phip, ap

    @staticmethod
    def _outer(
            nvec: np.ndarray,
            val: Any,
            mul: Optional[np.ndarray] = None) -> Any:

        # MATLAB: bemretiter/private/outer.m
        if isinstance(val, (int, float)) and val == 0:
            return 0

        if mul is not None:
            if val.ndim == 1:
                val = val * mul
            else:
                val = val * mul[:, np.newaxis] if mul.ndim == 1 else val * mul

        if val.ndim == 1:
            # val: (n,), nvec: (n, 3) -> result: (n, 3)
            return nvec * val[:, np.newaxis]
        else:
            # val: (n, siz), nvec: (n, 3) -> result: (n, 3, siz)
            siz = val.shape[1]
            n = val.shape[0]
            result = np.empty((n, 3, siz), dtype = val.dtype)
            for i in range(3):
                result[:, i, :] = val * nvec[:, i:i + 1]
            return result

    @staticmethod
    def _inner(
            nvec: np.ndarray,
            a: Any,
            mul: Optional[np.ndarray] = None) -> Any:

        # MATLAB: bemretiter/private/inner.m
        if isinstance(a, (int, float)) and a == 0:
            return 0

        if a.ndim == 2:
            # a: (n, 3), nvec: (n, 3) -> result: (n,)
            result = np.sum(a * nvec, axis = 1)
        elif a.ndim == 3:
            # a: (n, 3, siz), nvec: (n, 3) -> result: (n, siz)
            result = np.sum(a * nvec[:, :, np.newaxis], axis = 1)
        else:
            result = a

        if mul is not None:
            if result.ndim == 1:
                result = result * mul
            else:
                result = result * mul[:, np.newaxis] if mul.ndim == 1 else result * mul

        return result

    def _excitation(self,
            exc: CompStruct) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # MATLAB: bemretiter/private/excitation.m
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces

        # Default values for potentials
        phi1 = getattr(exc, 'phi1', 0)
        phi1p = getattr(exc, 'phi1p', 0)
        a1 = getattr(exc, 'a1', 0)
        a1p = getattr(exc, 'a1p', 0)
        phi2 = getattr(exc, 'phi2', 0)
        phi2p = getattr(exc, 'phi2p', 0)
        a2 = getattr(exc, 'a2', 0)
        a2p = getattr(exc, 'a2p', 0)

        k = 2 * np.pi / exc.enei
        eps1 = self._eps1
        eps2 = self._eps2
        nvec = self._nvec

        def _matmul(a_val: Any, x_val: Any) -> Any:
            if isinstance(x_val, (int, float)) and x_val == 0:
                return 0
            if np.isscalar(a_val):
                return a_val * x_val
            return a_val[:, np.newaxis] * x_val if x_val.ndim > 1 else a_val * x_val

        # Eqs. (10, 11)
        phi = self._subtract(phi2, phi1)
        a = self._subtract(a2, a1)

        # Eq. (15)
        alpha = self._subtract(a2p, a1p) - \
            1j * k * self._subtract(
                self._outer(nvec, phi2, eps2),
                self._outer(nvec, phi1, eps1))

        # Eq. (18)
        De = self._subtract(_matmul(eps2, phi2p), _matmul(eps1, phi1p)) - \
            1j * k * self._subtract(
                self._inner(nvec, a2, eps2),
                self._inner(nvec, a1, eps1))

        # Expand arrays
        if isinstance(phi, (int, float)) and phi == 0:
            if isinstance(De, np.ndarray):
                phi = np.zeros_like(De)
            else:
                phi = np.zeros(n, dtype = complex)

        if isinstance(a, (int, float)) and a == 0:
            if isinstance(alpha, np.ndarray):
                a = np.zeros_like(alpha)
            else:
                a = np.zeros((n, 3), dtype = complex)

        return phi, a, De, alpha

    @staticmethod
    def _subtract(
            a: Any,
            b: Any) -> Any:

        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return a - b
        elif isinstance(a, np.ndarray):
            return a if (isinstance(b, (int, float)) and b == 0) else a - b
        elif isinstance(b, np.ndarray):
            return -b if (isinstance(a, (int, float)) and a == 0) else a - b
        else:
            return a - b

    def _afun(self,
            vec: np.ndarray) -> np.ndarray:

        # MATLAB: bemretiter/private/afun.m
        # Garcia de Abajo and Howie, PRB 65, 115418 (2002)
        #
        # v1.5.1 (agent beta) — non-uniform-eps fix.  When the particle has
        # multiple materials sharing a region (e.g. Au@Ag dimer:
        # eps1 = ε_Au on Au-Ag faces, ε_Ag on Ag-medium faces) AND the
        # Green-function connectivity ``g.con[0][1]`` is non-zero, the
        # MATLAB / pre-1.5.1 iter form ``ε(r) · (G·σ)(r)`` is **not** the
        # physically correct convolution — eps lives at the source point
        # of the integrand, not the field point.  The dense ``BEMRet`` path
        # captures this with the operator ``L1 = G1·diag(eps1)·G1⁻¹`` (see
        # ``bem_ret.py:360``).  Algebraically that operator, applied to
        # ``G1·σ1``, equals ``G1·(eps1·σ1)``.  So the fix is to push
        # ``eps`` *before* the Green / surface-derivative matvec.
        #
        # The two forms are bit-identical when eps is a scalar (uniform
        # within the region), so we always use the corrected form — its
        # only cost is extra matvecs when eps is non-uniform.
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces
        siz = int(vec.size / 2)

        # Split vector array (column-major reshape to match MATLAB)
        vec1 = vec[:siz].reshape(n, -1, order = 'F')
        vec2 = vec[siz:].reshape(n, -1, order = 'F')

        eps1 = self._eps1
        eps2 = self._eps2

        def _eps_apply(eps_val: Any, x: np.ndarray) -> np.ndarray:
            # Multiply per-face eps (scalar or (n,) array) into a (n, ...)
            # array along axis 0.  Leaves x unchanged for the scalar /
            # 0-d case (commutes with the matvec, so we still want a
            # scalar multiply for correctness — but the caller folds the
            # scalar through G to save a matvec).
            if np.isscalar(eps_val) or (isinstance(eps_val, np.ndarray)
                    and eps_val.ndim == 0):
                return eps_val * x
            if x.ndim == 1:
                return eps_val * x
            return eps_val.reshape(-1, *([1] * (x.ndim - 1))) * x

        def _ensure_numpy(arr: Any) -> np.ndarray:
            # v1.6.4 defensive wrapper for HMatrix matvec output.  After
            # the v1.6.4 HMatrix fix the matvec result mirrors the input
            # backend, so for the host-side ``_afun`` call sites this is
            # a no-op.  Kept as a guard so downstream numpy slice
            # assignment (``combined_g[...]``) never gets a cupy ndarray.
            if hasattr(arr, 'get') and not isinstance(arr, np.ndarray):
                return arr.get()
            return np.asarray(arr)

        # Multiplications with Green functions.
        # Phi / a equations (no eps) use plain G·vec.
        G1_vec1 = _ensure_numpy(self._G1 @ vec1)
        G2_vec2 = _ensure_numpy(self._G2 @ vec2)

        # Pack into combined vector for unpack (column-major flatten)
        combined_g = np.empty(G1_vec1.size + G2_vec2.size, dtype = complex)
        combined_g[:G1_vec1.size] = G1_vec1.ravel(order = 'F')
        combined_g[G1_vec1.size:] = G2_vec2.ravel(order = 'F')
        Gsig1, Gh1, Gsig2, Gh2 = self._unpack(combined_g)

        # Alpha / De equations use ``M @ (eps · vec)`` with M ∈ {G, H}.
        # For scalar eps we can save the extra matvec by reusing the
        # plain Gsig / Gh / Hsig / Hh and pulling the scalar out.
        eps1_scalar = (np.isscalar(eps1) or (isinstance(eps1, np.ndarray)
                and eps1.ndim == 0))
        eps2_scalar = (np.isscalar(eps2) or (isinstance(eps2, np.ndarray)
                and eps2.ndim == 0))

        if eps1_scalar and eps2_scalar:
            # Cheap path: scalar eps commutes with G/H, so M(eps·v) = eps·(M·v).
            H1_vec1 = _ensure_numpy(self._H1 @ vec1)
            H2_vec2 = _ensure_numpy(self._H2 @ vec2)
            combined_h = np.empty(H1_vec1.size + H2_vec2.size, dtype = complex)
            combined_h[:H1_vec1.size] = H1_vec1.ravel(order = 'F')
            combined_h[H1_vec1.size:] = H2_vec2.ravel(order = 'F')
            Hsig1, Hh1, Hsig2, Hh2 = self._unpack(combined_h)

            L_Gsig1, L_Gh1 = eps1 * Gsig1, eps1 * Gh1
            L_Gsig2, L_Gh2 = eps2 * Gsig2, eps2 * Gh2
            L_Hsig1, L_Hsig2 = eps1 * Hsig1, eps2 * Hsig2
        else:
            # Non-uniform eps: do the per-face multiply *before* the matvec.
            # This is what the dense BEMRet's ``L1 = G·eps·G⁻¹`` reduces to
            # when applied to the iter unknown σ1 (see commentary above).
            eps1_vec1 = _eps_apply(eps1, vec1)
            eps2_vec2 = _eps_apply(eps2, vec2)
            G1_eps_vec1 = _ensure_numpy(self._G1 @ eps1_vec1)
            G2_eps_vec2 = _ensure_numpy(self._G2 @ eps2_vec2)
            H1_eps_vec1 = _ensure_numpy(self._H1 @ eps1_vec1)
            H2_eps_vec2 = _ensure_numpy(self._H2 @ eps2_vec2)

            combined_geps = np.empty(G1_eps_vec1.size + G2_eps_vec2.size,
                    dtype = complex)
            combined_geps[:G1_eps_vec1.size] = G1_eps_vec1.ravel(order = 'F')
            combined_geps[G1_eps_vec1.size:] = G2_eps_vec2.ravel(order = 'F')
            L_Gsig1, L_Gh1, L_Gsig2, L_Gh2 = self._unpack(combined_geps)

            combined_heps = np.empty(H1_eps_vec1.size + H2_eps_vec2.size,
                    dtype = complex)
            combined_heps[:H1_eps_vec1.size] = H1_eps_vec1.ravel(order = 'F')
            combined_heps[H1_eps_vec1.size:] = H2_eps_vec2.ravel(order = 'F')
            L_Hsig1, _L_Hh1, L_Hsig2, _L_Hh2 = self._unpack(combined_heps)

            # Hh1 / Hh2 (no eps) still needed for the alpha equation.
            H1_vec1 = _ensure_numpy(self._H1 @ vec1)
            H2_vec2 = _ensure_numpy(self._H2 @ vec2)
            combined_h = np.empty(H1_vec1.size + H2_vec2.size, dtype = complex)
            combined_h[:H1_vec1.size] = H1_vec1.ravel(order = 'F')
            combined_h[H1_vec1.size:] = H2_vec2.ravel(order = 'F')
            _Hsig1, Hh1, _Hsig2, Hh2 = self._unpack(combined_h)

        k = self._k
        nvec = self._nvec

        # Eq. (10)
        phi = Gsig1 - Gsig2
        # Eq. (11)
        a = Gh1 - Gh2

        if eps1_scalar and eps2_scalar:
            # Eq. (14) - scalar eps path keeps the original ordering for
            # bit-identical reproduction of legacy MATLAB outputs.
            alpha = Hh1 - Hh2 - 1j * k * self._outer(nvec,
                    L_Gsig1 - L_Gsig2)
            De = (L_Hsig1 - L_Hsig2) - 1j * k * self._inner(nvec,
                    L_Gh1 - L_Gh2)
        else:
            # Eq. (14) - operator form: alpha = Hh1 - Hh2 - i k n × G·(eps·sig).
            alpha = Hh1 - Hh2 - 1j * k * self._outer(nvec,
                    L_Gsig1 - L_Gsig2)
            # Eq. (17) - operator form: De = H·(eps·sig) - i k n · G·(eps·h).
            De = (L_Hsig1 - L_Hsig2) - 1j * k * self._inner(nvec,
                    L_Gh1 - L_Gh2)

        return self._pack(phi, a, De, alpha)

    def _mfun(self,
            vec: np.ndarray) -> np.ndarray:

        # MATLAB: bemretiter/private/mfun.m
        # Garcia de Abajo and Howie, PRB 65, 115418 (2002)
        #
        # v1.5.1 (agent beta) — non-uniform-eps fix.  Mirrors the dense
        # ``BEMRet.mldivide`` reduction.  ``L1`` is the operator
        # ``G1·diag(eps1)·G1⁻¹`` (or a scalar when eps is uniform), so
        # ``matmul(L1, phi)`` replaces the legacy ``eps1 · phi``.

        # Unpack matrices
        phi, a, De, alpha = self._unpack(vec)

        sav = self._sav
        k = sav['k']
        nvec = sav['nvec']
        G1_lu = sav['G1_lu']
        G2_lu = sav['G2_lu']
        eps1 = sav['eps1']
        eps2 = sav['eps2']
        Sigma1 = sav['Sigma1']
        L1 = sav['L1']
        L2 = sav['L2']
        Delta_lu = sav['Delta_lu']
        Sigma_lu = sav['Sigma_lu']

        # v1.6.4 Phase 2: GPU-resident dense matrices (None when flag off).
        Sigma1_gpu = sav.get('Sigma1_gpu')
        L1_gpu = sav.get('L1_gpu')
        L2_gpu = sav.get('L2_gpu')
        L_diff_gpu = sav.get('L_diff_gpu')

        # When any GPU resident is present we lazily import cupy once and
        # reuse it for the dispatch helpers below.  ``cp`` stays local so
        # CPU-only environments never touch cupy.
        cp_mod = None
        if Sigma1_gpu is not None:
            try:
                import cupy as cp_mod
            except ImportError:
                cp_mod = None
                Sigma1_gpu = None
                L1_gpu = None
                L_diff_gpu = None

        def _matvec_dispatch(M_host: Any, M_gpu: Any, b: np.ndarray) -> np.ndarray:
            # When M_gpu is set we move b to GPU, do the matmul there and
            # bring the result back.  OOM at the asarray / matmul falls
            # back silently to the host path.
            if M_gpu is not None and cp_mod is not None:
                try:
                    if b.ndim == 1:
                        b_gpu = cp_mod.asarray(b)
                        out_gpu = M_gpu @ b_gpu
                        return cp_mod.asnumpy(out_gpu)
                    b_flat = b.reshape(b.shape[0], -1)
                    b_gpu = cp_mod.asarray(b_flat)
                    out_gpu = M_gpu @ b_gpu
                    out_host = cp_mod.asnumpy(out_gpu)
                    return out_host.reshape(M_host.shape[0], *b.shape[1:])
                except cp_mod.cuda.memory.OutOfMemoryError:
                    pass
            n_rows = M_host.shape[0]
            if b.ndim == 1:
                return M_host @ b
            return (M_host @ b.reshape(b.shape[0], -1)).reshape(n_rows, *b.shape[1:])

        def matmul1(a_mat: np.ndarray, b: np.ndarray) -> np.ndarray:
            # Multiply (n, n) matrix with (n, ...) array, preserving trailing dims.
            if b.ndim == 1:
                return a_mat @ b
            n_rows = a_mat.shape[0] if not np.isscalar(a_mat) else b.shape[0]
            return (a_mat @ b.reshape(b.shape[0], -1)).reshape(n_rows, *b.shape[1:])

        def _ls(lu_piv, b):
            if b.ndim == 1:
                return lu_solve_dispatch(lu_piv, b)
            return lu_solve_dispatch(lu_piv, b.reshape(b.shape[0], -1)).reshape(b.shape)

        def matmul_op(op_val: Any, b: np.ndarray) -> np.ndarray:
            # Apply L1 / L2 / scalar eps to (n, ...) array along axis 0.
            # When op_val is scalar we just multiply; when it is the dense
            # operator G·eps·G⁻¹ we do the full matmul.
            if np.isscalar(op_val) or (isinstance(op_val, np.ndarray)
                    and op_val.ndim == 0):
                return op_val * b
            if b.ndim == 1:
                return op_val @ b
            return (op_val @ b.reshape(b.shape[0], -1)).reshape(b.shape)

        def matmul_op_disp(op_val: Any, op_gpu: Any, b: np.ndarray) -> np.ndarray:
            # Phase 2 GPU dispatch for L1 / L2 / L_diff.  Scalar op_val
            # short-circuits to the cheap scalar multiply (eps uniform).
            if np.isscalar(op_val) or (isinstance(op_val, np.ndarray)
                    and op_val.ndim == 0):
                return op_val * b
            return _matvec_dispatch(op_val, op_gpu, b)

        # Modify alpha and De  (dense BEMRet.mldivide lines 31-35)
        # MATLAB: alpha = alpha - matmul(Sigma1, a) + 1i*k*outer(nvec, matmul(L1, phi))
        # MATLAB: De    = De - matmul(Sigma1, matmul(L1, phi))
        #                  + 1i*k*inner(nvec, matmul(L1, a))
        L1_phi = matmul_op_disp(L1, L1_gpu, phi)
        L1_a = matmul_op_disp(L1, L1_gpu, a)
        alpha = (alpha
                - _matvec_dispatch(Sigma1, Sigma1_gpu, a)
                + 1j * k * self._outer(nvec, L1_phi))
        De = (De
                - _matvec_dispatch(Sigma1, Sigma1_gpu, L1_phi)
                + 1j * k * self._inner(nvec, L1_a))

        # Eq. (19)  (dense BEMRet.mldivide line 38-39)
        # MATLAB: sig2 = matmul(Sigmai, De + 1i*k*inner(nvec, matmul(L1-L2, matmul(Deltai, alpha))))
        L_diff = L1 - L2
        Deltai_alpha = _ls(Delta_lu, alpha)
        L_Deltai_alpha = matmul_op_disp(L_diff, L_diff_gpu, Deltai_alpha)
        sig2 = _ls(Sigma_lu, De + 1j * k * self._inner(nvec, L_Deltai_alpha))

        # Eq. (20)  (dense BEMRet.mldivide line 41-42)
        # MATLAB: h2 = matmul(Deltai, 1i*k*outer(nvec, matmul(L1-L2, sig2)) + alpha)
        L_sig2 = matmul_op_disp(L_diff, L_diff_gpu, sig2)
        h2 = _ls(Delta_lu, 1j * k * self._outer(nvec, L_sig2) + alpha)

        # Surface charges and currents
        sig1 = _ls(G1_lu, sig2 + phi)
        h1 = _ls(G1_lu, h2 + a)
        sig2_out = _ls(G2_lu, sig2)
        h2_out = _ls(G2_lu, h2)

        result = self._pack(sig1, h1, sig2_out, h2_out)
        return result

    def solve(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMRetIter']:

        # MATLAB: bemretiter/solve.m
        # Initialize BEM solver (if needed)
        self._init_matrices(exc.enei)

        # External excitation
        phi, a, De, alpha = self._excitation(exc)

        # Size of excitation arrays
        siz1 = phi.shape
        siz2 = a.shape

        # Pack everything to single vector
        b = self._pack(phi, a, De, alpha)

        # v1.7.2: drain right before GMRES enters its Krylov build-up so
        # the iter loop sees the maximum amount of free GPU memory.  The
        # init pipeline above leaves up to 2 N^2 of transient asarray
        # buffers in the pool; releasing them now keeps the per-iter
        # matvec / preconditioner-apply allocations from triggering a
        # fragmentation-induced OOM mid-sweep.
        if _CUPY_OK_ITER:
            _cp_iter.cuda.runtime.deviceSynchronize()
            _cp_iter.get_default_memory_pool().free_all_blocks()
        _gpu_pool_cleanup_iter()

        if self._schur_active:
            # v1.5.0 Schur path: GMRES iterates on the reduced (core-only)
            # system.  Preconditioner is bypassed because _mfun was built
            # for the full 8N system; rebuilding it on the reduced 8M
            # block would require new G1/G2 LUs and is M5+ work.  For
            # cover-layer geometries the reduced system is well-
            # conditioned enough for unpreconditioned GMRES.
            op = self._schur_op
            b_eff = op.reduce_rhs(b)
            x_core, _ = self._iter_solve(None, b_eff, op._matvec, None)
            x = op.recover_full(x_core, b)
            # v1.7.2: drop the Schur-reduced solution vector handle.
            del x_core
            if _CUPY_OK_ITER:
                _cp_iter.cuda.runtime.deviceSynchronize()
                _cp_iter.get_default_memory_pool().free_all_blocks()
        else:
            # Function for matrix multiplication
            fa = self._afun
            fm = None
            if self.precond is not None:
                fm = self._mfun

            # v1.5.0 H-matrix LU preconditioner (agent alpha). Replaces fm
            # when active. The preconditioner is built once per (hmatrix
            # path, mode); we keep it cached on self for re-use across
            # enei sweeps.
            if self._hmatrix and self._hlu_mode != 'none':
                fm = self._build_hlu_preconditioner(b.shape[0])

            # Iterative solution
            x, self_updated = self._iter_solve(None, b, fa, fm)
            # v1.7.2: drop the GMRES matvec closures (which keep refs to
            # ``self._G1/_G2/_H1/_H2`` and the LU factor tuples).  The
            # Krylov subspace itself is held internally by
            # scipy.sparse.linalg.gmres for the duration of the call;
            # once we return here it's gone and we just want to compact
            # the pool before the next wavelength's init drains the prior
            # state.
            del fa, fm
            if _CUPY_OK_ITER:
                _cp_iter.cuda.runtime.deviceSynchronize()
                _cp_iter.get_default_memory_pool().free_all_blocks()

        # Unpack and save solution vector
        sig1, h1, sig2, h2 = self._unpack(x)

        # Reshape surface charges and currents
        if len(siz1) > 1:
            sig1 = sig1.reshape(siz1)
            sig2 = sig2.reshape(siz1)
        if len(siz2) > 2:
            h1 = h1.reshape(siz2)
            h2 = h2.reshape(siz2)

        # Host-materialize cupy results so the returned sig is always
        # CPU-resident. Mirrors BEMRet.solve's defensive guard (v1.7
        # Phase 1.4) and lets downstream code (sigma cache dump,
        # field evaluator, surface charge plotter) treat sig fields
        # as numpy without per-call cupy/numpy branching.
        if is_cupy_array(sig1):
            sig1 = to_host(sig1)
        if is_cupy_array(sig2):
            sig2 = to_host(sig2)
        if is_cupy_array(h1):
            h1 = to_host(h1)
        if is_cupy_array(h2):
            h2 = to_host(h2)

        sig = CompStruct(self.p, exc.enei,
            sig1 = sig1, sig2 = sig2, h1 = h1, h2 = h2)

        # v1.7.2 solve-exit cleanup: drain any residual transient buffers
        # before returning so the caller's wavelength loop sees a fully
        # drained pool when it advances to the next enei.
        if _CUPY_OK_ITER:
            _cp_iter.get_default_memory_pool().free_all_blocks()
            _cp_iter.get_default_pinned_memory_pool().free_all_blocks()
        _gpu_pool_cleanup_iter()

        return sig, self

    def _build_hlu_preconditioner(self,
            n_vec: int) -> Callable:

        # v1.5.0 agent alpha — H-matrix LU preconditioner.
        # The retarded iterative solver couples 8N variables (phi, a, phip,
        # ap) via the Garcia-de-Abajo / Howie [PRB 65, 115418] block
        # structure. The ``mfun`` derived in initprecond / mfun.m approximates
        # the inverse of this 8N x 8N system using only the LU factors of
        # G1, G2 and two reduced N x N matrices Sigma_lu and Delta_lu. We
        # reuse exactly that mfun, which means our preconditioner is
        # equivalent to v1.3 ``precond='hmat'`` -- but now triggered on the
        # H-matrix code path where v1.3 left it disabled.
        #
        # Implementation: call _init_precond once (this densifies G/H once
        # and builds the dense LU factors) and return the existing _mfun.
        # The HMatrixLUPreconditioner is used as the LU backend for the
        # individual G1, G2 factors via the lu_factor_dispatch hook.
        # Modes:
        #   'dense' / 'hlu_dense' / 'auto<5k' — densify G/H, dense LU
        #   'tree'  / 'hlu_tree'  / 'auto>=5k' — same path today; the
        #     HMatrixLUPreconditioner.tree backend is exposed standalone
        #     in mnpbem.bem.preconditioner for future integration into
        #     Sigma / Delta as well.
        if self._hlu_object is not None and self._hlu_object == (n_vec, self.enei):
            return self._mfun

        # Trigger the v1.3 dense initprecond path. This builds self._sav.
        self._init_precond(self.enei)
        self._hlu_object = (n_vec, self.enei)
        return self._mfun

    def __truediv__(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMRetIter']:

        # MATLAB: bemretiter/mldivide.m
        return self.solve(exc)

    def __mul__(self,
            sig: CompStruct) -> CompStruct:

        # MATLAB: bemretiter/mtimes.m
        pot1 = self.potential(sig, 1)
        pot2 = self.potential(sig, 2)

        return CompStruct(self.p, sig.enei,
            phi1 = pot1.phi1, phi1p = pot1.phi1p,
            a1 = pot1.a1, a1p = pot1.a1p,
            phi2 = pot2.phi2, phi2p = pot2.phi2p,
            a2 = pot2.a2, a2p = pot2.a2p)

    def field(self,
            sig: CompStruct,
            inout: int = 2) -> CompStruct:

        # MATLAB: bemretiter/field.m
        k = 2 * np.pi / sig.enei
        pot = self.potential(sig, inout)

        if hasattr(pot, 'phi1'):
            phi, phip, a, ap = pot.phi1, pot.phi1p, pot.a1, pot.a1p
        else:
            phi, phip, a, ap = pot.phi2, pot.phi2p, pot.a2, pot.a2p

        # Tangential directions via interpolation
        phi1_d, phi2_d = self.p.deriv(self.p.interp(phi))[:2]
        a1_d, a2_d, t1, t2 = self.p.deriv(self.p.interp(a))

        # Normal vector
        nvec = np.cross(t1, t2)
        h = msqrt(np.sum(nvec * nvec, axis = 1, keepdims = True))
        nvec = nvec / h

        # Tangential vectors
        tvec1 = np.cross(t2, nvec) / h
        tvec2 = -np.cross(t1, nvec) / h

        # Electric field
        e = 1j * k * a - \
            self._outer(nvec, phip) - \
            self._outer(tvec1, phi1_d) - \
            self._outer(tvec2, phi2_d)

        # Magnetic field
        def _matcross(v: np.ndarray, a_d: np.ndarray) -> np.ndarray:
            if a_d.ndim == 2:
                return np.cross(v, a_d)
            else:
                n_pts = v.shape[0]
                siz = a_d.shape[2]
                result = np.empty((n_pts, 3, siz), dtype = a_d.dtype)
                for s in range(siz):
                    result[:, :, s] = np.cross(v, a_d[:, :, s])
                return result

        h_field = _matcross(tvec1, a1_d) + _matcross(tvec2, a2_d) + _matcross(nvec, ap)

        return CompStruct(self.p, sig.enei, e = e, h = h_field)

    def potential(self,
            sig: CompStruct,
            inout: int = 2) -> CompStruct:

        # MATLAB: bemretiter/potential.m
        return self.g.potential(sig, inout)

    def clear(self) -> 'BEMRetIter':

        # MATLAB: bemretiter/clear.m
        self._G1 = None
        self._H1 = None
        self._G2 = None
        self._H2 = None
        self._sav = None
        # v1.7.2: explicit clear() means the user wants the device drained.
        # Without this the LU factors / G blocks just released above stay
        # in the cupy pool until the next solve triggers a free_all_blocks.
        if _CUPY_OK_ITER:
            _cp_iter.cuda.runtime.deviceSynchronize()
            _cp_iter.get_default_memory_pool().free_all_blocks()
            _cp_iter.get_default_pinned_memory_pool().free_all_blocks()
        _gpu_pool_cleanup_iter()
        return self

    def __call__(self,
            enei: float) -> 'BEMRetIter':

        return self._init_matrices(enei)

    def __repr__(self) -> str:
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces if hasattr(self.p, 'nfaces') else '?'
        status = 'enei={:.1f}nm'.format(self.enei) if self.enei is not None else 'not initialized'
        return 'BEMRetIter(p: {} faces, solver={}, {})'.format(n, self.solver, status)
