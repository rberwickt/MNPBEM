import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator

from ..greenfun import CompStruct
from ..utils.gpu import lu_factor_dispatch, lu_solve_dispatch, to_host, is_cupy_array
from ..utils.matlab_compat import msqrt
from .bem_iter import BEMIter


# ---------------------------------------------------------------------------
# v1.7.3 B-3 quasi-static distributed-build helpers (Agent D pattern)
# ---------------------------------------------------------------------------

def _vram_share_lu_kwargs_stat() -> dict:
    """Read MNPBEM_VRAM_SHARE_* env vars and return kwargs for lu_factor_dispatch.

    Mirrors ``bem_ret_iter._vram_share_lu_kwargs`` exactly (the helper is
    duplicated here so the BEM module stays a single import boundary).
    Returns an empty dict when VRAM-share is disabled so the call site is
    bit-identical to the single-GPU path.  When enabled the returned
    kwargs route ``lu_factor_dispatch`` through ``factor_multi_gpu``
    (cuSolverMg by default) and the matrix is partitioned across
    ``n_gpus`` devices.  Required for very large quasi-static meshes
    (e.g. 25k+ faces) where the dense LU plus G/F/Lambda residents
    exceed the 49 GB single-A6000 cap.
    """
    if os.environ.get('MNPBEM_VRAM_SHARE', '0') != '1':
        return {}
    n_gpus = int(os.environ.get('MNPBEM_VRAM_SHARE_GPUS', '1'))
    if n_gpus <= 1:
        return {}
    backend = os.environ.get('MNPBEM_VRAM_SHARE_BACKEND', 'cusolvermg')
    return {'n_gpus': n_gpus, 'backend': backend}


def _vram_share_active_stat() -> bool:
    """Return True when distributed-build is enabled for quasi-static iter.

    Activation requirements mirror ``bem_ret_iter._vram_share_active``:
    - ``MNPBEM_VRAM_SHARE=1``
    - ``MNPBEM_VRAM_SHARE_DISTRIBUTED=1`` (gates the heavier distributed
      build path; default off so existing call sites stay bit-identical)
    - ``MNPBEM_VRAM_SHARE_GPUS>=2``
    - cupy + cuSolverMg are importable

    Distributed build assembles F (quasi-static BEM matrix) directly
    across N GPUs via ``DistributedMatrix.from_func`` + the eval_block
    callback, avoiding the host-resident full ``N x N`` matrix on a
    single device during the precond LU pipeline.
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
    try:
        import cupy as _cp_probe  # noqa: F401
    except Exception:
        return False
    try:
        from ..utils.multi_gpu_lu import cusolvermg_available
        return bool(cusolvermg_available())
    except Exception:
        return False


def _vram_share_env_config_stat() -> Tuple[int, str, Optional[List[int]]]:
    """Resolve (n_gpus, backend, device_ids) for the distributed path.

    Caller should already have checked ``_vram_share_active_stat()``.
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


def _vram_share_distributed_kwargs_stat() -> Dict[str, Any]:
    """Return ``{n_gpus, device_ids, block_size}`` for DistributedMatrix.

    v1.8 (VRAM-Schur): mirror of ``bem_stat._vram_share_distributed_kwargs``
    used by the BEMStatIter distributed-Schur path.  Honours
    ``MNPBEM_VRAM_SHARE_GPUS`` / ``MNPBEM_VRAM_SHARE_DEVICE_IDS`` and the
    cuSolverMg-recommended 256-column block size.
    """
    n_gpus, _backend, device_ids = _vram_share_env_config_stat()
    return {'n_gpus': n_gpus, 'device_ids': device_ids, 'block_size': 256}


def _compgreen_stat_eval_block(green: Any,
        key: str,
        c0: int,
        c1: int) -> np.ndarray:
    """Evaluate a column slice ``[c0, c1)`` of a CompGreenStat scalar key.

    v1.7.3 Phase 3 (B-3) placeholder for ``CompGreenStat.eval_block`` while
    Agent L is in flight.  Once Agent L lands, the call site below will be
    swapped to ``green.eval_block(key, c0, c1)`` (no other change needed).

    Quasi-static Green-function matrices (G, F, H1, H2) are
    wavelength-independent and pre-computed in ``CompGreenStat.__init__``,
    so the slice can be served directly from the cached self.G / self.F
    arrays.  H1/H2 add the diagonal ±2π correction only on the closed-
    surface (p1 is p2) diagonal blocks; that correction must be applied
    over the slice's global row indices when the slice overlaps the
    main diagonal.

    Parameters
    ----------
    green : CompGreenStat
        Green function instance.  Must already have ``self.G`` / ``self.F``
        materialised (the ``__init__`` pipeline does this eagerly).
    key : str
        ``'G'``, ``'F'``, ``'H1'`` or ``'H2'``.
    c0, c1 : int
        Column range (global p2 indices) — ``[c0, c1)``.

    Returns
    -------
    np.ndarray
        Sliced matrix of shape ``(p1.n, c1 - c0)`` with the same dtype as
        the underlying full matrix.
    """
    c0 = int(c0)
    c1 = int(c1)

    # Defensive: callers should not pass empty slices but DistributedMatrix
    # may issue them when the global N is not divisible by n_gpus.
    if c1 <= c0:
        n_rows = green.G.shape[0] if hasattr(green, 'G') else green.F.shape[0]
        return np.zeros((n_rows, 0), dtype = complex)

    if key == 'G':
        return green.G[:, c0:c1].copy()
    elif key == 'F':
        return green.F[:, c0:c1].copy()
    elif key == 'H1':
        # H1 = F + 2π · I on the closed-surface diagonal.  The slice may
        # overlap the diagonal only where col index == row index, so we
        # add 2π only to the entries (i, i - c0) for i in [c0, c1).
        H1 = green.F[:, c0:c1].copy()
        if green.p1 is green.p2:
            for i in range(c0, c1):
                H1[i, i - c0] += 2.0 * np.pi
        return H1
    elif key == 'H2':
        H2 = green.F[:, c0:c1].copy()
        if green.p1 is green.p2:
            for i in range(c0, c1):
                H2[i, i - c0] -= 2.0 * np.pi
        return H2
    else:
        raise ValueError(
                '[error] _compgreen_stat_eval_block: unsupported key <{}> '
                '(expected G/F/H1/H2)'.format(key))


def _distributed_block_assemble_stat(green: Any,
        ncomb_list: List[Tuple[str, int]],
        nrows: int,
        ncols: int,
        n_gpus: int,
        device_ids: Optional[List[int]]) -> Any:
    """Build a linear combination of CompGreenStat scalar-key blocks distributed.

    Mirrors ``bem_ret_iter._distributed_block_assemble`` but specialised to
    the quasi-static case where Green-function matrices are
    wavelength-independent and there is no (i, j) region pair (the
    composite particle is a single object).  ``ncomb_list`` is a list of
    ``(key, sign)`` tuples, e.g. ``[('F', 1)]`` for plain F,
    ``[('F', 1), ('G', -1)]`` for F - G, etc.

    Per-GPU eval_func sums ``sign · <key-slice>`` over the column tile
    owned by that GPU, then the DistributedMatrix scatter writes the
    result into the per-GPU local_array.  No full ``N x N`` matrix ever
    materialises on a single device during the build.

    Parameters
    ----------
    green : CompGreenStat
        Green function instance (must expose ``G`` and ``F`` attributes).
    ncomb_list : list of (str, int)
        Linear combination terms.
    nrows, ncols : int
        Output matrix shape.
    n_gpus : int
        Number of GPUs to distribute over.
    device_ids : list of int, optional
        Explicit device id mapping; falls back to ``range(n_gpus)``.

    Returns
    -------
    DistributedMatrix
        Caller does ``.to_host()`` when it needs a host copy for the
        downstream LU / matvec pipeline.
    """
    from ..utils.distributed_matrix import DistributedMatrix
    import numpy as _np_local

    try:
        import cupy as _cp_local  # type: ignore
        _cp_ok = True
    except Exception:
        _cp_local = None  # type: ignore
        _cp_ok = False

    # Prefer the real CompGreenStat.eval_block (Agent L) when available.
    # Agent L's signature is (key, enei, col_start, col_stop) — enei is
    # accepted but ignored on the quasi-static side.  Older binaries that
    # don't have eval_block fall back to the in-module placeholder, which
    # slices self.G / self.F directly with the same diagonal correction
    # semantics.
    _have_real_eval_block = hasattr(green, 'eval_block')

    def _safe_eval(key: str, c0: int, c1: int) -> Any:
        if _have_real_eval_block:
            # Try Agent L's 4-arg signature first; if a hypothetical
            # future revision drops enei, fall back to a 3-arg call.
            try:
                return green.eval_block(key, 0.0, c0, c1)
            except TypeError:
                return green.eval_block(key, c0, c1)
        return _compgreen_stat_eval_block(green, key, c0, c1)

    def _is_cupy(x: Any) -> bool:
        return _cp_ok and isinstance(x, _cp_local.ndarray)

    def _coerce_pair(a: Any, b: Any) -> Tuple[Any, Any]:
        if _is_cupy(a) or _is_cupy(b):
            if not _is_cupy(a):
                a = _cp_local.asarray(a, dtype = _np_local.complex128)
            if not _is_cupy(b):
                b = _cp_local.asarray(b, dtype = _np_local.complex128)
        return a, b

    def _evalfn(gpu_idx: int, c0: int, c1: int) -> Any:
        out = None
        for (key, sign) in ncomb_list:
            blk = _safe_eval(key, c0, c1)
            # Coerce dtype up-front so downstream arithmetic is clean.
            if isinstance(blk, (int, float)) and blk == 0:
                blk = _np_local.zeros(
                        (nrows, c1 - c0), dtype = _np_local.complex128)
            if out is None:
                if sign == 1:
                    out = (blk.copy() if hasattr(blk, 'copy')
                            else _np_local.asarray(blk).copy())
                elif sign == -1:
                    out = -blk
                else:
                    out = sign * blk
            else:
                out, blk = _coerce_pair(out, blk)
                if sign == 1:
                    out = out + blk
                elif sign == -1:
                    out = out - blk
                else:
                    out = out + sign * blk
        # Ensure dtype is the DistributedMatrix dtype so scatter is
        # straight memcpy without intermediate cast.
        if hasattr(out, 'dtype') and out.dtype != _np_local.complex128:
            if _is_cupy(out):
                out = out.astype(_np_local.complex128)
            else:
                out = _np_local.asarray(out, dtype = _np_local.complex128)
        return out

    return DistributedMatrix.from_func(
            shape = (nrows, ncols),
            dtype = np.complex128,
            n_gpus = n_gpus,
            eval_func = _evalfn,
            device_ids = device_ids)

# v1.7.2 GPU memory-pool cleanup: mirror BEMRet's wavelength-end immediate free
# pattern so cupy returns blocks to the driver at every wavelength rather than
# accumulating across the sweep (MATLAB-parity).  When cupy is not importable
# the helper becomes a no-op so the CPU path stays untouched.
try:
    import cupy as _cp_stat  # type: ignore
    _CUPY_OK_STAT = True
except Exception:
    _cp_stat = None  # type: ignore
    _CUPY_OK_STAT = False


def _gpu_pool_cleanup_stat(apply_limit: bool = False) -> None:
    """Synchronise CUDA stream then drain cupy default + pinned memory pools.

    Mirrors the v1.7.2 BEMRet cleanup (bem_ret.py:485-503, 734-737): the
    deviceSynchronize() before free_all_blocks() is load-bearing — without
    it, blocks that are still in flight on the CUDA stream are NOT actually
    idle yet and the pool refuses to return them to the driver, so the
    high-water mark keeps creeping up across wavelengths.

    Honours MNPBEM_GPU_POOL_LIMIT_GB (legitimate peaks past this cap will
    OOM; default 0 = uncapped).
    """
    if not _CUPY_OK_STAT:
        return
    try:
        mempool = _cp_stat.get_default_memory_pool()
        pinned = _cp_stat.get_default_pinned_memory_pool()
        if apply_limit:
            try:
                pool_limit_gb = float(os.environ.get(
                        'MNPBEM_GPU_POOL_LIMIT_GB', '0'))
            except (TypeError, ValueError):
                pool_limit_gb = 0.0
            if pool_limit_gb > 0:
                mempool.set_limit(size = int(pool_limit_gb * (1024 ** 3)))
        _cp_stat.cuda.runtime.deviceSynchronize()
        mempool.free_all_blocks()
        pinned.free_all_blocks()
    except Exception:
        pass


class BEMStatIter(BEMIter):

    # MATLAB: @bemstatiter properties (Constant)
    name = 'bemsolver'
    needs = {'sim': 'stat'}

    def __init__(self,
            p: Any,
            enei: Optional[float] = None,
            **options: Any) -> None:

        # Schur option (v1.5.0): cover-layer (EpsNonlocal) shell-face
        # elimination on the iterative path. Combines with hmatrix=True
        # via SchurIterOperator -- no explicit inv(G_ss) is built, only
        # full matvecs and a small shell-block solve are required.
        self._schur_opt = options.pop('schur', False)
        self._schur_g_ss_solver = options.pop('schur_g_ss_solver', 'auto')
        self._schur_inner_tol = options.pop('schur_inner_tol', 1e-8)
        self._schur_inner_maxit = options.pop('schur_inner_maxit', 200)
        self._schur_active = False
        self._shell_face_idx = None
        self._core_face_idx = None
        self._schur_op = None
        # v1.8 (VRAM-Schur): host-side keepalive for the distributed
        # Schur reduction (M_ss LU + M_sc + M_cs + D_inv_C).  None on
        # the legacy single-GPU / CPU paths.
        self._schur_reduce_keepalive = None
        self._schur_dist_reduce_rhs = None
        self._schur_dist_recover = None

        # H-matrix (v1.3.0): opt-in ACA acceleration of F. The matvec used
        # by GMRES then uses HMatrix @ x rather than dense matmul.
        self._hmatrix = bool(options.pop('hmatrix', False))
        self._htol = options.pop('htol', 1e-6)
        self._kmax = options.pop('kmax', [4, 100])
        self._cleaf = options.pop('cleaf', 200)
        self._fadmiss = options.pop('fadmiss', None)

        # H-matrix LU preconditioner (v1.5.0, agent alpha) for the
        # quasistatic iterative solver. See BEMRetIter for the full mode
        # list. Active only when self._hmatrix is True.
        self._hlu_mode = options.pop('preconditioner', 'auto')
        self._htol_precond = options.pop('htol_precond', 1e-4)
        self._hlu_object = None

        # Same default-precond logic as BEMRetIter: don't densify F just to
        # build an LU when the user opted into compression.
        if self._hmatrix and 'precond' not in options:
            options['precond'] = None

        # Initialize BEMIter base class
        super(BEMStatIter, self).__init__(**options)

        # MATLAB: @bemstatiter properties
        self.p = p
        self.enei = None
        self.F = None

        # MATLAB: @bemstatiter properties (Access = private)
        self._op = options
        self._g = None
        self._lambda = None
        self._mat_lu = None

        # Green function
        # MATLAB: obj.g = aca.compgreenstat(p, varargin{:}, 'htol', ...)
        # For iterative solver, Green function is computed as H-matrix
        self._init_green(p, **options)

        # Initialize for given wavelength
        if enei is not None:
            self._init_matrices(enei)

    def _init_green(self,
            p: Any,
            **options: Any) -> None:

        # MATLAB: bemstatiter/private/init.m
        # H-matrix path uses ACACompGreenStat with cluster-tree ACA on F.
        # Dense path uses CompGreenStat (legacy / small mesh / tests).
        # ``hmode`` legacy alias maps onto hmatrix=True.
        hmode = options.pop('hmode', None)
        if self._hmatrix or hmode is not None:
            from ..greenfun import ACACompGreenStat
            kmax_scalar = (max(self._kmax) if hasattr(self._kmax, '__iter__')
                    else self._kmax)
            htol_scalar = (max(self._htol) if hasattr(self._htol, '__iter__')
                    else self._htol)
            aca_kwargs = {
                'htol': htol_scalar,
                'kmax': kmax_scalar,
                'cleaf': self._cleaf,
            }
            if self._fadmiss is not None:
                aca_kwargs['fadmiss'] = self._fadmiss
            self._g = ACACompGreenStat(p, **aca_kwargs, **options)
        else:
            from ..greenfun import CompGreenStat
            self._g = CompGreenStat(p, p, **options)

        # Surface derivative of Green function
        # MATLAB: obj.F = eval(obj.g, 'F')
        # v1.7.3 Phase 3 (B-3): when distributed build is active AND we are
        # on the dense (non-H-matrix) path, build the F matrix via
        # ``DistributedMatrix.from_func`` + ``CompGreenStat.eval_block`` so
        # the per-GPU column tile is ~(N · N/n_gpus · 16 B) instead of
        # the full ``N x N`` complex128 on a single device.  For
        # quasi-static F is wavelength-independent so this happens once
        # at construction time and the result is cached on self.F /
        # self._F_distributed_built.  The downstream LU dispatch
        # (``_init_matrices``) consumes self.F directly.
        #
        # Note: BEMStatIter's F is enei-independent, so unlike BEMRetIter
        # we do NOT rebuild the distributed assembly per-wavelength —
        # the host F is built once and reused for all enei values.
        # The distributed-build memory win is mostly in keeping the build
        # itself off a single GPU; the resulting host F is what feeds
        # the (-Lambda - F) LU below.
        self._F_distributed_built = False
        if (_vram_share_active_stat()
                and not self._hmatrix
                and not (hmode is not None)):
            try:
                self._build_distributed_F()
                self._F_distributed_built = True
            except Exception as exc:
                print('[info] BEMStatIter init: distributed F build failed '
                        '({}), falling back to host F.'.format(exc),
                        flush = True)
                self.F = self._g.F
        else:
            self.F = self._g.F

    def _build_distributed_F(self) -> None:
        """v1.7.3 Phase 3 (B-3): build F via distributed column-tile assembly.

        Activated by ``MNPBEM_VRAM_SHARE_DISTRIBUTED=1`` and gated by
        ``_vram_share_active_stat()``.  The quasi-static F matrix is
        wavelength-independent, so this runs exactly once at solver
        construction (vs once per wavelength for the retarded sibling).

        Per-GPU column tile is ~``N · (N / n_gpus) · 16 B``; at N=15072 with
        n_gpus=4 that's ~0.9 GB per GPU vs ~3.6 GB for a single-device
        full build.  After distributed assembly the tiles are gathered to
        host once via ``.to_host()`` and cached on ``self.F``; the
        downstream LU (``_init_matrices``) reads the host F and routes
        the actual factorisation through cuSolverMg via
        ``_vram_share_lu_kwargs_stat()``.

        Bit-identical to the legacy ``self.F = self._g.F`` path to the
        floating-point tolerance of ufunc ordering; only the *transient*
        memory profile differs (and only when the env vars are set).
        """
        green = self._g
        if not hasattr(green, 'eval_block') and not hasattr(green, 'F'):
            raise RuntimeError(
                    '[error] BEMStatIter distributed F build requires '
                    'CompGreenStat with eval_block or .F (got {})'
                    .format(type(green)))

        nfaces = green.F.shape[0]
        n_gpus, backend, device_ids = _vram_share_env_config_stat()

        print('[info] BEMStatIter init: distributed F build '
                '(N={}, n_gpus={}, backend={}).'
                .format(nfaces, n_gpus, backend),
                flush = True)

        dm_F = _distributed_block_assemble_stat(
                green,
                ncomb_list = [('F', 1)],
                nrows = nfaces, ncols = nfaces,
                n_gpus = n_gpus, device_ids = device_ids)
        # Gather once to host so the downstream _init_matrices/_init_precond
        # pipeline (which reads self.F) sees the canonical layout.  The
        # per-GPU tiles are freed immediately after gather; the dense LU
        # below allocates its own multi-GPU partition through
        # ``lu_factor_dispatch`` + ``factor_multi_gpu``.
        self.F = dm_F.to_host()
        dm_F.free()
        if _CUPY_OK_STAT:
            _cp_stat.cuda.runtime.deviceSynchronize()
            _cp_stat.get_default_memory_pool().free_all_blocks()
            _cp_stat.get_default_pinned_memory_pool().free_all_blocks()
        _gpu_pool_cleanup_stat()

    def _init_matrices(self,
            enei: float) -> 'BEMStatIter':

        # MATLAB: bemstatiter/private/initmat.m
        if self.enei is not None and self.enei == enei:
            return self

        # v1.8 (VRAM-Schur): when both distributed-build AND Schur are
        # active we take a dedicated path that assembles the reduced
        # (n_core, n_core) BEM matrix directly distributed and uses it
        # as the GMRES preconditioner.  The host-side full LU build
        # below is bypassed; the SchurIterOperator is wired up later in
        # the same method so its presence still triggers the reduced
        # GMRES on solve().
        if (_vram_share_active_stat() and self._schur_opt
                and not self._hmatrix):
            try:
                return self._init_distributed_schur(enei)
            except Exception as _exc:
                print('[info] BEMStatIter._init_distributed_schur failed '
                        '({}), falling back to host build.'.format(_exc),
                        flush = True)

        # v1.7.2 wavelength-entry GPU cleanup: drain any stale residents from
        # the previous wavelength BEFORE we re-factor (-Lambda - F) so the
        # new dense LU sees the maximum amount of free device memory.
        # MATLAB-parity (MATLAB releases its workspace at the end of every
        # wavelength loop iteration; cupy needs an explicit drain).
        #
        # Pattern mirrors bem_ret.py:485-503: deviceSynchronize() then both
        # default + pinned pool free_all_blocks().  The full cleanup (sync +
        # both pools + optional limit) is the helper; the per-step inline
        # drains below are cheaper single-pool free_all_blocks() calls used
        # after individual GEMM / LU stages where the prior op has already
        # had its result captured.
        if self._mat_lu is not None:
            self._mat_lu = None
        _gpu_pool_cleanup_stat(apply_limit = True)
        # Belt-and-braces inline drain in case the helper short-circuited
        # (cupy not importable on this build).  Guarded so CPU path is
        # untouched.
        if _CUPY_OK_STAT:
            _cp_stat.get_default_memory_pool().free_all_blocks()

        self.enei = enei

        # Dielectric functions
        eps1 = self.p.eps1(enei)
        eps2 = self.p.eps2(enei)

        # Lambda function [Garcia de Abajo, Eq. (23)]
        # MATLAB: obj.lambda = 2 * pi * (eps1 + eps2) ./ (eps1 - eps2)
        self._lambda = 2 * np.pi * (eps1 + eps2) / (eps1 - eps2)

        # Initialize preconditioner
        if self.precond is not None:
            F = self.F
            # Densify if HMatrix — preconditioner LU is dense.
            if hasattr(F, 'full') and not isinstance(F, np.ndarray):
                F_dense = F.full()
            else:
                F_dense = F
            n = F_dense.shape[0]
            # v1.7.2: densification of F (when F is an HMatrix) can leave
            # the per-block GPU staging buffers in the pool.  Drain so the
            # subsequent Lambda allocation does not push the pool past the
            # 49 GB cap on a 49 GB A6000.
            if _CUPY_OK_STAT:
                _cp_stat.get_default_memory_pool().free_all_blocks()

            # Build diagonal Lambda matrix from lambda values
            # MATLAB: spdiag(obj.lambda) handles both scalar and array
            if np.isscalar(self._lambda) or (isinstance(self._lambda, np.ndarray) and self._lambda.ndim == 0):
                Lambda = self._lambda * np.eye(n)
            else:
                Lambda = np.diag(self._lambda)

            # v1.7.3 Phase 3 (B-3): when ``MNPBEM_VRAM_SHARE=1`` AND
            # ``MNPBEM_VRAM_SHARE_GPUS>=2``, ``_vram_share_lu_kwargs_stat()``
            # returns ``{'n_gpus': N, 'backend': '<backend>'}`` and
            # ``lu_factor_dispatch`` routes the dense LU through
            # ``factor_multi_gpu`` (cuSolverMg by default).  The (-Lambda - F)
            # matrix is partitioned column-block round-robin across N
            # devices.  When the env vars are not set the kwargs dict is
            # empty and the call is bit-identical to the single-GPU path.
            _lu_kwargs = _vram_share_lu_kwargs_stat()

            if self.precond == 'hmat':
                # MATLAB: obj.mat = lu(-lambda - F)
                self._mat_lu = lu_factor_dispatch(-Lambda - F_dense,
                        **_lu_kwargs)

            elif self.precond == 'full':
                # MATLAB: obj.mat = inv(-lambda - full(F))
                self._mat_lu = lu_factor_dispatch(-Lambda - F_dense,
                        **_lu_kwargs)

            else:
                raise ValueError('[error] preconditioner not known: <{}>'.format(self.precond))

            # v1.7.2: ``F_dense`` / ``Lambda`` / their sum are transient
            # buffers consumed by ``lu_factor_dispatch`` (overwrite_a paths)
            # but the Python frame still references them.  Drop the names
            # explicitly so cupy can reclaim before the next wavelength's
            # LU runs.
            del F_dense, Lambda
            if _CUPY_OK_STAT:
                _cp_stat.cuda.runtime.deviceSynchronize()
                _cp_stat.get_default_memory_pool().free_all_blocks()
            _gpu_pool_cleanup_stat()

        # Schur (v1.5.0): detect cover-layer partition and prepare the
        # SchurIterOperator that wraps _afun. Done lazily here so that
        # the partition is recomputed if the user constructs the solver
        # without enei and queries it later. When no EpsNonlocal cover
        # layer is present, schur silently falls back to the full path.
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
                self._schur_op = SchurIterOperator(
                        self._afun,
                        shell_idx,
                        core_idx,
                        nfaces = nfaces,
                        components = 1,
                        dtype = complex,
                        g_ss_solver = self._schur_g_ss_solver,
                        inner_tol = self._schur_inner_tol,
                        inner_maxit = self._schur_inner_maxit)
                self._schur_active = True

        return self

    def _init_distributed_schur(self,
            enei: float) -> 'BEMStatIter':
        """v1.8 VRAM-Schur: distributed Schur reduction for the iter path.

        Builds the reduced ``M_eff = M_cc - M_cs @ inv(M_ss) @ M_sc``
        directly distributed across N GPUs via
        ``schur_eliminate_distributed`` (see ``schur_helpers.py``).  The
        reduced LU acts as the GMRES preconditioner; the
        :class:`SchurIterOperator` wraps ``_afun`` as before so the
        GMRES matvec stays on the full (-Lambda - F) operator (this
        keeps Schur reduction's algebraic equivalence intact while
        letting the preconditioner exploit the reduced-block structure).

        The full preconditioner LU (size ``(N, N)``) that the legacy
        path stores in ``self._mat_lu`` is bypassed entirely — only the
        reduced ``(n_core, n_core)`` LU is allocated, partitioned across
        the N participating GPUs.

        Bit-identity: the algebraic GMRES residual is unchanged (same
        matvec); only the preconditioner approximation changes, which
        affects iteration count not the final solution to ``rtol``.
        """

        # Reuse cached LU when the wavelength has not changed.  Mirrors
        # the guard at the top of ``_init_matrices``.
        if (self.enei is not None and self.enei == enei
                and self._schur_active and self._mat_lu is not None):
            return self

        import gc as _gc
        from .schur_iter_helpers import (
                SchurIterOperator,
                detect_iter_partition,
        )
        from .schur_helpers import schur_eliminate_distributed

        partition = detect_iter_partition(self.p)
        if partition is None:
            # No cover layer — defer to the standard init path.  Temporarily
            # disable ``_schur_opt`` so the top-of-``_init_matrices`` guard
            # does not recurse into this method.  The non-Schur branch
            # below still allows ``schur=True`` callers to silently fall
            # back when no EpsNonlocal layer is present (matches the
            # behaviour of the host-side ``_init_matrices`` Schur block).
            self._schur_active = False
            self._schur_op = None
            self.enei = None
            saved_schur_opt = self._schur_opt
            self._schur_opt = False
            try:
                return self._init_matrices(enei)
            finally:
                self._schur_opt = saved_schur_opt

        shell_idx, core_idx = partition
        dist_kw = _vram_share_distributed_kwargs_stat()
        n_gpus = int(dist_kw['n_gpus'])
        device_ids = dist_kw['device_ids']
        block_size = int(dist_kw['block_size'])
        if device_ids is None:
            device_ids = list(range(n_gpus))

        # ---- Per-wavelength cleanup: close stale LU + free old tiles ----
        old = self._mat_lu
        if (isinstance(old, tuple) and len(old) == 3
                and old[0] == 'mgpu' and old[1] is not None):
            handle = old[1]
            try:
                handle.close()
            except Exception:
                pass
            dm_old = getattr(handle, '_distmat_keepalive', None)
            if dm_old is not None:
                try:
                    dm_old.free()
                except Exception:
                    pass
                try:
                    handle._distmat_keepalive = None
                except Exception:
                    pass
        self._mat_lu = None
        self._schur_reduce_keepalive = None
        _gc.collect()
        _gpu_pool_cleanup_stat(apply_limit = True)
        if _CUPY_OK_STAT:
            _cp_stat.get_default_memory_pool().free_all_blocks()

        self.enei = enei

        # ---- Dielectric / Lambda ---------------------------------------
        eps1 = self.p.eps1(enei)
        eps2 = self.p.eps2(enei)
        self._lambda = 2 * np.pi * (eps1 + eps2) / (eps1 - eps2)

        # ---- M_eval (full column slice) -------------------------------
        N = int(self.F.shape[0])
        F_host = np.asarray(self.F)
        # Promote lambda to (N,) array for diagonal addition.
        if (np.isscalar(self._lambda)
                or (isinstance(self._lambda, np.ndarray)
                        and self._lambda.ndim == 0)):
            lambda_diag = np.full(
                    N, complex(self._lambda), dtype = np.complex128)
        else:
            lambda_diag = np.asarray(self._lambda, dtype = np.complex128)

        def _M_eval(c0: int, c1: int) -> np.ndarray:
            # Build the legacy preconditioner form -(diag(Lambda) + F) so
            # the reduced LU mirrors what _mfun would solve on the full
            # path.
            block = F_host[:, c0:c1].astype(np.complex128, copy = True)
            ncol = c1 - c0
            for k in range(ncol):
                j = c0 + k
                block[j, k] += lambda_diag[j]
            return -block

        # ---- Build reduced distributed matrix + reduce/recover --------
        M_eff_dm, reduce_rhs, recover, keepalive = schur_eliminate_distributed(
                M_eval = _M_eval,
                N_full = N,
                shell_indices = shell_idx,
                core_indices = core_idx,
                n_gpus = n_gpus,
                device_ids = device_ids,
                block_size = block_size)

        # ---- Factor reduced matrix in place ---------------------------
        M_mglu = M_eff_dm.lu_factor(backend = 'cusolvermg')
        M_mglu._distmat_keepalive = M_eff_dm  # type: ignore[attr-defined]
        self._mat_lu = ('mgpu', M_mglu, None)
        self._schur_reduce_keepalive = keepalive

        # ---- SchurIterOperator (wraps _afun = full M matvec) -----------
        # The GMRES iteration runs on the reduced operator M_eff.  Each
        # matvec costs 2 full-mesh M @ v matvecs + 1 small M_ss^-1 apply.
        nfaces = self.p.n if hasattr(self.p, 'n') else self.p.nfaces
        self._shell_face_idx = shell_idx
        self._core_face_idx = core_idx
        # Reuse the cached host M_ss LU from ``schur_eliminate_distributed``
        # to back the SchurIterOperator's A_ss^{-1} (callable mode).  This
        # makes the operator's reduce_rhs / recover_full numerically
        # identical to the closures we just stored, and avoids re-probing
        # the shell block in lu_dense mode (which would otherwise burn
        # n_shell extra full matvecs per wavelength).
        lu_ss, piv_ss = keepalive['M_ss_lu']

        def _user_g_ss_solver(rhs):
            from scipy.linalg import lu_solve as _lu_solve_host
            if rhs.ndim == 1:
                return _lu_solve_host(
                        (lu_ss, piv_ss), rhs, check_finite = False)
            return _lu_solve_host(
                    (lu_ss, piv_ss),
                    rhs.reshape(rhs.shape[0], -1),
                    check_finite = False).reshape(rhs.shape)

        self._schur_op = SchurIterOperator(
                self._afun,
                shell_idx,
                core_idx,
                nfaces = nfaces,
                components = 1,
                dtype = complex,
                g_ss_solver = 'callable',
                user_g_ss_solver = _user_g_ss_solver,
                inner_tol = self._schur_inner_tol,
                inner_maxit = self._schur_inner_maxit)
        # Also store the host reduce/recover for diagnostics / parity checks.
        self._schur_dist_reduce_rhs = reduce_rhs
        self._schur_dist_recover = recover
        self._schur_active = True

        # Final sync + pool compaction across all participating devices.
        if _CUPY_OK_STAT:
            try:
                for d in device_ids:
                    _cp_stat.cuda.runtime.setDevice(d)
                    _cp_stat.cuda.runtime.deviceSynchronize()
                    _cp_stat.get_default_memory_pool().free_all_blocks()
                _cp_stat.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

        return self

    def _afun(self,
            vec: np.ndarray) -> np.ndarray:

        # MATLAB: bemstatiter/private/afun.m
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces
        vec_2d = vec.reshape(n, -1)

        # -(lambda + F) * vec
        # Handle both scalar and array lambda
        if np.isscalar(self._lambda) or (isinstance(self._lambda, np.ndarray) and self._lambda.ndim == 0):
            result = -(self.F @ vec_2d + vec_2d * self._lambda)
        else:
            result = -(self.F @ vec_2d + vec_2d * self._lambda[:, np.newaxis])
        return result.reshape(-1)

    def _mfun(self,
            vec: np.ndarray) -> np.ndarray:

        # MATLAB: bemstatiter/private/mfun.m
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces
        vec_2d = vec.reshape(n, -1)

        if self.precond == 'hmat' or self.precond == 'full':
            # MATLAB: vec = solve(obj.mat, vec) or obj.mat * vec
            result = lu_solve_dispatch(self._mat_lu, vec_2d)
        else:
            result = vec_2d

        return result.reshape(-1)

    def solve(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMStatIter']:

        # MATLAB: bemstatiter/solve.m
        # Initialize BEM solver (if needed)
        self._init_matrices(exc.enei)

        # Excitation and size of excitation array
        b = exc.phip.ravel().astype(complex)
        siz = exc.phip.shape

        # v1.7.2: drain the cupy pool right before GMRES enters its Krylov
        # build-up.  The init pipeline above leaves up to 2 N^2 * 16 B of
        # transient asarray buffers in the pool; releasing them now keeps
        # peak usage during the iter loop bounded to LU + Krylov subspace.
        if _CUPY_OK_STAT:
            _cp_stat.cuda.runtime.deviceSynchronize()
            _cp_stat.get_default_memory_pool().free_all_blocks()
        _gpu_pool_cleanup_stat()

        if self._schur_active:
            # Schur path: GMRES is run on the reduced (core-only) operator.
            # v1.8 (VRAM-Schur): when ``self._mat_lu`` carries a reduced LU
            # (the ('mgpu', ...) tag set by ``_init_distributed_schur``)
            # we hand it to GMRES as the preconditioner so the
            # iteration count benefits from the distributed multi-GPU
            # LU.  Otherwise (v1.5.0 legacy path) the preconditioner is
            # bypassed because ``_mfun`` was built for the full (N, N)
            # (-Lambda - F) factor and would need re-factoring on the
            # core block.
            from .schur_iter_helpers import make_distributed_schur_preconditioner
            op = self._schur_op
            b_eff = op.reduce_rhs(b)
            fm_schur = make_distributed_schur_preconditioner(self._mat_lu)
            x_core, _ = self._iter_solve(None, b_eff, op._matvec, fm_schur)
            x = op.recover_full(x_core, b)
            # v1.7.2: drop the Schur-reduced Krylov subspace handle and
            # the matvec closure references before the next wavelength
            # entry drains the pool.
            del x_core
            if _CUPY_OK_STAT:
                _cp_stat.cuda.runtime.deviceSynchronize()
                _cp_stat.get_default_memory_pool().free_all_blocks()
            _gpu_pool_cleanup_stat()
        else:
            # Function for matrix multiplication
            fa = self._afun
            fm = None
            if self.precond is not None:
                fm = self._mfun

            # v1.5.0 H-matrix LU preconditioner (agent alpha). Replaces fm
            # when active on the H-matrix path.
            if self._hmatrix and self._hlu_mode != 'none':
                fm = self._build_hlu_preconditioner(b.shape[0])

            # Iterative solution
            x, self_updated = self._iter_solve(None, b, fa, fm)
            # v1.7.2: GMRES holds up to ``restart`` Krylov vectors of length
            # n in its scipy internal buffer; ``fa`` / ``fm`` close over
            # ``self.F`` and ``self._mat_lu`` which may be cupy resident.
            # Once x is computed the matvec closures are dead refs;
            # drop them so the pool can compact before the next solve.
            del fa, fm
            if _CUPY_OK_STAT:
                _cp_stat.cuda.runtime.deviceSynchronize()
                _cp_stat.get_default_memory_pool().free_all_blocks()
            _gpu_pool_cleanup_stat()

        # Host-materialize cupy result so the returned sig is always
        # CPU-resident (mirrors BEMStat.solve defensive guard).
        sig_arr = x.reshape(siz)
        if is_cupy_array(sig_arr):
            sig_arr = to_host(sig_arr)

        # Save everything in single structure
        sig = CompStruct(self.p, exc.enei, sig = sig_arr)

        # v1.7.2 solve-exit cleanup: free any residual transient buffers
        # (RHS reshape staging, matvec scratch) before returning so the
        # caller's wavelength loop sees a fully drained pool when it
        # advances to the next enei.
        if _CUPY_OK_STAT:
            _cp_stat.get_default_memory_pool().free_all_blocks()
        _gpu_pool_cleanup_stat()

        return sig, self

    def _build_hlu_preconditioner(self,
            n_vec: int) -> Callable:

        # v1.5.0 agent alpha — H-matrix LU preconditioner for BEMStatIter.
        # The quasistatic operator A = -(lambda*I + F) has an exact dense
        # LU built inside _init_matrices when self.precond is set. We turn
        # precond='hmat' on for this solve so that the existing _mfun acts
        # as the GMRES preconditioner. This is equivalent to v1.3
        # ``precond='hmat'`` but now triggered on the H-matrix code path
        # (where v1.3 left it disabled by default).
        if self._hlu_object is not None and self._hlu_object == (n_vec, self.enei):
            return self._mfun

        if self._mat_lu is None:
            # Build the dense LU once (lambda + F densified to ndarray).
            self.precond = 'hmat'
            cached_enei = self.enei
            self.enei = None
            self._init_matrices(cached_enei)
            # v1.7.2: the densify-and-LU pipeline above can spike GPU
            # usage by ~3 N^2 transient; drain so GMRES sees max headroom.
            if _CUPY_OK_STAT:
                _cp_stat.cuda.runtime.deviceSynchronize()
                _cp_stat.get_default_memory_pool().free_all_blocks()

        self._hlu_object = (n_vec, self.enei)
        return self._mfun

    def __truediv__(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMStatIter']:

        # MATLAB: bemstatiter/mldivide.m
        return self.solve(exc)

    def __mul__(self,
            sig: CompStruct) -> CompStruct:

        # MATLAB: bemstatiter/mtimes.m
        pot1 = self.potential(sig, 1)
        pot2 = self.potential(sig, 2)

        phi = CompStruct(self.p, sig.enei,
            phi1 = pot1.phi1, phi1p = pot1.phi1p,
            phi2 = pot2.phi2, phi2p = pot2.phi2p)
        return phi

    def field(self,
            sig: CompStruct,
            inout: int = 2) -> CompStruct:

        # MATLAB: bemstatiter/field.m
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces
        nvec = self.p.nvec

        # Electric field in normal direction
        if inout == 1:
            H = self._g.H1
        else:
            H = self._g.H2

        # MATLAB: e = -outer(obj.p.nvec, matmul(obj.g.H, sig.sig))
        H_sig = H @ sig.sig.reshape(n, -1)
        if H_sig.ndim == 1:
            e = -nvec * H_sig[:, np.newaxis]
        else:
            e = -nvec[:, :, np.newaxis] * H_sig[:, np.newaxis, :]
        # v1.7.2: drain after the field-side matvec to keep the pool from
        # accumulating across repeated field() calls in a wavelength loop.
        if _CUPY_OK_STAT:
            _cp_stat.get_default_memory_pool().free_all_blocks()

        # Tangential directions via interpolation
        G_sig = self._g.G @ sig.sig.reshape(n, -1)
        phi = self.p.interp(G_sig)
        phi1, phi2, t1, t2 = self.p.deriv(phi)

        # Normal vector
        nvec_c = np.cross(t1, t2)
        h = msqrt(np.sum(nvec_c * nvec_c, axis = 1, keepdims = True))
        nvec_c = nvec_c / h

        # Tangential derivative of PHI
        tvec1 = np.cross(t2, nvec_c) / h
        tvec2 = np.cross(t1, nvec_c) / h

        if phi1.ndim == 1:
            phip = tvec1 * phi1[:, np.newaxis] - tvec2 * phi2[:, np.newaxis]
        else:
            phip = tvec1[:, :, np.newaxis] * phi1[:, np.newaxis, :] - \
                   tvec2[:, :, np.newaxis] * phi2[:, np.newaxis, :]

        e = e - phip
        # v1.7.2: drain residual transient buffers from the interp/deriv
        # pipeline before returning so a follow-up potential()/field() at
        # a different wavelength does not see leftover blocks.
        if _CUPY_OK_STAT:
            _cp_stat.get_default_memory_pool().free_all_blocks()

        return CompStruct(self.p, sig.enei, e = e)

    def potential(self,
            sig: CompStruct,
            inout: int = 2) -> CompStruct:

        # MATLAB: bemstatiter/potential.m
        pot = self._g.potential(sig, inout)
        # v1.7.2: drain after the Green-function potential evaluation so
        # repeated potential() calls in a wavelength loop don't leak
        # transient asarray buffers into the cupy pool.
        if _CUPY_OK_STAT:
            _cp_stat.get_default_memory_pool().free_all_blocks()
        return pot

    def clear(self) -> 'BEMStatIter':

        # MATLAB: bemstatiter/clear.m
        # v1.7 A3 fix: also reset the cache gate (enei) and the wavelength-
        # dependent auxiliaries (_lambda, Schur state, _hlu_object).
        # Otherwise a follow-up solve at the same wavelength hits the
        # cache, finds _mat_lu=None, and crashes inside _mfun.
        #
        # v1.8 (VRAM-Schur): if _mat_lu carries an 'mgpu' tag, close the
        # distributed LU handle and free its tiles before nulling out
        # the reference.  Mirrors BEMStat.clear's distributed-LU cleanup.
        old = self._mat_lu
        if (isinstance(old, tuple) and len(old) == 3
                and old[0] == 'mgpu' and old[1] is not None):
            handle = old[1]
            try:
                handle.close()
            except Exception:
                pass
            dm_old = getattr(handle, '_distmat_keepalive', None)
            if dm_old is not None:
                try:
                    dm_old.free()
                except Exception:
                    pass
                try:
                    handle._distmat_keepalive = None
                except Exception:
                    pass
        self._mat_lu = None
        self.enei = None
        self._lambda = None
        self._schur_active = False
        self._schur_op = None
        self._hlu_object = None
        # v1.8 (VRAM-Schur): drop the distributed-Schur keepalive that
        # holds host M_ss LU + M_sc + M_cs + D_inv_C.  Safe no-op on the
        # legacy path (attribute may not exist).
        if hasattr(self, '_schur_reduce_keepalive'):
            self._schur_reduce_keepalive = None
        if hasattr(self, '_schur_dist_reduce_rhs'):
            self._schur_dist_reduce_rhs = None
        if hasattr(self, '_schur_dist_recover'):
            self._schur_dist_recover = None
        # v1.7.2: explicit clear() means the user wants the device drained.
        # Without this the LU buffer just released above stays in the cupy
        # pool until the next solve triggers a free_all_blocks.
        if _CUPY_OK_STAT:
            _cp_stat.cuda.runtime.deviceSynchronize()
            _cp_stat.get_default_memory_pool().free_all_blocks()
            _cp_stat.get_default_pinned_memory_pool().free_all_blocks()
        _gpu_pool_cleanup_stat()
        return self

    def __call__(self,
            enei: float) -> 'BEMStatIter':

        # v1.7.2: explicit __call__(enei) is a user-driven wavelength step.
        # ``_init_matrices`` already drains on cache miss; we make the
        # drain unconditional here so consecutive __call__(enei) /
        # __call__(enei') sequences keep the pool tight even when only
        # the H-matrix evaluator side allocates transient buffers.
        out = self._init_matrices(enei)
        if _CUPY_OK_STAT:
            _cp_stat.get_default_memory_pool().free_all_blocks()
        return out

    def __repr__(self) -> str:
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces if hasattr(self.p, 'nfaces') else '?'
        status = 'enei={:.1f}nm'.format(self.enei) if self.enei is not None else 'not initialized'
        return 'BEMStatIter(p: {} faces, solver={}, {})'.format(n, self.solver, status)
