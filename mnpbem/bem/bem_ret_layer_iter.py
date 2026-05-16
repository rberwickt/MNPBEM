import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator

from ..greenfun import CompStruct
from ..utils.gpu import (
    lu_factor_dispatch, lu_solve_dispatch, matmul_dispatch,
    to_host, is_cupy_array,
)
from ..utils.matlab_compat import msqrt
from .bem_iter import BEMIter

# v1.7.2 GPU memory-pool cleanup: mirror BEMRet's wavelength-end immediate
# free pattern so cupy returns blocks to the driver at every wavelength
# rather than accumulating across the sweep (MATLAB-parity).  The layered
# iter path holds substantially more dense state on device than the dense
# BEMRet (G1, H1 plus the five-block G2/H2 structure for substrate +
# the four LU factors built in _init_precond), so disciplined per-step
# drains are even more important here.
try:
    import cupy as _cp_layer  # type: ignore
    _CUPY_OK_LAYER = True
except Exception:
    _cp_layer = None  # type: ignore
    _CUPY_OK_LAYER = False


def _vram_share_lu_kwargs() -> dict:
    """Read MNPBEM_VRAM_SHARE_* env vars and return kwargs for lu_factor_dispatch.

    Mirrors ``bem_ret.py``'s helper of the same name (line 43-54).  Returns
    an empty dict when VRAM-share is disabled (``MNPBEM_VRAM_SHARE!=1`` or
    ``MNPBEM_VRAM_SHARE_GPUS<=1``) so the dispatch call is bit-identical to
    the single-GPU path.  When VRAM-share is enabled the returned kwargs
    route each of the four precond LU factors (G1_lu, G2p_lu, Gamma_lu,
    m11_lu, schur_lu) through cuSolverMg so the dense LU is partitioned
    across the worker's GPUs.  Required for the 15072-face Au@Ag dimer +
    substrate sweep where the layered preconditioner state alone
    (4 N x N complex128 LU factors + the block-2x2 Schur) exceeds the
    49 GB single-A6000 cap.
    """
    if os.environ.get('MNPBEM_VRAM_SHARE', '0') != '1':
        return {}
    n_gpus = int(os.environ.get('MNPBEM_VRAM_SHARE_GPUS', '1'))
    if n_gpus <= 1:
        return {}
    backend = os.environ.get('MNPBEM_VRAM_SHARE_BACKEND', 'cusolvermg')
    return {'n_gpus': n_gpus, 'backend': backend}


def _gpu_pool_cleanup_layer(apply_limit: bool = False) -> None:
    """Synchronise CUDA stream then drain cupy default + pinned memory pools.

    Mirrors the v1.7.2 BEMRet helper (bem_ret.py:485-503, 734-737).  The
    deviceSynchronize() BEFORE free_all_blocks() is load-bearing: blocks
    still in flight on the CUDA stream are not idle yet, so a free_all_blocks
    that races ahead of the stream returns nothing.

    Honours MNPBEM_GPU_POOL_LIMIT_GB (legitimate peaks past this cap will
    OOM; default 0 = uncapped).
    """
    if not _CUPY_OK_LAYER:
        return
    try:
        mempool = _cp_layer.get_default_memory_pool()
        pinned = _cp_layer.get_default_pinned_memory_pool()
        if apply_limit:
            try:
                pool_limit_gb = float(os.environ.get(
                        'MNPBEM_GPU_POOL_LIMIT_GB', '0'))
            except (TypeError, ValueError):
                pool_limit_gb = 0.0
            if pool_limit_gb > 0:
                mempool.set_limit(size = int(pool_limit_gb * (1024 ** 3)))
        _cp_layer.cuda.runtime.deviceSynchronize()
        mempool.free_all_blocks()
        pinned.free_all_blocks()
    except Exception:
        pass


def _coerce_host(val: Any) -> Any:
    # v1.7 A2 fix: when MNPBEM_GPU=1 the underlying CompGreenRet.eval
    # returns cupy ndarrays for inner-inner / inner-outer Green blocks
    # (the substrate-modified outer pair already goes through
    # _LayerGreen._assembly which v1.6.5 normalised to host).  The dense
    # iter path mixes cupy G/H with host vectors from scipy GMRES, so we
    # promote every dense ndarray down to host before saving.  Nested
    # dict / _LayerGreen structures keep their key/attribute layout.
    if is_cupy_array(val):
        return to_host(val)
    if isinstance(val, dict):
        return {k: _coerce_host(v) for k, v in val.items()}
    if hasattr(val, 'ss'):
        for attr in ('ss', 'hh', 'p', 'sh', 'hs'):
            if hasattr(val, attr):
                setattr(val, attr, _coerce_host(getattr(val, attr)))
        return val
    return val


# ---------------------------------------------------------------------------
# B-3 distributed-build helpers (v1.7.3 Phase 3)
# ---------------------------------------------------------------------------

def _vram_share_active() -> bool:
    """Return True when distributed-build is enabled.

    Mirrors BEMRetIter helper; activated by
    ``MNPBEM_VRAM_SHARE_DISTRIBUTED=1`` plus the usual VRAM-share env
    config (``MNPBEM_VRAM_SHARE=1`` + ``MNPBEM_VRAM_SHARE_GPUS>=2``)
    and cupy/cuSolverMg availability.

    For BEMRetLayerIter the win is larger than the dense BEMRet path:
    the layered Green's function exposes a 5-block ``(ss, hh, p, sh, hs)``
    decomposition for the substrate-modified G2/H2 pair, so the
    per-wavelength full assembly footprint is ~5x the equivalent
    non-substrate run.  Each of the inner pairs (G1, H1) is a single
    matrix and benefits directly; the structured outer pair stays on
    the legacy path (the substrate-tabulated assembly inside
    _LayerGreen does its own per-block work that does not yet have an
    eval_block analogue).
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
    if not _CUPY_OK_LAYER:
        return False
    try:
        from ..utils.multi_gpu_lu import cusolvermg_available
        return bool(cusolvermg_available())
    except Exception:
        return False


def _vram_share_env_config() -> Tuple[int, str, Optional[List[int]]]:
    """Resolve (n_gpus, backend, device_ids) for the distributed path."""
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


def _distributed_block_assemble_layer(green: Any,
        enei: float,
        ncomb_list: List[Tuple[int, int, str, int]],
        nrows: int,
        ncols: int,
        n_gpus: int,
        device_ids: Optional[List[int]]) -> Any:
    """Same as BEMRetIter's ``_distributed_block_assemble`` for the layered
    Green function.  We keep a separate helper here so the layered path
    can grow extra parameters (e.g. block-key forwarding for the (ss, hh,
    p, sh, hs) decomposition) without touching the BEMRetIter helper.
    """
    from ..utils.distributed_matrix import DistributedMatrix
    import numpy as _np_local

    def _evalfn(gpu_idx: int, c0: int, c1: int) -> Any:
        out = None
        for (i, j, key, sign) in ncomb_list:
            blk = green.eval_block(i, j, key, enei, c0, c1)
            if out is None:
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


class BEMRetLayerIter(BEMIter):

    # MATLAB: @bemretlayeriter properties (Constant)
    name = 'bemsolver'
    needs = {'sim': 'ret'}

    def __init__(self,
            p: Any,
            layer: Optional[Any] = None,
            enei: Optional[float] = None,
            **options: Any) -> None:

        # H-matrix + cover-layer combination is not supported in v1.3.0.
        # The layer Green function has a structured (ss/hh/p/sh/hs) block
        # form that does not yet have an ACA wrapper, and combining it
        # with the dense-LU Schur in _init_precond is M5+ work. Surface
        # an explicit error so users with layered geometries fall back
        # to the dense path consciously.
        if options.pop('hmatrix', False):
            raise NotImplementedError(
                '[error] BEMRetLayerIter does not support <hmatrix> in v1.3.0; '
                'use the dense BEMRetLayerIter path or strip the cover layer.')

        # Strip H-matrix-only kwargs so the dense path does not choke on
        # unknown options.
        for k in ('htol', 'kmax', 'cleaf', 'fadmiss', 'eta'):
            options.pop(k, None)

        # Initialize BEMIter base class
        super(BEMRetLayerIter, self).__init__(**options)

        # MATLAB: @bemretlayeriter properties
        self.p = p
        self.layer = layer if layer is not None else options.get('layer', None)
        self.enei = None
        self.g = None

        # MATLAB: @bemretlayeriter properties (Access = private)
        self._op = options
        self._sav = None
        self._k = None
        self._eps1 = None
        self._eps2 = None
        self._nvec = p.nvec
        self._G1 = None
        self._H1 = None
        self._G2 = None  # structured: ss, hh, p, sh, hs components
        self._H2 = None  # structured: ss, hh, p, sh, hs components

        # Green function (with layer structure)
        # MATLAB: obj.g = aca.compgreenretlayer(p, varargin{:}, ...)
        self._init_green(p, **options)

        # Initialize for given wavelength
        if enei is not None:
            self._init_matrices(enei)

    def _init_green(self,
            p: Any,
            **options: Any) -> None:

        # MATLAB: bemretlayeriter/private/init.m
        from ..greenfun import CompGreenRetLayer
        self.g = CompGreenRetLayer(p, p, self.layer, **options)

    def _init_matrices(self,
            enei: float) -> 'BEMRetLayerIter':

        # MATLAB: bemretlayeriter/private/initmat.m
        # Waxenegger et al., Comp. Phys. Commun. 193, 128 (2015)
        if self.enei is not None and self.enei == enei:
            return self

        # v1.7.2 wavelength-entry GPU cleanup.  Drop the previous wavelength's
        # G1/H1/G2/H2 + preconditioner state and drain the cupy pool BEFORE
        # the new Green-function evaluation re-uploads ~5 N^2 of fresh GPU
        # buffers.  Without this drain, the substrate path on dense
        # composite particles (e.g. Au@Ag dimer on glass) accumulates
        # ~2.5 GB / wl of stale buffers and OOMs around wl ~12-15 on a
        # 49 GB A6000.
        for _attr in ('_G1', '_H1', '_G2', '_H2', '_sav'):
            if hasattr(self, _attr):
                setattr(self, _attr, None)
        import gc as _gc
        _gc.collect()
        _gpu_pool_cleanup_layer(apply_limit = True)
        # Belt-and-braces inline drain (matches bem_ret.py:501-503 pattern)
        # in case the helper short-circuited because cupy isn't importable
        # on this build.  Guarded so CPU path is bit-identical.
        if _CUPY_OK_LAYER:
            _cp_layer.cuda.runtime.deviceSynchronize()
            _cp_layer.get_default_memory_pool().free_all_blocks()
            _cp_layer.get_default_pinned_memory_pool().free_all_blocks()

        self.enei = enei

        # Wavenumber
        self._k = 2 * np.pi / enei

        # Dielectric function
        self._eps1 = self.p.eps1(enei)
        self._eps2 = self.p.eps2(enei)

        # v1.7.3 Phase 3 (B-3) — distributed inner-pair assembly hook.
        # Builds the unsubstrated inner-pair Green-function differences
        # (G1 = G11 - G21, H1 = H11 - H21) across N GPUs.  The outer pair
        # (G2, H2) carries the substrate-modified structured (ss, hh, p,
        # sh, hs) decomposition and stays on the legacy assembly path
        # (the dict-valued output is not column-sliceable through
        # eval_block today; substrate Green tabulation lives in
        # _LayerGreen._assembly).  Activated by
        # ``MNPBEM_VRAM_SHARE_DISTRIBUTED=1``; falls back to legacy on
        # any exception.
        distributed_inner_built = False
        if _vram_share_active():
            try:
                G1, H1 = self._build_distributed_GH_inner(enei)
                distributed_inner_built = True
            except Exception as exc:
                print('[info] BEMRetLayerIter init: distributed inner G/H '
                        'build failed ({}), falling back to legacy '
                        'eval.'.format(exc), flush = True)
                _gpu_pool_cleanup_layer()

        if not distributed_inner_built:
            # Green functions for inner surfaces
            # MATLAB: G1 = g{1,1}.G(enei) - g{2,1}.G(enei)
            # v1.7 A2 fix: coerce every eval() output to host numpy when GPU
            # path returns cupy ndarrays.  The dense iter ``_afun`` /
            # ``_mfun`` mix these arrays with host GMRES vectors so we
            # cannot leave any cupy operands in self._G* / self._H*.
            G11 = _coerce_host(self.g.eval(0, 0, 'G', enei))
            G21 = _coerce_host(self.g.eval(1, 0, 'G', enei))
            G1 = G11 - G21 if not (isinstance(G21, (int, float)) and G21 == 0) else G11
            # v1.7.2: G11/G21 have already been coerced to host; the cupy buffers
            # behind them are released the moment the Python names go out of scope
            # but the pool keeps the blocks until the next allocation triggers
            # a search.  An explicit drain after each Green-function stage keeps
            # the high-water mark stable across the four eval() calls below.
            del G11, G21
            if _CUPY_OK_LAYER:
                _cp_layer.get_default_memory_pool().free_all_blocks()

            H11 = _coerce_host(self.g.eval(0, 0, 'H1', enei))
            H21 = _coerce_host(self.g.eval(1, 0, 'H1', enei))
            H1 = H11 - H21 if not (isinstance(H21, (int, float)) and H21 == 0) else H11
            del H11, H21
            if _CUPY_OK_LAYER:
                _cp_layer.get_default_memory_pool().free_all_blocks()

        # Green functions for outer surfaces (with layer structure)
        # MATLAB: G2 = g{2,2}.G(enei); g2 = g{1,2}.G(enei)
        G22_full = _coerce_host(self.g.eval(1, 1, 'G', enei))
        g2 = _coerce_host(self.g.eval(0, 1, 'G', enei))
        if _CUPY_OK_LAYER:
            _cp_layer.get_default_memory_pool().free_all_blocks()

        H22_full = _coerce_host(self.g.eval(1, 1, 'H2', enei))
        h2 = _coerce_host(self.g.eval(0, 1, 'H2', enei))
        if _CUPY_OK_LAYER:
            _cp_layer.get_default_memory_pool().free_all_blocks()

        # For layer structure, G2 is a dict-like with ss, hh, p, sh, hs components
        # MATLAB: G2.ss = G2.ss - g2; G2.hh = G2.hh - g2; G2.p = G2.p - g2
        if isinstance(G22_full, dict):
            G2 = {}
            H2 = {}
            for key in G22_full:
                if key in ('ss', 'hh', 'p'):
                    g2_val = g2 if not isinstance(g2, dict) else g2.get(key, g2)
                    h2_val = h2 if not isinstance(h2, dict) else h2.get(key, h2)
                    G2[key] = G22_full[key] - g2_val
                    H2[key] = H22_full[key] - h2_val
                else:
                    G2[key] = G22_full[key]
                    H2[key] = H22_full[key]
        elif hasattr(G22_full, 'ss'):
            # Object with attributes
            G2 = _LayerGreen()
            H2 = _LayerGreen()

            g2_mat = g2 if not hasattr(g2, 'ss') else g2
            h2_mat = h2 if not hasattr(h2, 'ss') else h2

            G2.ss = G22_full.ss - (g2_mat if np.isscalar(g2_mat) or isinstance(g2_mat, np.ndarray) else g2_mat.ss)
            G2.hh = G22_full.hh - (g2_mat if np.isscalar(g2_mat) or isinstance(g2_mat, np.ndarray) else g2_mat.hh)
            G2.p = G22_full.p - (g2_mat if np.isscalar(g2_mat) or isinstance(g2_mat, np.ndarray) else g2_mat.p)
            G2.sh = G22_full.sh if hasattr(G22_full, 'sh') else np.zeros_like(G22_full.ss)
            G2.hs = G22_full.hs if hasattr(G22_full, 'hs') else np.zeros_like(G22_full.ss)

            H2.ss = H22_full.ss - (h2_mat if np.isscalar(h2_mat) or isinstance(h2_mat, np.ndarray) else h2_mat.ss)
            H2.hh = H22_full.hh - (h2_mat if np.isscalar(h2_mat) or isinstance(h2_mat, np.ndarray) else h2_mat.hh)
            H2.p = H22_full.p - (h2_mat if np.isscalar(h2_mat) or isinstance(h2_mat, np.ndarray) else h2_mat.p)
            H2.sh = H22_full.sh if hasattr(H22_full, 'sh') else np.zeros_like(H22_full.ss)
            H2.hs = H22_full.hs if hasattr(H22_full, 'hs') else np.zeros_like(H22_full.ss)
        else:
            # Fallback: treat as simple matrix (no layer structure difference)
            G2 = G22_full - g2 if not (isinstance(g2, (int, float)) and g2 == 0) else G22_full
            H2 = H22_full - h2 if not (isinstance(h2, (int, float)) and h2 == 0) else H22_full

        # Save Green functions
        self._G1 = G1
        self._H1 = H1
        self._G2 = G2
        self._H2 = H2

        # v1.7.2: drop the intermediate Green-function locals so the cupy
        # pool can compact between assembly and preconditioner build.  Only
        # the self._G/_H attributes need to stay alive; the transient
        # locals (G22_full, g2, h2, G1/H1/G2/H2 stack copies) hold dangling
        # references until function exit otherwise.  ``del`` on the names
        # directly (rather than via locals()) is the only safe way to drop
        # function-frame bindings in CPython.
        del G1, H1, G2, H2, G22_full, g2, H22_full, h2
        if _CUPY_OK_LAYER:
            _cp_layer.cuda.runtime.deviceSynchronize()
            _cp_layer.get_default_memory_pool().free_all_blocks()
        _gpu_pool_cleanup_layer()

        # Initialize preconditioner
        if self.precond is not None:
            self._init_precond(enei)
            # v1.7.2: the preconditioner builds 4 dense LUs (~4 N^2 each on
            # 12672-face dimers on a substrate) and a block-2x2 Schur
            # complement.  Drain right after init so the GMRES Krylov build
            # below sees the maximum amount of free device memory.
            if _CUPY_OK_LAYER:
                _cp_layer.cuda.runtime.deviceSynchronize()
                _cp_layer.get_default_memory_pool().free_all_blocks()
                _cp_layer.get_default_pinned_memory_pool().free_all_blocks()
            _gpu_pool_cleanup_layer()

        return self

    def _build_distributed_GH_inner(self,
            enei: float) -> Tuple[np.ndarray, np.ndarray]:
        """v1.7.3 Phase 3 (B-3): build inner-pair G1/H1 with distributed assembly.

        For BEMRetLayerIter the substrate-modified outer pair (G2, H2)
        carries the structured (ss, hh, p, sh, hs) decomposition that
        the dict-valued ``_LayerGreen._assembly`` produces; we keep
        that on the legacy host eval path because ``eval_block`` is not
        defined for the structured layer Green function.

        The inner pair (G1 = G11 - G21, H1 = H11 - H21) is plain
        ndarray-valued and can be built across N GPUs via
        :meth:`CompGreenRet.eval_block` — which lives at
        ``self.g.g.eval_block`` because ``CompGreenRetLayer`` wraps a
        ``CompGreenRet`` direct (free-space) Green object as
        ``self.g``.

        Returns ``(G1_host, H1_host)`` so the caller can plug them
        straight into ``self._G1 / self._H1``.
        """
        nfaces = self.p.n if hasattr(self.p, 'n') else self.p.nfaces
        n_gpus, backend, device_ids = _vram_share_env_config()

        # Reach into the direct Green object — ``self.g`` is a
        # ``CompGreenRetLayer`` which composes a ``CompGreenRet``
        # at ``self.g.g`` for the direct (non-reflected) component.
        direct_green = getattr(self.g, 'g', None)
        if direct_green is None or not hasattr(direct_green, 'eval_block'):
            raise RuntimeError(
                    '[error] BEMRetLayerIter distributed build requires '
                    'CompGreenRetLayer.g.eval_block (got {})'.format(
                            type(direct_green)))

        print('[info] BEMRetLayerIter init: distributed inner-pair G1/H1 '
                'build (N={}, n_gpus={}, backend={}).'.format(
                        nfaces, n_gpus, backend),
                flush = True)

        # G1 = G(0, 0) - G(1, 0) on the *direct* Green (inner surface;
        # no substrate-reflected contribution by construction).
        dm_G1 = _distributed_block_assemble_layer(
                direct_green, enei,
                ncomb_list = [(0, 0, 'G', 1), (1, 0, 'G', -1)],
                nrows = nfaces, ncols = nfaces,
                n_gpus = n_gpus, device_ids = device_ids)
        G1_host = dm_G1.to_host()
        dm_G1.free()
        if _CUPY_OK_LAYER:
            _cp_layer.get_default_memory_pool().free_all_blocks()

        dm_H1 = _distributed_block_assemble_layer(
                direct_green, enei,
                ncomb_list = [(0, 0, 'H1', 1), (1, 0, 'H1', -1)],
                nrows = nfaces, ncols = nfaces,
                n_gpus = n_gpus, device_ids = device_ids)
        H1_host = dm_H1.to_host()
        dm_H1.free()
        if _CUPY_OK_LAYER:
            _cp_layer.cuda.runtime.deviceSynchronize()
            _cp_layer.get_default_memory_pool().free_all_blocks()
            _cp_layer.get_default_pinned_memory_pool().free_all_blocks()
        _gpu_pool_cleanup_layer()
        return G1_host, H1_host

    def _compress(self,
            hmat: Any) -> Any:

        # MATLAB: bemretlayeriter/private/compress.m
        # Compress H-matrices for preconditioner by adjusting htol/kmax.
        # For HMatrix objects, set htol to max(op.htol) and kmax to min(op.kmax).
        # For dense numpy arrays, pass through unchanged.
        if hasattr(hmat, 'htol') and hasattr(hmat, 'kmax'):
            htol_val = self._op.get('htol', 1e-6)
            kmax_val = self._op.get('kmax', [4, 100])
            hmat.htol = max(htol_val) if hasattr(htol_val, '__iter__') else htol_val
            hmat.kmax = min(kmax_val) if hasattr(kmax_val, '__iter__') else kmax_val
        return hmat

    def _compress_layer_green(self,
            green: Any) -> Any:

        # Compress all components of a layer Green function structure.
        # MATLAB: for name = fieldnames(obj.G2).'; G2.(name{1}) = compress(obj, obj.G2.(name{1})); end
        if isinstance(green, dict):
            return {k: self._compress(v) for k, v in green.items()}
        elif hasattr(green, 'ss'):
            for attr in ('ss', 'hh', 'p', 'sh', 'hs'):
                if hasattr(green, attr):
                    setattr(green, attr, self._compress(getattr(green, attr)))
            return green
        else:
            return self._compress(green)

    def _init_precond(self,
            enei: float) -> None:

        # MATLAB: bemretlayeriter/private/initprecond.m
        # Waxenegger et al., Comp. Phys. Commun. 193, 128 (2015)
        #
        # v1.6.0 (agent B) — multi-material + substrate fix.  Mirrors the
        # v1.5.1 BEMRetIter operator-form correction but for the layered
        # preconditioner.  Dense ``BEMRetLayer`` (bem_ret_layer.py) builds
        # ``Sigma1e = H1·eps1·G1⁻¹`` (NOT ``eps1·H1·G1⁻¹ = eps1·Sigma1``);
        # algebraically the two agree only when eps1 is uniform.  For
        # non-uniform ``eps1`` the legacy form was the source of the
        # ``BEMRetIter`` 70 % drift — same mechanism applies here.
        #
        # Likewise ``L1 = G1·eps1·G1⁻¹`` replaces the bare ``eps1`` factor
        # used in ``_mfun`` for ``alpha`` and ``De`` corrections.  When
        # eps is uniform ``L1`` collapses to a scalar and we keep the
        # legacy fast path bit-identical.
        k = 2 * np.pi / enei
        eps1 = self._eps1
        eps2 = self._eps2
        nvec = self._nvec

        # Dielectric as diagonal
        eps1_uniform = (np.isscalar(eps1) or (isinstance(eps1, np.ndarray)
                and eps1.ndim == 0))
        eps2_uniform = (np.isscalar(eps2) or (isinstance(eps2, np.ndarray)
                and eps2.ndim == 0))

        if eps1_uniform:
            eps1_diag = eps1 * np.eye(self._G1.shape[0])
        else:
            eps1_diag = np.diag(eps1)
        if eps2_uniform:
            eps2_diag = eps2 * np.eye(self._G1.shape[0])
        else:
            eps2_diag = np.diag(eps2)

        ikdeps = 1j * k * (eps1_diag - eps2_diag)

        # Compress Green functions and surface derivatives for preconditioner
        G1 = self._compress(self._G1)
        H1 = self._compress(self._H1)
        G2 = self._compress_layer_green(self._G2)
        H2 = self._compress_layer_green(self._H2)

        # Get the parallel Green function component
        G2_p = G2.p if hasattr(G2, 'p') else (G2['p'] if isinstance(G2, dict) else G2)
        H2_p = H2.p if hasattr(H2, 'p') else (H2['p'] if isinstance(H2, dict) else H2)

        # LU factorizations of G1 and parallel component
        # v1.7.3 (Phase 2 VRAM-share): forward MNPBEM_VRAM_SHARE_* env vars
        # so each precond LU is partitioned across the worker's GPUs via
        # cuSolverMg when MNPBEM_VRAM_SHARE=1 + MNPBEM_VRAM_SHARE_GPUS>=2.
        # Bit-identical to the single-GPU path when the env vars are off.
        _vram_kwargs = _vram_share_lu_kwargs()
        G1_lu = lu_factor_dispatch(G1, **_vram_kwargs)
        if _CUPY_OK_LAYER:
            _cp_layer.get_default_memory_pool().free_all_blocks()
        G2p_lu = lu_factor_dispatch(G2_p, **_vram_kwargs)
        if _CUPY_OK_LAYER:
            _cp_layer.get_default_memory_pool().free_all_blocks()
        G1i = lu_solve_dispatch(G1_lu, np.eye(G1.shape[0]))
        G2pi = lu_solve_dispatch(G2p_lu, np.eye(G2_p.shape[0]))
        if _CUPY_OK_LAYER:
            _cp_layer.get_default_memory_pool().free_all_blocks()

        # Sigma matrices [Eq. (21)]
        Sigma1 = matmul_dispatch(H1, G1i)
        Sigma2p = matmul_dispatch(H2_p, G2pi)

        # Operator-form Sigma1e and L1.  For uniform eps these reduce to
        # ``eps1 * Sigma1`` / scalar ``eps1`` (fast path); for non-uniform
        # eps they are the dense-BEMRetLayer combinations.
        if eps1_uniform:
            Sigma1e = eps1 * Sigma1
            L1 = eps1
        else:
            Sigma1e = matmul_dispatch(H1, matmul_dispatch(eps1_diag, G1i))
            L1 = matmul_dispatch(G1, matmul_dispatch(eps1_diag, G1i))

        # Perpendicular component of normal vector
        nperp_diag = np.diag(nvec[:, 3 - 1])  # nvec(:,3)

        # Gamma matrix
        Gamma_lu = lu_factor_dispatch(Sigma1 - Sigma2p, **_vram_kwargs)
        if _CUPY_OK_LAYER:
            _cp_layer.get_default_memory_pool().free_all_blocks()
        Gamma = lu_solve_dispatch(Gamma_lu, np.eye(Sigma1.shape[0]))
        if _CUPY_OK_LAYER:
            _cp_layer.get_default_memory_pool().free_all_blocks()

        # Gammapar with only parallel normal vector components
        Gammapar = ikdeps @ self._decorate_gamma(Gamma, nvec)

        # Get structured Green function components
        G2_ss = G2.ss if hasattr(G2, 'ss') else (G2['ss'] if isinstance(G2, dict) else G2)
        G2_sh = G2.sh if hasattr(G2, 'sh') else (G2['sh'] if isinstance(G2, dict) else np.zeros_like(G1))
        G2_hs = G2.hs if hasattr(G2, 'hs') else (G2['hs'] if isinstance(G2, dict) else np.zeros_like(G1))
        G2_hh = G2.hh if hasattr(G2, 'hh') else (G2['hh'] if isinstance(G2, dict) else G2)
        H2_ss = H2.ss if hasattr(H2, 'ss') else (H2['ss'] if isinstance(H2, dict) else H2)
        H2_sh = H2.sh if hasattr(H2, 'sh') else (H2['sh'] if isinstance(H2, dict) else np.zeros_like(H1))
        H2_hs = H2.hs if hasattr(H2, 'hs') else (H2['hs'] if isinstance(H2, dict) else np.zeros_like(H1))
        H2_hh = H2.hh if hasattr(H2, 'hh') else (H2['hh'] if isinstance(H2, dict) else H2)

        # Set up full matrix, Eq. (10).
        # NOTE: ``Sigma1e`` is the operator-form ``H1·eps1·G1⁻¹`` (NOT
        # ``eps1·Sigma1``) — bit-identical for uniform eps but correct for
        # composite particles on a substrate.  ``eps2_diag @ H2_*`` is
        # left as the legacy form: it reduces to ``eps2·H22 - eps2·H12``
        # while the dense form needs ``eps2·H22 - eps1·H12``.  These two
        # agree when eps1 == eps2 on the H12 cross-particle pairs (typical
        # substrate-only sims where the substrate is uniform).  Composite
        # *particles* with non-uniform eps1 + uniform eps2 — the most
        # common substrate composite case — are now correct because the
        # eps1·Sigma1 -> Sigma1e swap is the load-bearing fix.  Non-uniform
        # eps2 (rare) is an approximation; GMRES re-converges via ``_afun``.
        Gammapar_ikdeps = matmul_dispatch(Gammapar, ikdeps)
        nperp_ikdeps = matmul_dispatch(nperp_diag, ikdeps)
        m11 = (matmul_dispatch(Sigma1e - Gammapar_ikdeps, G2_ss)
            - matmul_dispatch(eps2_diag, H2_ss)
            - matmul_dispatch(nperp_ikdeps, G2_hs))
        m12 = (matmul_dispatch(Sigma1e - Gammapar_ikdeps, G2_sh)
            - matmul_dispatch(eps2_diag, H2_sh)
            - matmul_dispatch(nperp_ikdeps, G2_hh))
        m21 = matmul_dispatch(Sigma1, G2_hs) - H2_hs - matmul_dispatch(nperp_ikdeps, G2_ss)
        m22 = matmul_dispatch(Sigma1, G2_hh) - H2_hh - matmul_dispatch(nperp_ikdeps, G2_sh)

        # LU decomposition as block inverse
        # L11 * U11 = M11
        m11_lu = lu_factor_dispatch(m11, **_vram_kwargs)
        if _CUPY_OK_LAYER:
            _cp_layer.get_default_memory_pool().free_all_blocks()
        im11 = lu_solve_dispatch(m11_lu, np.eye(m11.shape[0]))
        # L11 * U12 = M12 -> U12 = inv(L11) * M12
        im12 = matmul_dispatch(im11, m12)
        # L21 * U11 = M21 -> L21 = M21 * inv(U11)
        im21 = matmul_dispatch(m21, im11)
        if _CUPY_OK_LAYER:
            _cp_layer.get_default_memory_pool().free_all_blocks()
        # L22 * U22 = M22 - L21 * U12
        schur = m22 - matmul_dispatch(im21, m12)
        schur_lu = lu_factor_dispatch(schur, **_vram_kwargs)
        if _CUPY_OK_LAYER:
            _cp_layer.get_default_memory_pool().free_all_blocks()
        im22 = lu_solve_dispatch(schur_lu, np.eye(schur.shape[0]))
        # Drop the dense m11/m12/m21/m22/schur buffers; only their LU
        # packages and the (im11/im12/im21/im22) inverses are needed by
        # ``_mfun``.  Each m-block is N x N complex128 (~2.5 GB at
        # N=12672) so this is a load-bearing release.
        del m11, m12, m21, m22, schur
        if _CUPY_OK_LAYER:
            _cp_layer.cuda.runtime.deviceSynchronize()
            _cp_layer.get_default_memory_pool().free_all_blocks()

        # Save variables
        sav = {}
        sav['k'] = k
        sav['nvec'] = nvec
        sav['eps1'] = eps1_diag
        sav['eps2'] = eps2_diag
        sav['G1_lu'] = G1_lu
        sav['G2p_lu'] = G2p_lu
        sav['G2'] = G2
        sav['Sigma1'] = Sigma1
        sav['Sigma1e'] = Sigma1e        # v1.6.0: H1·eps1·G1⁻¹
        sav['L1'] = L1                  # v1.6.0: G1·eps1·G1⁻¹ (or scalar)
        sav['Gamma'] = Gamma
        sav['im'] = [[im11, im12], [im21, im22]]

        self._sav = sav

    @staticmethod
    def _decorate_gamma(
            Gamma: np.ndarray,
            nvec: np.ndarray) -> np.ndarray:

        # MATLAB: fun(Gamma, nvec) in initprecond.m for layer
        # Only uses parallel (x,y) components of normal vector
        # Gamma_decorated = nvec1 * Gamma * nvec1 + nvec2 * Gamma * nvec2
        n = nvec.shape[0]
        result = np.zeros((n, n), dtype = Gamma.dtype)
        for i in range(2):  # only x, y components (parallel)
            nvec_i = np.diag(nvec[:, i])
            result = result + nvec_i @ Gamma @ nvec_i
        return result

    def _pack(self, *args: Any) -> np.ndarray:

        # MATLAB: bemretlayeriter/private/pack.m
        # MATLAB uses column-major (:) flatten, so we use order='F'.
        if len(args) == 4:
            phi, a, phip, ap = args
            total_len = phi.size + a.size + phip.size + ap.size
            vec = np.empty(total_len, dtype = complex)
            offset = 0
            for arr in [phi, a, phip, ap]:
                flat = arr.ravel(order = 'F')
                vec[offset:offset + flat.size] = flat
                offset += flat.size
            return vec
        elif len(args) == 6:
            # phi, apar, aperp, phip, appar, apperp
            phi, apar, aperp, phip, appar, apperp = args
            n = phi.shape[0] if isinstance(phi, np.ndarray) else aperp.shape[0]

            # Determine siz from aperp
            if aperp.ndim == 1:
                siz = 1
            else:
                siz = aperp.shape[1]

            # Combine parallel and perpendicular into full 3D vectors
            if siz == 1:
                a = np.empty((n, 3), dtype = complex)
                a[:, :2] = apar.reshape(n, 2) if apar.ndim >= 2 else apar
                a[:, 2] = aperp.ravel()

                ap = np.empty((n, 3), dtype = complex)
                ap[:, :2] = appar.reshape(n, 2) if appar.ndim >= 2 else appar
                ap[:, 2] = apperp.ravel()
            else:
                a = np.empty((n, 3, siz), dtype = complex)
                a[:, :2, :] = apar.reshape(n, 2, siz)
                a[:, 2, :] = aperp.reshape(n, siz)

                ap = np.empty((n, 3, siz), dtype = complex)
                ap[:, :2, :] = appar.reshape(n, 2, siz)
                ap[:, 2, :] = apperp.reshape(n, siz)

            total_len = phi.size + a.size + phip.size + ap.size
            vec = np.empty(total_len, dtype = complex)
            offset = 0
            for arr in [phi, a, phip, ap]:
                flat = arr.ravel(order = 'F')
                vec[offset:offset + flat.size] = flat
                offset += flat.size
            return vec

    def _unpack(self,
            vec: np.ndarray,
            nout: int = 4) -> Tuple:

        # MATLAB: bemretlayeriter/private/unpack.m
        # MATLAB uses column-major reshape, so we use order='F'.
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces

        # Last dimension
        siz = int(vec.size / (8 * n))

        # Reshape vector (column-major to match MATLAB)
        vec_2d = vec.reshape(-1, 8, order = 'F')

        # Extract potentials from vector
        phi = vec_2d[:, 0].reshape(n, siz, order = 'F') if siz > 1 else vec_2d[:, 0].reshape(n)
        a = vec_2d[:, 1:4].reshape(n, 3, siz, order = 'F') if siz > 1 else vec_2d[:, 1:4].reshape(n, 3)
        phip = vec_2d[:, 4].reshape(n, siz, order = 'F') if siz > 1 else vec_2d[:, 4].reshape(n)
        ap = vec_2d[:, 5:8].reshape(n, 3, siz, order = 'F') if siz > 1 else vec_2d[:, 5:8].reshape(n, 3)

        if nout == 4:
            return phi, a, phip, ap
        else:
            # Decompose vectors into parallel and perpendicular components
            if a.ndim == 2:
                apar = a[:, :2]
                aperp = a[:, 2]
            else:
                apar = a[:, :2, :]
                aperp = a[:, 2, :]

            if ap.ndim == 2:
                appar = ap[:, :2]
                apperp = ap[:, 2]
            else:
                appar = ap[:, :2, :]
                apperp = ap[:, 2, :]

            return phi, apar, aperp, phip, appar, apperp

    @staticmethod
    def _outer(
            nvec: np.ndarray,
            val: Any,
            mul: Optional[np.ndarray] = None) -> Any:

        # MATLAB: bemretlayeriter/private/outer.m
        if isinstance(val, (int, float)) and val == 0:
            return 0

        if mul is not None:
            if val.ndim == 1:
                val = val * mul
            else:
                val = val * mul[:, np.newaxis] if mul.ndim == 1 else val * mul

        ndim = nvec.shape[1]  # 2 for parallel, 3 for full

        if val.ndim == 1:
            n = val.shape[0]
            result = np.empty((n, ndim), dtype = val.dtype)
            for i in range(ndim):
                result[:, i] = val * nvec[:, i]
            return result
        else:
            n = val.shape[0]
            siz = val.shape[1]
            result = np.empty((n, ndim, siz), dtype = val.dtype)
            for i in range(ndim):
                result[:, i, :] = val * nvec[:, i:i + 1]
            return result

    @staticmethod
    def _inner(
            nvec: np.ndarray,
            a: Any,
            mul: Optional[np.ndarray] = None) -> Any:

        # MATLAB: bemretlayeriter/private/inner.m
        if isinstance(a, (int, float)) and a == 0:
            return 0

        ndim = nvec.shape[1]

        if a.ndim == 2:
            result = np.zeros(a.shape[0], dtype = a.dtype)
            for i in range(min(ndim, a.shape[1])):
                result = result + a[:, i] * nvec[:, i]
        elif a.ndim == 3:
            siz = a.shape[2]
            result = np.zeros((a.shape[0], siz), dtype = a.dtype)
            for i in range(min(ndim, a.shape[1])):
                result = result + a[:, i, :] * nvec[:, i:i + 1]
        else:
            result = a

        if mul is not None:
            if result.ndim == 1:
                result = result * mul
            else:
                result = result * mul[:, np.newaxis] if mul.ndim == 1 else result * mul

        return result

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

    def _excitation(self,
            exc: CompStruct) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # MATLAB: bemretlayeriter/private/excitation.m
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces

        phi1 = getattr(exc, 'phi1', 0)
        phi1p = getattr(exc, 'phi1p', 0)
        a1 = getattr(exc, 'a1', 0)
        a1p = getattr(exc, 'a1p', 0)
        phi2 = getattr(exc, 'phi2', 0)
        phi2p = getattr(exc, 'phi2p', 0)
        a2 = getattr(exc, 'a2', 0)
        a2p = getattr(exc, 'a2p', 0)

        k = self._k
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

        return phi, a, alpha, De

    def _afun(self,
            vec: np.ndarray) -> np.ndarray:

        # MATLAB: bemretlayeriter/private/afun.m
        # Waxenegger et al., Comp. Phys. Commun. 193, 138 (2015)
        #
        # v1.6.0 (agent B) — multi-material + substrate fix.  Mirrors the
        # v1.5.1 BEMRetIter operator-form correction.  The dense
        # ``BEMRetLayer`` (bem_ret_layer.py) absorbs eps into combinations
        # such as ``Sigma1e = H1·eps1·G1⁻¹`` and ``L1 = G1·eps1·G1⁻¹`` so
        # eps acts at the *source* point of the BEM convolution.  The
        # original Python iter form applied eps *after* the matvec
        # (``eps · (M·sig)``); algebraically that equals the operator form
        # only when eps is uniform.  For composite particles + substrate
        # (e.g. Au@Ag dimer on glass) the two forms diverge by the same
        # mechanism that drove the v1.5.1 70 % drift on BEMRetIter.
        #
        # Fix: for non-uniform eps, push the per-face eps multiply *into*
        # the matvec (M·diag(eps)·v).  Scalar eps commutes so the cheap
        # path is left bit-identical to the legacy behaviour.
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces

        # Split vector array into 6 components
        sig1, h1par, h1perp, sig2, h2par, h2perp = self._unpack(vec, nout = 6)

        k = self._k
        nvec = self._nvec
        eps1 = self._eps1
        eps2 = self._eps2
        npar = nvec[:, :2]
        nperp = nvec[:, 2]

        G1 = self._G1
        H1 = self._H1
        G2 = self._G2
        H2 = self._H2

        # Get layer components
        G2_ss = G2.ss if hasattr(G2, 'ss') else (G2['ss'] if isinstance(G2, dict) else G2)
        G2_sh = G2.sh if hasattr(G2, 'sh') else (G2['sh'] if isinstance(G2, dict) else np.zeros_like(G1))
        G2_hs = G2.hs if hasattr(G2, 'hs') else (G2['hs'] if isinstance(G2, dict) else np.zeros_like(G1))
        G2_hh = G2.hh if hasattr(G2, 'hh') else (G2['hh'] if isinstance(G2, dict) else G2)
        G2_p = G2.p if hasattr(G2, 'p') else (G2['p'] if isinstance(G2, dict) else G2)
        H2_ss = H2.ss if hasattr(H2, 'ss') else (H2['ss'] if isinstance(H2, dict) else H2)
        H2_sh = H2.sh if hasattr(H2, 'sh') else (H2['sh'] if isinstance(H2, dict) else np.zeros_like(H1))
        H2_hs = H2.hs if hasattr(H2, 'hs') else (H2['hs'] if isinstance(H2, dict) else np.zeros_like(H1))
        H2_hh = H2.hh if hasattr(H2, 'hh') else (H2['hh'] if isinstance(H2, dict) else H2)
        H2_p = H2.p if hasattr(H2, 'p') else (H2['p'] if isinstance(H2, dict) else H2)

        def matmul(a_mat: np.ndarray, b: np.ndarray) -> np.ndarray:
            if b.ndim == 1:
                return a_mat @ b
            n_rows = a_mat.shape[0]
            return (a_mat @ b.reshape(b.shape[0], -1)).reshape(n_rows, *b.shape[1:])

        def mul(x: Any, y: np.ndarray) -> np.ndarray:
            if np.isscalar(x):
                return x * y
            # x is array with leading dim n; y is 1-D (n,) diag or has matching shape.
            if y.ndim == 1 and x.ndim > 1:
                return x * y.reshape(-1, *([1] * (x.ndim - 1)))
            if y.ndim == 1:
                return x * y
            return x[:, np.newaxis] * y if x.ndim == 1 else x * y

        eps1_scalar = (np.isscalar(eps1) or (isinstance(eps1, np.ndarray)
                and eps1.ndim == 0))
        eps2_scalar = (np.isscalar(eps2) or (isinstance(eps2, np.ndarray)
                and eps2.ndim == 0))

        def _eps_apply(eps_val: Any, x: np.ndarray) -> np.ndarray:
            if np.isscalar(eps_val) or (isinstance(eps_val, np.ndarray)
                    and eps_val.ndim == 0):
                return eps_val * x
            if x.ndim == 1:
                return eps_val * x
            return eps_val.reshape(-1, *([1] * (x.ndim - 1))) * x

        # Apply Green functions to surface charges
        Gsig1 = G1 @ sig1
        Gsig2 = G2_ss @ sig2 + G2_sh @ h2perp
        Hsig1 = H1 @ sig1
        Hsig2 = H2_ss @ sig2 + H2_sh @ h2perp

        # Apply Green functions to parallel surface currents
        Gh1par = matmul(G1, h1par)
        Gh2par = matmul(G2_p, h2par)
        Hh1par = matmul(H1, h1par)
        Hh2par = matmul(H2_p, h2par)

        # Apply Green functions to perpendicular surface currents
        Gh1perp = G1 @ h1perp
        Gh2perp = G2_hh @ h2perp + G2_hs @ sig2
        Hh1perp = H1 @ h1perp
        Hh2perp = H2_hh @ h2perp + H2_hs @ sig2

        # eps-decorated Green-function applications for the alpha/De rows.
        # Scalar eps: ``eps · (M·v) == M · (eps·v)``, so reuse the legacy
        # post-matvec multiply (bit-identical).
        # Non-scalar eps: redo G/H with the eps weights pushed *into* the
        # source vector, matching the dense BEMRetLayer operator form.
        if eps1_scalar:
            L_Gsig1 = eps1 * Gsig1
            L_Hsig1 = eps1 * Hsig1
            L_Gh1par = eps1 * Gh1par
            L_Gh1perp = eps1 * Gh1perp
        else:
            eps1_sig1 = _eps_apply(eps1, sig1)
            eps1_h1par = _eps_apply(eps1, h1par)
            eps1_h1perp = _eps_apply(eps1, h1perp)
            L_Gsig1 = G1 @ eps1_sig1
            L_Hsig1 = H1 @ eps1_sig1
            L_Gh1par = matmul(G1, eps1_h1par)
            L_Gh1perp = G1 @ eps1_h1perp

        if eps2_scalar:
            L_Gsig2 = eps2 * Gsig2
            L_Hsig2 = eps2 * Hsig2
            L_Gh2par = eps2 * Gh2par
            L_Gh2perp = eps2 * Gh2perp
        else:
            eps2_sig2 = _eps_apply(eps2, sig2)
            eps2_h2par = _eps_apply(eps2, h2par)
            eps2_h2perp = _eps_apply(eps2, h2perp)
            L_Gsig2 = G2_ss @ eps2_sig2 + G2_sh @ eps2_h2perp
            L_Hsig2 = H2_ss @ eps2_sig2 + H2_sh @ eps2_h2perp
            L_Gh2par = matmul(G2_p, eps2_h2par)
            L_Gh2perp = G2_hh @ eps2_h2perp + G2_hs @ eps2_sig2

        # Eq. (7a)
        phi = Gsig1 - Gsig2
        # Eqs. (7b, c)
        apar = Gh1par - Gh2par
        aperp = Gh1perp - Gh2perp

        # Eqs. (8a, b) — operator form
        alphapar = Hh1par - Hh2par - \
            1j * k * (self._outer(npar, L_Gsig1) - self._outer(npar, L_Gsig2))
        alphaperp = Hh1perp - Hh2perp - \
            1j * k * (mul(L_Gsig1, nperp) - mul(L_Gsig2, nperp))

        # Eq. (9) — operator form
        De = L_Hsig1 - L_Hsig2 - \
            1j * k * (self._inner(npar, L_Gh1par) - self._inner(npar, L_Gh2par)) - \
            1j * k * (mul(L_Gh1perp, nperp) - mul(L_Gh2perp, nperp))

        return self._pack(phi, apar, aperp, De, alphapar, alphaperp)

    def _mfun(self,
            vec: np.ndarray) -> np.ndarray:

        # MATLAB: bemretlayeriter/private/mfun.m
        # Waxenegger et al., Comp. Phys. Commun. 193, 138 (2015)
        #
        # v1.6.0 (agent B) — operator-form alpha/De modifications.  The
        # legacy expression ``eps1 · phi`` and ``eps1 · (Sigma1·phi)`` is
        # replaced by ``L1 · phi`` and ``Sigma1e · phi`` where
        # ``L1 = G1·eps1·G1⁻¹`` and ``Sigma1e = H1·eps1·G1⁻¹``.  For uniform
        # eps these reduce to scalar multiplies (bit-identical) but for
        # composite particles on a substrate they correct the iter-precond
        # to the dense-BEMRetLayer reduction.

        # Unpack matrices
        phi, a, De, alpha = self._unpack(vec, nout = 4)

        sav = self._sav
        k = sav['k']
        nvec = sav['nvec']
        G2 = sav['G2']
        G1_lu = sav['G1_lu']
        G2p_lu = sav['G2p_lu']
        eps1 = sav['eps1']
        eps2 = sav['eps2']
        Sigma1 = sav['Sigma1']
        Sigma1e = sav['Sigma1e']
        L1 = sav['L1']
        Gamma = sav['Gamma']
        im = sav['im']

        deps = eps1 - eps2
        npar = nvec[:, :2]

        if a.ndim == 2:
            apar = a[:, :2]
            aperp = a[:, 2]
        else:
            apar = a[:, :2, :]
            aperp = a[:, 2, :]

        def matmul1(a_mat: np.ndarray, b: np.ndarray) -> np.ndarray:
            # Multiply (n, n) matrix with (n, ...) array, preserving trailing dims.
            if b.ndim == 1:
                return a_mat @ b
            n_rows = a_mat.shape[0] if not np.isscalar(a_mat) else b.shape[0]
            return (a_mat @ b.reshape(b.shape[0], -1)).reshape(n_rows, *b.shape[1:])

        def _ls(lu_piv, b):
            if b.ndim == 1:
                return lu_solve_dispatch(lu_piv, b)
            n_rows = b.shape[0]
            return lu_solve_dispatch(lu_piv, b.reshape(b.shape[0], -1)).reshape(n_rows, *b.shape[1:])

        def matmul_eps(eps_mat: Any, b: np.ndarray) -> np.ndarray:
            # Apply diagonal eps (scalar / (n,n) diag matrix) to (n, ...) array.
            if np.isscalar(eps_mat):
                return eps_mat * b
            if b.ndim == 1:
                return eps_mat @ b
            return (eps_mat @ b.reshape(b.shape[0], -1)).reshape(b.shape)

        def matmul_op(op_val: Any, b: np.ndarray) -> np.ndarray:
            # Apply operator (scalar / dense (n,n)) to (n, ...) array.
            if np.isscalar(op_val) or (isinstance(op_val, np.ndarray)
                    and op_val.ndim == 0):
                return op_val * b
            if b.ndim == 1:
                return op_val @ b
            return (op_val @ b.reshape(b.shape[0], -1)).reshape(b.shape)

        # Modify alpha — operator form: L1·phi instead of eps1·phi.
        L1_phi = matmul_op(L1, phi)
        L1_a = matmul_op(L1, a)
        alpha = alpha - matmul1(Sigma1, a) + 1j * k * self._outer(nvec, L1_phi)
        if alpha.ndim == 2:
            alphapar = alpha[:, :2]
            alphaperp = alpha[:, 2]
        else:
            alphapar = alpha[:, :2, :]
            alphaperp = alpha[:, 2, :]

        # Modify De — operator form: Sigma1e·phi instead of eps1·(Sigma1·phi),
        # and L1·a instead of eps1·a.
        De = De - matmul1(Sigma1e, phi) + \
            1j * k * self._inner(nvec, L1_a) + \
            1j * k * self._inner(npar, matmul1(deps @ Gamma, alphapar))

        # Solve Eq. (10) using block LU
        sig2, h2perp = self._solve_block_lu(im, De, alphaperp)

        # Get G2 components
        # G1_lu = ("cpu"/"gpu"/"mgpu", lu_matrix-or-handle, piv).  For
        # mgpu the second slot is a MultiGPULU handle with ``.N``; for
        # cpu/gpu it's an ndarray with ``.shape``.
        _lu_payload = G1_lu[1]
        n_g = int(getattr(_lu_payload, 'N', None)
                if hasattr(_lu_payload, 'N')
                else _lu_payload.shape[0])
        G2_ss = G2.ss if hasattr(G2, 'ss') else (G2['ss'] if isinstance(G2, dict) else G2)
        G2_sh = G2.sh if hasattr(G2, 'sh') else (G2['sh'] if isinstance(G2, dict) else np.zeros((n_g, n_g)))
        G2_p = G2.p if hasattr(G2, 'p') else (G2['p'] if isinstance(G2, dict) else G2)
        G2_hh = G2.hh if hasattr(G2, 'hh') else (G2['hh'] if isinstance(G2, dict) else G2)
        G2_hs = G2.hs if hasattr(G2, 'hs') else (G2['hs'] if isinstance(G2, dict) else np.zeros((n_g, n_g)))

        # Parallel component, Eq. (A.1)
        h2par = _ls(G2p_lu, matmul1(Gamma, alphapar +
            1j * k * self._outer(npar, deps @ (G2_ss @ sig2 + G2_sh @ h2perp))))

        # Surface charges at inner interface
        sig1 = _ls(G1_lu, G2_ss @ sig2 + G2_sh @ h2perp + phi)

        # Surface currents at inner interface
        h1perp = _ls(G1_lu, G2_hh @ h2perp + G2_hs @ sig2 + aperp)
        h1par = _ls(G1_lu, matmul1(G2_p, h2par) + apar)

        result = self._pack(sig1, h1par, h1perp, sig2, h2par, h2perp)
        return result

    @staticmethod
    def _solve_block_lu(
            im: List[List[np.ndarray]],
            b1: np.ndarray,
            b2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        # MATLAB: fun(M, b1, b2) in mfun.m
        # Solve 2x2 block system using pre-computed inverse factors:
        #   im11 = inv(M11), im12 = inv(M11) @ M12,
        #   im21 = M21 @ inv(M11), im22 = inv(Schur)
        # where Schur = M22 - M21 @ inv(M11) @ M12

        im11, im12 = im[0][0], im[0][1]
        im21, im22 = im[1][0], im[1][1]

        x2 = im22 @ (b2 - im21 @ b1)
        x1 = im11 @ b1 - im12 @ x2

        return x1, x2

    def solve(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMRetLayerIter']:

        # MATLAB: bemretlayeriter/solve.m
        # Initialize BEM solver (if needed)
        self._init_matrices(exc.enei)

        # External excitation
        phi, a, alpha, De = self._excitation(exc)

        # Size of excitation arrays
        siz1 = phi.shape
        siz2 = a.shape

        # Pack everything to single vector
        b = self._pack(phi, a, De, alpha)

        # Function for matrix multiplication
        fa = self._afun
        fm = None
        if self.precond is not None:
            fm = self._mfun

        # v1.7.2: drain right before GMRES enters its Krylov build-up so
        # the iter loop sees the maximum amount of free GPU memory.  The
        # init pipeline above leaves up to 2 N^2 of transient asarray
        # buffers in the pool; releasing them now keeps the per-iter
        # matvec / preconditioner-apply allocations from triggering a
        # fragmentation-induced OOM mid-sweep.
        if _CUPY_OK_LAYER:
            _cp_layer.cuda.runtime.deviceSynchronize()
            _cp_layer.get_default_memory_pool().free_all_blocks()
        _gpu_pool_cleanup_layer()

        # Iterative solution
        x, self_updated = self._iter_solve(None, b, fa, fm)
        # v1.7.2: drop the GMRES matvec closures (which keep references to
        # the cached G/H matrices and the LU factor tuples).  The Krylov
        # subspace itself is held internally by scipy.sparse.linalg.gmres
        # for the duration of the call; once we return here it's gone and
        # we just want to compact the pool before the next wavelength's
        # init drains the prior state.
        del fa, fm
        if _CUPY_OK_LAYER:
            _cp_layer.cuda.runtime.deviceSynchronize()
            _cp_layer.get_default_memory_pool().free_all_blocks()
        _gpu_pool_cleanup_layer()

        # Unpack and save solution vector
        sig1, h1, sig2, h2 = self._unpack(x, nout = 4)

        # Reshape surface charges and currents
        if len(siz1) > 1:
            sig1 = sig1.reshape(siz1)
            sig2 = sig2.reshape(siz1)
        if len(siz2) > 2:
            h1 = h1.reshape(siz2)
            h2 = h2.reshape(siz2)

        # Host-materialize cupy results so the returned sig is always
        # CPU-resident. Mirrors BEMRetLayer.solve's defensive guard and
        # lets downstream code treat sig fields as numpy uniformly.
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
        if _CUPY_OK_LAYER:
            _cp_layer.get_default_memory_pool().free_all_blocks()
            _cp_layer.get_default_pinned_memory_pool().free_all_blocks()
        _gpu_pool_cleanup_layer()

        return sig, self

    def __truediv__(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMRetLayerIter']:

        # MATLAB: bemretlayeriter/mldivide.m
        return self.solve(exc)

    def __mul__(self,
            sig: CompStruct) -> CompStruct:

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

        # MATLAB: bemretlayeriter/field.m
        # Waxenegger et al., Comp. Phys. Commun. 193, 138 (2015)
        if hasattr(self.g, 'deriv') and self.g.deriv == 'cart':
            return self.g.field(sig, inout)

        # Norm-based derivative approach
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

        # MATLAB: bemretlayeriter/potential.m
        return self.g.potential(sig, inout)

    def setup_tabulation(self, nr = 30, nz = 20):

        if self.g is not None:
            self.g.setup_tabulation(nr = nr, nz = nz)

    def clear(self) -> 'BEMRetLayerIter':

        # MATLAB: bemretlayeriter/clear.m
        self._G1 = None
        self._H1 = None
        self._G2 = None
        self._H2 = None
        self._sav = None
        # v1.7.2: explicit clear() means the user wants the device drained.
        # Without this the LU factors / G blocks just released above stay
        # in the cupy pool until the next solve triggers a free_all_blocks.
        if _CUPY_OK_LAYER:
            _cp_layer.cuda.runtime.deviceSynchronize()
            _cp_layer.get_default_memory_pool().free_all_blocks()
            _cp_layer.get_default_pinned_memory_pool().free_all_blocks()
        _gpu_pool_cleanup_layer()
        return self

    def __call__(self,
            enei: float) -> 'BEMRetLayerIter':

        return self._init_matrices(enei)

    def __repr__(self) -> str:
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces if hasattr(self.p, 'nfaces') else '?'
        status = 'enei={:.1f}nm'.format(self.enei) if self.enei is not None else 'not initialized'
        return 'BEMRetLayerIter(p: {} faces, solver={}, {})'.format(n, self.solver, status)


class _LayerGreen(object):

    def __init__(self) -> None:
        self.ss = None
        self.hh = None
        self.p = None
        self.sh = None
        self.hs = None
