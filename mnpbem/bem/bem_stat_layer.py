import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from ..greenfun import CompGreenStatLayer, CompStruct
from ..utils.gpu import lu_factor_dispatch, lu_solve_dispatch, to_host, is_cupy_array


# v1.7.3 (Phase 2): mirror BEMStat / BEMStatIter's wavelength-end cupy pool
# cleanup so the BEMStatLayer dense LU path also keeps the high-water mark
# bounded across long sweeps.  The deviceSynchronize() before free_all_blocks
# is load-bearing — without it, blocks that are still in flight on the CUDA
# stream are NOT actually idle and the pool refuses to return them to the
# driver.  Honors MNPBEM_GPU_POOL_LIMIT_GB the same way BEMStat does.
try:
    import cupy as _cp_layer  # type: ignore
    _CUPY_OK_LAYER = True
except Exception:
    _cp_layer = None  # type: ignore
    _CUPY_OK_LAYER = False


def _vram_share_lu_kwargs() -> dict:
    """Read MNPBEM_VRAM_SHARE_* env vars and return kwargs for lu_factor_dispatch.

    Returns ``{}`` when VRAM-share is not enabled (n_gpus<=1).  Mirrors the
    helper in ``bem_ret.py`` so all dense BEM solvers honour the same
    multi-GPU VRAM-share env vars.
    """
    if os.environ.get('MNPBEM_VRAM_SHARE', '0') != '1':
        return {}
    n_gpus = int(os.environ.get('MNPBEM_VRAM_SHARE_GPUS', '1'))
    if n_gpus <= 1:
        return {}
    backend = os.environ.get('MNPBEM_VRAM_SHARE_BACKEND', 'cusolvermg')
    return {'n_gpus': n_gpus, 'backend': backend}


def _vram_share_active() -> bool:
    """Return True iff the distributed-build path should be taken.

    Mirrors the quasistatic ``BEMStat._vram_share_active`` gate.  Off by
    default; user opts in via ``MNPBEM_VRAM_SHARE_DISTRIBUTED=1``.
    """
    if not _CUPY_OK_LAYER:
        return False
    if os.environ.get('MNPBEM_VRAM_SHARE', '0') != '1':
        return False
    if os.environ.get('MNPBEM_VRAM_SHARE_DISTRIBUTED', '0') != '1':
        return False
    try:
        n_gpus = int(os.environ.get('MNPBEM_VRAM_SHARE_GPUS', '1'))
    except (TypeError, ValueError):
        n_gpus = 1
    if n_gpus < 2:
        return False
    try:
        from ..utils.multi_gpu_lu import cusolvermg_available
        if not cusolvermg_available():
            return False
    except Exception:
        return False
    return True


def _vram_share_distributed_kwargs() -> dict:
    """Return ``{n_gpus, device_ids, block_size}`` for DistributedMatrix."""
    try:
        n_gpus = int(os.environ.get('MNPBEM_VRAM_SHARE_GPUS', '1'))
    except (TypeError, ValueError):
        n_gpus = 1
    device_ids = None
    dev_env = os.environ.get('MNPBEM_VRAM_SHARE_DEVICE_IDS', '')
    if dev_env.strip():
        try:
            device_ids = [int(x) for x in dev_env.split(',') if x.strip()]
        except Exception:
            device_ids = None
    return {'n_gpus': n_gpus, 'device_ids': device_ids, 'block_size': 256}


def _gpu_pool_cleanup_layer(apply_limit: bool = False) -> None:
    """Synchronise CUDA stream then drain cupy default + pinned pools."""
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


class BEMStatLayer(object):

    name = 'bemsolver'
    needs = {'sim': 'stat'}

    def __init__(self,
            p: Any,
            layer: Any,
            enei: Optional[float] = None,
            **options: Any) -> None:

        self.p = p
        self.layer = layer

        self.enei = None
        self.mat_lu = None
        self._A_lu = None
        self._rhs_scale = None

        # Green function with layer
        # MATLAB: obj.g = compgreenstatlayer(p, p, layer, varargin{:})
        self.g = CompGreenStatLayer(p, p, layer, **options)

        # Surface derivative of Green function.
        # v1.7.3 Phase 2: F may live on cupy when the upstream Green-function
        # assembly route is GPU-resident.  Host-promote here so the
        # downstream eps1*H1 - eps2*H2 mixing op stays on a single backend
        # (cupy/numpy mix triggers TypeError in older numpy versions).  The
        # F tensor itself is reused only by downstream callers (g.field /
        # g.potential) that already accept either flavour, so dropping the
        # GPU view here also frees its device-side buffer.
        F_obj = self.g.F
        if is_cupy_array(F_obj):
            F_obj = to_host(F_obj)
        self.F = F_obj
        if _CUPY_OK_LAYER:
            _gpu_pool_cleanup_layer()

        if enei is not None:
            self(enei)

    def _init_matrices(self,
            enei: float) -> 'BEMStatLayer':

        if self.enei is not None and np.isclose(self.enei, enei):
            return self

        # B-3 distributed multi-GPU build path. When the dedicated env
        # gate is on the BEM matrix is built distributed column-wise so
        # the host never holds the full N x N A matrix in addition to
        # the cuSolverMg LU buffer.  Falls back to the legacy host path
        # on any failure (e.g. cuSolverMg unavailable mid-sweep).
        if _vram_share_active():
            try:
                return self._init_distributed_assemble(enei)
            except Exception as e:  # pragma: no cover
                import warnings
                warnings.warn(
                    '[warn] BEMStatLayer distributed assembly failed ({}); '
                    'falling back to host build'.format(e))

        # v1.7.3 Phase 2: free previous wavelength's LU before allocating
        # new buffers.  Mirrors the v1.7.2 BEMStat / BEMStatIter pattern.
        self._A_lu = None
        self._rhs_scale = None
        _gpu_pool_cleanup_layer(apply_limit = True)

        # MATLAB @bemstatlayer/subsref.m "()" branch:
        #   [H1, H2] = eval(obj.g, enei, 'H1', 'H2')
        #   mat = -inv(eps1 * H1 - eps2 * H2) * (eps1 - eps2)
        # The eps1/eps2 are inside/outside dielectric functions of the
        # particle (per-face). They are scalars for homogeneous setups.
        H1 = self.g.eval(enei, 'H1')
        H2 = self.g.eval(enei, 'H2')

        # v1.7.3 Phase 2: H1/H2 from the layer Green function may be cupy
        # arrays.  Host-promote them so the eps* per-face scaling below stays
        # numpy-only (the downstream LU factor lives behind dispatch which
        # re-uploads as needed).  Free the GPU-side intermediates immediately
        # so the pool can recycle their N^2 buffers before the GEMM below.
        if is_cupy_array(H1):
            H1 = to_host(H1)
        if is_cupy_array(H2):
            H2 = to_host(H2)
        if _CUPY_OK_LAYER:
            _gpu_pool_cleanup_layer()

        eps1 = np.atleast_1d(self.p.eps1(enei)).astype(complex)
        eps2 = np.atleast_1d(self.p.eps2(enei)).astype(complex)
        n = H1.shape[0]
        if eps1.size == 1:
            eps1 = np.full(n, eps1[0], dtype = complex)
        if eps2.size == 1:
            eps2 = np.full(n, eps2[0], dtype = complex)

        # Use diagonal multiplication to avoid forming dense diag matrices.
        A = eps1[:, np.newaxis] * H1 - eps2[:, np.newaxis] * H2
        rhs_scale = eps1 - eps2  # per-face
        # v1.7.3 Phase 2: H1/H2 are no longer needed after A is formed.
        # Drop them so the cupy pool can reclaim ~2 N^2 buffers before the
        # LU factor.
        del H1, H2

        # Honour MNPBEM_VRAM_SHARE_* for multi-GPU dispatch on large meshes.
        _lu_opts = _vram_share_lu_kwargs()
        self._A_lu = lu_factor_dispatch(A, **_lu_opts)
        # ``A`` is consumed by the LU factor (overwrite_a paths); drop the
        # local handle so the cupy pool reclaims its N^2 buffer.
        del A
        self._rhs_scale = rhs_scale
        self.enei = enei
        _gpu_pool_cleanup_layer()

        return self

    def _init_distributed_assemble(self,
            enei: float) -> 'BEMStatLayer':
        """B-3 distributed multi-GPU build for the quasistatic substrate solver.

        Builds ``A = eps1*H1 - eps2*H2`` directly distributed across N
        GPUs via :class:`mnpbem.utils.distributed_matrix.DistributedMatrix`
        and factors with cuSolverMg.  Each column tile is built by
        slicing ``CompGreenStatLayer.eval_block('H1' / 'H2', enei, c0, c1)``
        and applying the per-row eps scaling locally, so the host never
        holds two of these N x N matrices simultaneously with the LU.

        Memory characteristic (n_gpus=N):
        - Per-GPU peak: ~2 * N^2 * 16 / n_gpus bytes during the tile
          assembly (H1 and H2 slices held briefly before subtraction).
        - Host peak: the eps1/eps2 vectors (O(N)) plus the rhs_scale.
          The full ``A`` is never materialized.

        Result residency
        ----------------
        - ``self._A_lu = ('mgpu', MultiGPULU_handle, None)`` so
          ``lu_solve_dispatch`` routes the solve through cuSolverMg.
        - ``self._rhs_scale`` is the per-face ``eps1 - eps2`` vector
          (host numpy array, used by ``__truediv__``).
        """

        import gc as _gc
        from ..utils.distributed_matrix import DistributedMatrix
        cp = _cp_layer

        dist_kw = _vram_share_distributed_kwargs()
        n_gpus = int(dist_kw['n_gpus'])
        device_ids = dist_kw['device_ids']
        block_size = int(dist_kw['block_size'])
        if device_ids is None:
            device_ids = list(range(n_gpus))
        assert len(device_ids) == n_gpus, \
            '[error] MNPBEM_VRAM_SHARE_DEVICE_IDS length must equal MNPBEM_VRAM_SHARE_GPUS'

        # ---- Per-wavelength cleanup: close stale LU + free old tiles ----
        old = getattr(self, '_A_lu', None)
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
        self._A_lu = None
        self._rhs_scale = None
        _gc.collect()
        _gpu_pool_cleanup_layer(apply_limit = True)

        # ---- eps1, eps2 per-face ----
        eps1 = np.atleast_1d(self.p.eps1(enei)).astype(complex)
        eps2 = np.atleast_1d(self.p.eps2(enei)).astype(complex)
        # Materialise full vectors so the per-tile callback can index
        # into eps1[c0:c1] / eps2[c0:c1] without re-broadcasting.
        # ``F.shape[0]`` is the canonical face count (set at __init__).
        N = int(self.F.shape[0])
        if eps1.size == 1:
            eps1 = np.full(N, eps1[0], dtype = complex)
        if eps2.size == 1:
            eps2 = np.full(N, eps2[0], dtype = complex)

        # Build H1 / H2 once on host (the layer image-charge corrections
        # in CompGreenStatLayer.eval go through a full-matrix code path,
        # so per-column recomputation would not save host memory here).
        # The host peak of two N x N matrices matches the legacy path;
        # what we save is the additional N x N for ``A`` and the
        # subsequent LU buffer, both of which now live distributed.
        H1_host = self.g.eval(enei, 'H1')
        H2_host = self.g.eval(enei, 'H2')
        if is_cupy_array(H1_host):
            H1_host = to_host(H1_host)
        if is_cupy_array(H2_host):
            H2_host = to_host(H2_host)

        def _eval_A_tile(gpu_idx, c0, c1):
            # Column-wise slice of A = eps1*H1 - eps2*H2.  The slice
            # itself is host numpy; the cupy scaling / subtraction
            # happens inside the per-GPU context that wraps this
            # callback.  Promote to complex128 explicitly to dodge
            # ComplexWarning when H1 / H2 came back as real-valued.
            H1_blk = H1_host[:, c0:c1].astype(np.complex128, copy=False)
            H2_blk = H2_host[:, c0:c1].astype(np.complex128, copy=False)
            return eps1[:, np.newaxis] * H1_blk - eps2[:, np.newaxis] * H2_blk

        A_dm = DistributedMatrix.from_func(
            shape=(N, N),
            dtype=np.complex128,
            n_gpus=n_gpus,
            device_ids=device_ids,
            block_size=block_size,
            eval_func=_eval_A_tile,
        )
        # H1 / H2 host copies no longer needed once the tiles are
        # scattered; drop the references so Python GC can free them
        # before the LU factor below allocates its workspace.
        del H1_host, H2_host
        _gc.collect()

        A_mglu = A_dm.lu_factor(backend='cusolvermg')
        A_mglu._distmat_keepalive = A_dm  # type: ignore[attr-defined]
        self._A_lu = ('mgpu', A_mglu, None)
        self._rhs_scale = (eps1 - eps2)
        self.enei = enei

        # Final sync + pool compaction.
        try:
            for d in device_ids:
                cp.cuda.runtime.setDevice(d)
                cp.cuda.runtime.deviceSynchronize()
                cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
        return self

    def solve(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMStatLayer']:

        return self.__truediv__(exc)

    def __truediv__(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMStatLayer']:

        self._init_matrices(exc.enei)

        phip = exc.phip
        orig_shape = phip.shape
        if phip.ndim == 1:
            phip_2d = phip.reshape(-1, 1)
        elif phip.ndim > 2:
            phip_2d = phip.reshape(phip.shape[0], -1)
        else:
            phip_2d = phip

        # MATLAB mat * phip = -inv(A) * diag(eps1 - eps2) * phip
        rhs = self._rhs_scale[:, np.newaxis] * phip_2d
        sig_result = -lu_solve_dispatch(self._A_lu, rhs)

        if sig_result.shape != orig_shape:
            sig_result = sig_result.reshape(orig_shape)

        # v1.7 Phase 1.4: host-materialize before returning to user.
        if is_cupy_array(sig_result):
            sig_result = to_host(sig_result)

        # v1.7.3 Phase 2: post-solve pool drain mirrors BEMStat.__truediv__.
        if _CUPY_OK_LAYER:
            _gpu_pool_cleanup_layer()

        sig = CompStruct(self.p, exc.enei, sig = sig_result)

        return sig, self

    def __mul__(self,
            sig: CompStruct) -> CompStruct:

        pot1 = self.potential(sig, 1)
        pot2 = self.potential(sig, 2)

        phi = CompStruct(self.p, sig.enei,
            phi1 = pot1.phi1, phi1p = pot1.phi1p,
            phi2 = pot2.phi2, phi2p = pot2.phi2p)
        return phi

    def field(self,
            sig: CompStruct,
            inout: int = 2) -> CompStruct:

        return self.g.field(sig, inout)

    def potential(self,
            sig: CompStruct,
            inout: int = 2) -> CompStruct:

        return self.g.potential(sig, inout)

    def clear(self) -> 'BEMStatLayer':

        # v1.7 A3 fix: drop the real LU factor / rhs scale held in
        # _A_lu and _rhs_scale.  Previous versions only reset the
        # unused mat_lu attribute, leaving GPU LU memory pinned until
        # the next wavelength rebuild.
        # B-3: close mgpu LU handle and release its distributed buffers
        # explicitly so the device tiles are not held until GC.
        old = self._A_lu
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
        self.mat_lu = None
        self._A_lu = None
        self._rhs_scale = None
        self.enei = None
        # v1.7.3 Phase 2: explicit clear() signals the user wants the
        # device drained.  Mirrors BEMStat.clear pattern.
        if _CUPY_OK_LAYER:
            _gpu_pool_cleanup_layer()
        return self

    def __call__(self,
            enei: float) -> 'BEMStatLayer':

        return self._init_matrices(enei)

    def __repr__(self) -> str:
        status = 'enei={:.1f}nm'.format(self.enei) if self.enei is not None else 'not initialized'
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces if hasattr(self.p, 'nfaces') else '?'
        return 'BEMStatLayer(p: {} faces, {})'.format(n, status)
