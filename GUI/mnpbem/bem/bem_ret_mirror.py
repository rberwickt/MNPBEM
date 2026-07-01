import os
import numpy as np
from typing import Optional, List, Tuple, Any, Dict, Union

from ..greenfun import CompStruct
from ..greenfun.compgreen_ret_mirror import CompGreenRetMirror
from ..geometry.comparticle_mirror import CompStructMirror
from ..utils.gpu import lu_factor_dispatch, lu_solve_dispatch, to_host, is_cupy_array
from .bem_ret import _vram_share_lu_kwargs

# v1.7.3: cupy pool clean-up helpers mirroring bem_ret_layer.py
try:
    import cupy as _cp_v173  # type: ignore
    _CUPY_OK_V173 = True
except Exception:
    _cp_v173 = None  # type: ignore
    _CUPY_OK_V173 = False


def _free_cupy_pool() -> None:
    """deviceSynchronize + free_all_blocks (no-op when cupy unavailable)."""
    if not _CUPY_OK_V173:
        return
    try:
        _cp_v173.cuda.runtime.deviceSynchronize()
        _cp_v173.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# B-3 distributed build gate (Agent E)
# ---------------------------------------------------------------------------
# Same gate semantics as bem_ret_layer.py — defined locally so the two files
# stay self-contained per the integration task.  When ON, every per-symmetry
# block is built as a DistributedMatrix so the per-symmetry LU (and the
# Sigma cache later in ``_init_sigma_lu``) is factored across N GPUs via
# cuSolverMg.  When OFF, the legacy host path runs verbatim.
# ---------------------------------------------------------------------------


def _vram_share_active() -> bool:
    if not _CUPY_OK_V173:
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
    try:
        n_gpus = int(os.environ.get('MNPBEM_VRAM_SHARE_GPUS', '1'))
    except (TypeError, ValueError):
        n_gpus = 1
    device_ids: Optional[List[int]] = None
    dev_env = os.environ.get('MNPBEM_VRAM_SHARE_DEVICE_IDS', '')
    if dev_env.strip():
        try:
            device_ids = [int(x) for x in dev_env.split(',') if x.strip()]
        except Exception:
            device_ids = None
    return {'n_gpus': n_gpus, 'device_ids': device_ids, 'block_size': 256}


def _mirror_eval_host(g: Any,
        i: int,
        j: int,
        key: str,
        enei: float) -> List:
    """Mirror-symmetry-contracted Green block list as host (numpy) arrays.

    Wraps ``CompGreenRetMirror.eval`` so the result is always a list of
    numpy arrays (or scalar zeros) even when ``MNPBEM_GPU=1`` causes the
    underlying base eval to return cupy ndarrays.  The upstream mirror
    ``eval`` skips the contraction silently when ``isinstance(mat, np.ndarray)``
    is False for a cupy ndarray, producing a zero list that hides the GPU
    code path.  v1.7 A4 audit fix: route the contraction through the host.
    """
    tab = g.p.symtable
    n_sym = tab.shape[0]
    out: List = [0.0] * n_sym

    mat = g.g.eval(i, j, key, enei)
    if isinstance(mat, (int, float)) and mat == 0:
        return out
    if is_cupy_array(mat):
        mat = to_host(mat)
    if not isinstance(mat, np.ndarray):
        return out

    if mat.ndim == 2:
        n = mat.shape[0]
        n_blocks = mat.shape[1] // n
        sub_mats = [mat[:, b * n:(b + 1) * n] for b in range(n_blocks)]
        for i_sym in range(n_sym):
            out[i_sym] = np.zeros_like(sub_mats[0])
            for j_block in range(tab.shape[1]):
                out[i_sym] = out[i_sym] + tab[i_sym, j_block] * sub_mats[j_block]
    elif mat.ndim == 3:
        n = mat.shape[0]
        n_blocks = mat.shape[2] // n
        sub_mats = [mat[:, :, b * n:(b + 1) * n] for b in range(n_blocks)]
        for i_sym in range(n_sym):
            out[i_sym] = np.zeros_like(sub_mats[0])
            for j_block in range(tab.shape[1]):
                out[i_sym] = out[i_sym] + tab[i_sym, j_block] * sub_mats[j_block]
    return out


class BEMRetMirror(object):
    """BEM solver for full Maxwell equations with mirror symmetry.

    Given an external excitation, BEMRetMirror computes the surface
    charges such that the boundary conditions of Maxwell's equations
    are fulfilled, exploiting mirror symmetry to reduce computation.

    Reference:
        Garcia de Abajo and Howie, PRB 65, 115418 (2002)

    MATLAB: @bemretmirror

    Parameters
    ----------
    p : ComParticleMirror
        Composite particle with mirror symmetry
    enei : float, optional
        Light wavelength in vacuum for pre-initialization
    """

    name = 'bemsolver'
    needs = {'sim': 'ret', 'sym': True}

    def __init__(self,
            p: Any,
            enei: Optional[float] = None,
            **options: Any) -> None:
        self.p = p
        self.enei = None  # type: Optional[float]

        # BEM matrices (initialized on demand)
        self.k = None  # type: Optional[float]
        self._nvec = None
        self._eps1 = None
        self._eps2 = None
        self._G1_lu = None  # type: Optional[List]
        self._G2_lu = None  # type: Optional[List]
        self._L1 = None  # type: Optional[List]
        self._L2 = None  # type: Optional[List]
        self._Sigma1 = None  # type: Optional[List]
        self._Sigma2 = None  # type: Optional[List]
        self._Delta_lu = None  # type: Optional[List]
        self._Sigma_lu = None  # type: Optional[Any]

        # B-3 (Agent E): distributed buffer caches.  Per-symmetry lists
        # so each symmetry's G1/G2/Delta tiles stay alive for the
        # lifetime of the matching ('mgpu', ...) LU handle.
        self._G1_dm = None  # type: Optional[List]
        self._G2_dm = None  # type: Optional[List]
        self._Delta_dm = None  # type: Optional[List]
        self._Sigma_dm = None  # type: Optional[Dict]

        # Green function
        self.g = CompGreenRetMirror(p, p, **options)

        if enei is not None:
            self.init(enei)

    def init(self, enei: float) -> 'BEMRetMirror':
        """Initialize matrices for BEM solver.

        MATLAB: @bemretmirror/private/initmat.m
        """
        if self.enei is not None and np.isclose(self.enei, enei):
            return self

        # B-3 (Agent E): distributed-build path.
        if _vram_share_active():
            try:
                return self._init_distributed_precond(enei)
            except Exception as _dist_exc:
                import warnings as _w
                _w.warn(
                    '[warn] BEMRetMirror distributed build path failed ({}); '
                    'falling back to legacy host path.'.format(_dist_exc),
                    RuntimeWarning,
                    stacklevel=2,
                )

        # v1.7.3 MATLAB-parity: drop previous wavelength's cached LU /
        # Sigma residues + free cupy pool before allocating the new
        # wavelength's BEM matrices.  Mirrors the BEMRet/BEMRetLayer
        # per-wavelength fragmentation cleanup so the symmetry loop's
        # repeated N^2 inverses do not balloon the pool across a sweep.
        for _attr in ('_G1_lu', '_G2_lu', '_L1', '_L2',
                      '_Sigma1', '_Sigma2', '_Delta_lu', '_Sigma_lu'):
            if hasattr(self, _attr):
                setattr(self, _attr, None)
        if _CUPY_OK_V173:
            try:
                _mempool_pre = _cp_v173.get_default_memory_pool()
                _pinned_pre = _cp_v173.get_default_pinned_memory_pool()
                try:
                    _pool_limit_gb = float(
                        os.environ.get('MNPBEM_GPU_POOL_LIMIT_GB', '0')
                    )
                except (TypeError, ValueError):
                    _pool_limit_gb = 0.0
                if _pool_limit_gb > 0:
                    _mempool_pre.set_limit(size=int(_pool_limit_gb * (1024 ** 3)))
                import gc as _gc
                _gc.collect()
                _cp_v173.cuda.runtime.deviceSynchronize()
                _mempool_pre.free_all_blocks()
                _pinned_pre.free_all_blocks()
            except Exception:
                pass

        self.enei = enei
        self._nvec = self.p.nvec
        self.k = 2 * np.pi / enei

        # dielectric functions
        eps1_vals = self.p.eps1(enei)
        eps2_vals = self.p.eps2(enei)

        if np.allclose(eps1_vals, eps1_vals[0]) and np.allclose(eps2_vals, eps2_vals[0]):
            self._eps1 = eps1_vals[0]
            self._eps2 = eps2_vals[0]
        else:
            self._eps1 = np.diag(eps1_vals)
            self._eps2 = np.diag(eps2_vals)

        # Green functions and surface derivatives for each symmetry value.
        # Use the host-promoting wrapper so MNPBEM_GPU=1 (cupy assembly) does
        # not silently produce a zero list -- see _mirror_eval_host.
        G1_list = _mirror_eval_host(self.g, 0, 0, 'G', enei)
        G1_cross = _mirror_eval_host(self.g, 1, 0, 'G', enei)
        G2_list = _mirror_eval_host(self.g, 1, 1, 'G', enei)
        G2_cross = _mirror_eval_host(self.g, 0, 1, 'G', enei)

        H1_list = _mirror_eval_host(self.g, 0, 0, 'H1', enei)
        H1_cross = _mirror_eval_host(self.g, 1, 0, 'H1', enei)
        H2_list = _mirror_eval_host(self.g, 1, 1, 'H2', enei)
        H2_cross = _mirror_eval_host(self.g, 0, 1, 'H2', enei)

        n_sym = len(G1_list)

        G1 = _subtract_list(G1_list, G1_cross)
        G2 = _subtract_list(G2_list, G2_cross)
        H1 = _subtract_list(H1_list, H1_cross)
        H2 = _subtract_list(H2_list, H2_cross)

        self._G1_lu = []
        self._G2_lu = []
        self._L1 = []
        self._L2 = []
        self._Sigma1 = []
        self._Sigma2 = []
        self._Delta_lu = []

        con_cross_zero = True  # check if cross connectivity is zero
        # MATLAB: if all(obj.g.con{1,2} == 0)

        for i in range(n_sym):
            g1_lu = lu_factor_dispatch(G1[i], **_vram_share_lu_kwargs())
            g2_lu = lu_factor_dispatch(G2[i], **_vram_share_lu_kwargs())
            g1i = lu_solve_dispatch(g1_lu, np.eye(G1[i].shape[0]))
            g2i = lu_solve_dispatch(g2_lu, np.eye(G2[i].shape[0]))
            self._G1_lu.append(g1_lu)
            self._G2_lu.append(g2_lu)

            if con_cross_zero:
                self._L1.append(self._eps1)
                self._L2.append(self._eps2)
            else:
                self._L1.append(G1[i] @ self._eps1 @ g1i if not np.isscalar(self._eps1) else self._eps1)
                self._L2.append(G2[i] @ self._eps2 @ g2i if not np.isscalar(self._eps2) else self._eps2)

            sigma1 = H1[i] @ g1i
            sigma2 = H2[i] @ g2i
            # v1.7.3: per-symmetry intermediates (g1i, g2i are full N^2
            # complex inverses) consume ~16 GB for 12672-face dimers and
            # are no longer needed after Sigma is formed.  Drop them so
            # the device pool can recycle before the next symmetry's
            # LU factor + inverse round.
            if is_cupy_array(sigma1):
                sigma1 = to_host(sigma1)
            if is_cupy_array(sigma2):
                sigma2 = to_host(sigma2)
            self._Sigma1.append(sigma1)
            self._Sigma2.append(sigma2)
            self._Delta_lu.append(lu_factor_dispatch(sigma1 - sigma2, **_vram_share_lu_kwargs()))
            del g1i, g2i, sigma1, sigma2
            _free_cupy_pool()

        # Sigma_lu cache: indexed by (x, y, z) symmetry indices
        n_tab = self.p.symtable.shape[0]
        self._Sigma_lu = {}

        # v1.7.3: final cleanup at wavelength-end — ensure every transient
        # G/H/g1i/g2i buffer has been compacted out of the cupy pool
        # before returning to the caller (mirrors BEMRetLayer pattern).
        del G1_list, G1_cross, G2_list, G2_cross
        del H1_list, H1_cross, H2_list, H2_cross
        del G1, G2, H1, H2
        if _CUPY_OK_V173:
            try:
                _cp_v173.cuda.runtime.deviceSynchronize()
                _cp_v173.get_default_memory_pool().free_all_blocks()
                _cp_v173.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

        return self

    def _init_distributed_precond(self, enei: float) -> 'BEMRetMirror':
        """Distributed BEM build for the mirror-symmetry solver.

        Mirrors :meth:`init` but partitions every per-symmetry dense
        N x N matrix (``G1``, ``G2``, ``Delta = Sigma1 - Sigma2``) across
        N GPUs via :class:`DistributedMatrix`, so the LU factors live in
        block-cyclic column tiles and the host never holds more than a
        single full ``N x N`` matrix at any moment (one for ``Sigma1``
        and one for ``Sigma2`` per symmetry, which is the host-resident
        peak in the legacy path anyway).

        For ``n_sym`` symmetries the function builds and factors three
        N x N matrices per symmetry; the resulting handles are stored
        as ``('mgpu', MultiGPULU, None)`` tuples so :meth:`solve` -- via
        ``_lu_solve_multi`` -> ``lu_solve_dispatch`` -- transparently
        gathers solutions back to host without code changes.

        ``self._Sigma_lu`` (the symmetry-index keyed cache for
        :meth:`_init_sigma_lu`) is left empty here so the user-side
        Sigma combinations are produced lazily during :meth:`solve`,
        same as the legacy path.  When ``MNPBEM_VRAM_SHARE_GPUS >= 2``
        each lazy LU also uses the distributed solver via
        ``_vram_share_lu_kwargs()``.
        """

        from ..utils.distributed_matrix import DistributedMatrix

        cp = _cp_v173
        dist_kw = _vram_share_distributed_kwargs()
        n_gpus = int(dist_kw['n_gpus'])
        device_ids = dist_kw['device_ids']
        block_size = int(dist_kw['block_size'])

        # ---- Per-wavelength cleanup of cached state ----
        # Close any cuSolverMg handles cached from a previous wavelength's
        # init.  This must happen BEFORE the new factor() calls because
        # leaving stale handles around can leave cuSolverMg's per-process
        # context in a state that makes the next ``cusolverMgGetrs`` fail
        # with status 6 (the cached descriptor / grid are still bound to
        # the previous distributed buffers).
        for _attr in ('_G1_lu', '_G2_lu', '_Delta_lu'):
            _lst = getattr(self, _attr, None)
            if _lst is not None:
                for _entry in _lst:
                    if isinstance(_entry, tuple) and len(_entry) == 3 and _entry[0] == 'mgpu':
                        try:
                            _entry[1].close()
                        except Exception:
                            pass
            setattr(self, _attr, None)
        # Sigma_lu cache is a dict of ('mgpu', handle, None) tuples.
        _sig_lu_old = getattr(self, '_Sigma_lu', None)
        if _sig_lu_old is not None:
            try:
                for _entry in _sig_lu_old.values():
                    if isinstance(_entry, tuple) and len(_entry) == 3 and _entry[0] == 'mgpu':
                        try:
                            _entry[1].close()
                        except Exception:
                            pass
            except Exception:
                pass
        for _attr in ('_L1', '_L2', '_Sigma1', '_Sigma2', '_Sigma_lu'):
            if hasattr(self, _attr):
                setattr(self, _attr, None)
        # Release any distributed buffers from the previous wavelength.
        for _attr in ('_G1_dm', '_G2_dm', '_Delta_dm', '_Sigma_dm'):
            _cache = getattr(self, _attr, None)
            if _cache is not None:
                try:
                    if isinstance(_cache, dict):
                        for _dm in _cache.values():
                            if _dm is not None:
                                _dm.free()
                    else:
                        for _dm in _cache:
                            if _dm is not None:
                                _dm.free()
                except Exception:
                    pass
            setattr(self, _attr, None)
        try:
            _pool_limit_gb = float(
                os.environ.get('MNPBEM_GPU_POOL_LIMIT_GB', '0')
            )
        except (TypeError, ValueError):
            _pool_limit_gb = 0.0
        try:
            _mempool = cp.get_default_memory_pool()
            _pinned = cp.get_default_pinned_memory_pool()
            if _pool_limit_gb > 0:
                _mempool.set_limit(size=int(_pool_limit_gb * (1024 ** 3)))
            import gc as _gc
            _gc.collect()
            cp.cuda.runtime.deviceSynchronize()
            _mempool.free_all_blocks()
            _pinned.free_all_blocks()
        except Exception:
            pass

        self.enei = enei
        self._nvec = self.p.nvec
        self.k = 2 * np.pi / enei

        eps1_vals = self.p.eps1(enei)
        eps2_vals = self.p.eps2(enei)
        if np.allclose(eps1_vals, eps1_vals[0]) and np.allclose(eps2_vals, eps2_vals[0]):
            self._eps1 = eps1_vals[0]
            self._eps2 = eps2_vals[0]
        else:
            self._eps1 = np.diag(eps1_vals)
            self._eps2 = np.diag(eps2_vals)

        G1_list = _mirror_eval_host(self.g, 0, 0, 'G', enei)
        G1_cross = _mirror_eval_host(self.g, 1, 0, 'G', enei)
        G2_list = _mirror_eval_host(self.g, 1, 1, 'G', enei)
        G2_cross = _mirror_eval_host(self.g, 0, 1, 'G', enei)

        H1_list = _mirror_eval_host(self.g, 0, 0, 'H1', enei)
        H1_cross = _mirror_eval_host(self.g, 1, 0, 'H1', enei)
        H2_list = _mirror_eval_host(self.g, 1, 1, 'H2', enei)
        H2_cross = _mirror_eval_host(self.g, 0, 1, 'H2', enei)

        G1 = _subtract_list(G1_list, G1_cross)
        G2 = _subtract_list(G2_list, G2_cross)
        H1 = _subtract_list(H1_list, H1_cross)
        H2 = _subtract_list(H2_list, H2_cross)
        del G1_list, G1_cross, G2_list, G2_cross
        del H1_list, H1_cross, H2_list, H2_cross

        n_sym = len(G1)
        self._G1_lu = []
        self._G2_lu = []
        self._L1 = []
        self._L2 = []
        self._Sigma1 = []
        self._Sigma2 = []
        self._Delta_lu = []
        self._G1_dm = []
        self._G2_dm = []
        self._Delta_dm = []

        con_cross_zero = True  # mirror solver assumption (matches legacy)

        for i in range(n_sym):
            # --------- Distributed LU of G1[i] -----------------------
            G1_dm_i = DistributedMatrix.from_host(
                np.ascontiguousarray(G1[i]),
                n_gpus=n_gpus,
                device_ids=device_ids,
                block_size=block_size,
            )
            G1_lu_h = G1_dm_i.lu_factor(backend='cusolvermg')
            self._G1_lu.append(('mgpu', G1_lu_h, None))
            self._G1_dm.append(G1_dm_i)
            # Inverse recovered once (host) so Sigma1 = H1 @ G1i matches
            # the legacy code path bit-for-bit on the GEMM ordering.
            g1i = G1_lu_h.solve(np.eye(G1[i].shape[0], dtype=complex))

            # --------- Distributed LU of G2[i] -----------------------
            G2_dm_i = DistributedMatrix.from_host(
                np.ascontiguousarray(G2[i]),
                n_gpus=n_gpus,
                device_ids=device_ids,
                block_size=block_size,
            )
            G2_lu_h = G2_dm_i.lu_factor(backend='cusolvermg')
            self._G2_lu.append(('mgpu', G2_lu_h, None))
            self._G2_dm.append(G2_dm_i)
            g2i = G2_lu_h.solve(np.eye(G2[i].shape[0], dtype=complex))

            # L1 / L2 (con_cross_zero branch matches legacy)
            if con_cross_zero:
                self._L1.append(self._eps1)
                self._L2.append(self._eps2)
            else:
                # Match the legacy fallback exactly.
                self._L1.append(
                    G1[i] @ self._eps1 @ g1i
                    if not np.isscalar(self._eps1) else self._eps1)
                self._L2.append(
                    G2[i] @ self._eps2 @ g2i
                    if not np.isscalar(self._eps2) else self._eps2)

            # Sigma1 / Sigma2 on host (cheap GEMM, single N^2 alive each).
            sigma1 = H1[i] @ g1i
            sigma2 = H2[i] @ g2i
            if is_cupy_array(sigma1):
                sigma1 = to_host(sigma1)
            if is_cupy_array(sigma2):
                sigma2 = to_host(sigma2)
            self._Sigma1.append(sigma1)
            self._Sigma2.append(sigma2)

            # --------- Distributed LU of Delta = Sigma1 - Sigma2 -----
            Delta_host = np.ascontiguousarray(sigma1 - sigma2)
            Delta_dm_i = DistributedMatrix.from_host(
                Delta_host,
                n_gpus=n_gpus,
                device_ids=device_ids,
                block_size=block_size,
            )
            del Delta_host
            Delta_lu_h = Delta_dm_i.lu_factor(backend='cusolvermg')
            self._Delta_lu.append(('mgpu', Delta_lu_h, None))
            self._Delta_dm.append(Delta_dm_i)

            # Drop the per-symmetry inverses (g1i/g2i are full N^2 each
            # ~ 16 GB on 12672-face dimers); the distributed LU handles
            # own everything we need for the solve.
            del g1i, g2i
            _free_cupy_pool()

        # Sigma_lu cache (populated lazily by _init_sigma_lu); the
        # distributed buffer cache for Sigma keeps tiles alive once the
        # per-symmetry Sigma LU is factored on demand.
        self._Sigma_lu = {}
        self._Sigma_dm = {}

        del G1, G2, H1, H2
        # Sync ALL devices in the distributed grid before returning.
        # cuSolverMg leaves async work queued on each device and the next
        # ``solve()`` (running on a separate handle) can otherwise observe
        # half-written descriptors and fail with status 6.
        try:
            for _dev in (device_ids or list(range(n_gpus))):
                try:
                    cp.cuda.runtime.setDevice(int(_dev))
                    cp.cuda.runtime.deviceSynchronize()
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
            cp.cuda.runtime.setDevice(int((device_ids or [0])[0]))
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
        return self

    def _init_sigma_lu(self, x: int, y: int, z: int) -> Tuple:
        """Initialize Sigma LU factorization for BEM solver (if needed).

        MATLAB: @bemretmirror/private/initsigmai.m
        Eq. (21,22) of Garcia de Abajo and Howie, PRB 65, 115418 (2002).
        """
        key = (x, y, z)
        if key in self._Sigma_lu:
            return self._Sigma_lu[key]

        k = self.k
        nvec = self._nvec
        eps1 = self._eps1
        eps2 = self._eps2

        # outer product of normal components
        def outer_ii(i: int) -> np.ndarray:
            return np.outer(nvec[:, i], nvec[:, i])

        # L = L1 - L2
        L = [None, None, None]
        for dim in range(3):
            idx = [x, y, z][dim]
            if np.isscalar(self._L1[idx]):
                L[dim] = self._L1[idx] - self._L2[idx]
            else:
                L[dim] = self._L1[idx] - self._L2[idx]

        # Sigma = Sigma1_z * L1_z - Sigma2_z * L2_z +
        #         k^2 * sum over i of ((L_i * Deltai_i) .* outer(i)) * L_z
        if np.isscalar(self._L1[z]):
            Sigma = (self._Sigma1[z] * self._L1[z] - self._Sigma2[z] * self._L2[z])
        else:
            Sigma = (self._Sigma1[z] @ self._L1[z] - self._Sigma2[z] @ self._L2[z])

        for dim, idx in enumerate([x, y, z]):
            Deltai_idx = lu_solve_dispatch(self._Delta_lu[idx], np.eye(self._Sigma1[0].shape[0]))
            if np.isscalar(L[dim]):
                term = k ** 2 * (L[dim] * Deltai_idx * outer_ii(dim))
                if np.isscalar(L[2]):
                    Sigma = Sigma + term * L[2]
                else:
                    Sigma = Sigma + term @ L[2]
            else:
                term = k ** 2 * ((L[dim] @ Deltai_idx) * outer_ii(dim))
                Sigma = Sigma + term @ (self._L1[z] - self._L2[z])

        sigma_lu = lu_factor_dispatch(Sigma, **_vram_share_lu_kwargs())
        self._Sigma_lu[key] = sigma_lu
        del Sigma
        _free_cupy_pool()
        return sigma_lu

    def _excitation(self, exc: Any) -> Tuple:
        """Compute excitation variables for BEM solver.

        MATLAB: @bemretmirror/private/excitation.m
        """
        eps1 = self.p.eps1(self.enei)
        eps2 = self.p.eps2(self.enei)
        k = self.k
        nvec = self._nvec

        phi1 = getattr(exc, 'phi1', 0)
        phi1p = getattr(exc, 'phi1p', 0)
        a1 = getattr(exc, 'a1', 0)
        a1p = getattr(exc, 'a1p', 0)
        phi2 = getattr(exc, 'phi2', 0)
        phi2p = getattr(exc, 'phi2p', 0)
        a2 = getattr(exc, 'a2', 0)
        a2p = getattr(exc, 'a2p', 0)

        # Eqs. (10,11)
        phi = _sub(phi2, phi1)
        a = _sub(a2, a1)

        # Eq. (15)
        alpha = (_sub(a2p, a1p)
                 - 1j * k * _sub(_outer_eps(nvec, phi2, eps2), _outer_eps(nvec, phi1, eps1)))

        # Eq. (18)
        De = (_sub(_matmul_diag(eps2, phi2p), _matmul_diag(eps1, phi1p))
              - 1j * k * _sub(_inner_eps(nvec, a2, eps2), _inner_eps(nvec, a1, eps1)))

        return phi, a, alpha, De

    def solve(self, exc: CompStructMirror) -> Tuple[CompStructMirror, 'BEMRetMirror']:
        """Surface charges and currents for given excitation.

        MATLAB: @bemretmirror/mldivide.m

        Parameters
        ----------
        exc : CompStructMirror
            External excitation

        Returns
        -------
        sig : CompStructMirror
            Surface charges and currents
        obj : BEMRetMirror
            Updated solver
        """
        self.init(exc.enei)

        k = self.k
        nvec = self._nvec
        nx, ny, nz = nvec[:, 0], nvec[:, 1], nvec[:, 2]

        sig = CompStructMirror(self.p, exc.enei, exc.fun)

        for i in range(len(exc.val)):
            exc_i = exc.val[i]
            phi, a, alpha, De = self._excitation(exc_i)

            symval = exc_i.symval
            x = self.p.symindex(symval[0, :])
            y = self.p.symindex(symval[1, :])
            z = self.p.symindex(symval[2, :])

            sigma_lu = self._init_sigma_lu(x, y, z)

            # modify alpha and De
            alphax = (_index_vec(alpha, 0)
                      - _matmul(self._Sigma1[x], _index_vec(a, 0))
                      + 1j * k * _matmul_diag_vec(nx, _matmul(self._L1[z], phi)))
            alphay = (_index_vec(alpha, 1)
                      - _matmul(self._Sigma1[y], _index_vec(a, 1))
                      + 1j * k * _matmul_diag_vec(ny, _matmul(self._L1[z], phi)))
            alphaz = (_index_vec(alpha, 2)
                      - _matmul(self._Sigma1[z], _index_vec(a, 2))
                      + 1j * k * _matmul_diag_vec(nz, _matmul(self._L1[z], phi)))

            De_mod = (De
                      - _matmul(self._Sigma1[z], _matmul(self._L1[z], phi))
                      + 1j * k * _matmul_diag_vec(nx, _matmul(self._L1[x], _index_vec(a, 0)))
                      + 1j * k * _matmul_diag_vec(ny, _matmul(self._L1[y], _index_vec(a, 1)))
                      + 1j * k * _matmul_diag_vec(nz, _matmul(self._L1[z], _index_vec(a, 2))))

            # Eq. (19)
            L_diff_x = _scalar_or_mat_sub(self._L1[x], self._L2[x])
            L_diff_y = _scalar_or_mat_sub(self._L1[y], self._L2[y])
            L_diff_z = _scalar_or_mat_sub(self._L1[z], self._L2[z])

            inner_term = (1j * k * (
                _matmul_diag_vec(nx, _matmul(L_diff_x, _lu_solve_multi(self._Delta_lu[x], alphax)))
                + _matmul_diag_vec(ny, _matmul(L_diff_y, _lu_solve_multi(self._Delta_lu[y], alphay)))
                + _matmul_diag_vec(nz, _matmul(L_diff_z, _lu_solve_multi(self._Delta_lu[z], alphaz)))))

            sig2 = _lu_solve_multi(sigma_lu, De_mod + inner_term)

            # Eq. (20)
            h2x = _lu_solve_multi(self._Delta_lu[x],
                1j * k * _matmul_diag_vec(nx, _matmul(L_diff_z, sig2)) + alphax)
            h2y = _lu_solve_multi(self._Delta_lu[y],
                1j * k * _matmul_diag_vec(ny, _matmul(L_diff_z, sig2)) + alphay)
            h2z = _lu_solve_multi(self._Delta_lu[z],
                1j * k * _matmul_diag_vec(nz, _matmul(L_diff_z, sig2)) + alphaz)

            # surface charges and currents
            sig1_val = _lu_solve_multi(self._G1_lu[z], _add(sig2, phi))
            sig2_val = _lu_solve_multi(self._G2_lu[z], sig2)

            h1_val = _vector(
                _lu_solve_multi(self._G1_lu[x], _add(h2x, _index_vec(a, 0))),
                _lu_solve_multi(self._G1_lu[y], _add(h2y, _index_vec(a, 1))),
                _lu_solve_multi(self._G1_lu[z], _add(h2z, _index_vec(a, 2))))

            h2_val = _vector(
                _lu_solve_multi(self._G2_lu[x], h2x),
                _lu_solve_multi(self._G2_lu[y], h2y),
                _lu_solve_multi(self._G2_lu[z], h2z))

            # v1.7 Phase 1.4: host-materialize so user code can call np.asarray.
            if is_cupy_array(sig1_val):
                sig1_val = to_host(sig1_val)
            if is_cupy_array(sig2_val):
                sig2_val = to_host(sig2_val)
            if is_cupy_array(h1_val):
                h1_val = to_host(h1_val)
            if is_cupy_array(h2_val):
                h2_val = to_host(h2_val)

            val = CompStruct(self.p, exc.enei,
                             sig1 = sig1_val, sig2 = sig2_val,
                             h1 = h1_val, h2 = h2_val)
            val.symval = exc_i.symval
            sig.val.append(val)

        # v1.7.3: free per-polarization LU scratch (cupy 2n RHS + GEMMs).
        _free_cupy_pool()

        return sig, self

    def __truediv__(self, exc: CompStructMirror) -> Tuple[CompStructMirror, 'BEMRetMirror']:
        return self.solve(exc)

    def __mul__(self, sig: CompStructMirror) -> CompStructMirror:
        """Induced potential for given surface charge.

        MATLAB: @bemretmirror/mtimes.m
        """
        pot1 = self.potential(sig, 1)
        pot2 = self.potential(sig, 2)

        result = CompStructMirror(self.p, sig.enei, sig.fun)
        for i in range(len(sig.val)):
            combined = CompStruct(self.p, sig.enei)
            for attr in ('phi1', 'phi1p', 'a1', 'a1p'):
                v = getattr(pot1.val[i], attr, None)
                if v is not None:
                    setattr(combined, attr, v)
            for attr in ('phi2', 'phi2p', 'a2', 'a2p'):
                v = getattr(pot2.val[i], attr, None)
                if v is not None:
                    setattr(combined, attr, v)
            combined.symval = sig.val[i].symval
            result.val.append(combined)

        return result

    def potential(self,
            sig: CompStructMirror,
            inout: int = 2) -> CompStructMirror:
        """Potentials and surface derivatives inside/outside of particle.

        MATLAB: @bemretmirror/potential.m
        """
        return self.g.potential(sig, inout)

    def field(self,
            sig: CompStructMirror,
            inout: int = 2) -> CompStructMirror:
        """Electric and magnetic field inside/outside of particle surface.

        MATLAB: @bemretmirror/field.m
        """
        return self.g.field(sig, inout)

    def __call__(self, enei: float) -> 'BEMRetMirror':
        return self.init(enei)

    def __repr__(self) -> str:
        status = 'enei={}'.format(self.enei) if self.enei is not None else 'not initialized'
        return 'BEMRetMirror(p={}, {})'.format(self.p, status)


# ==================== Helper functions ====================

def _lu_solve_multi(lu_piv: Tuple, b: Any) -> Any:
    if isinstance(b, (int, float)) and b == 0:
        return 0
    if isinstance(b, np.ndarray):
        if b.ndim == 1:
            return lu_solve_dispatch(lu_piv, b)
        else:
            return lu_solve_dispatch(lu_piv, b.reshape(b.shape[0], -1)).reshape(b.shape)
    return lu_solve_dispatch(lu_piv, np.asarray(b))


def _subtract_list(a_list: List, b_list: List) -> List:
    """Subtract two lists of matrices element-wise."""
    result = []
    for a, b in zip(a_list, b_list):
        if isinstance(a, (int, float)) and a == 0:
            if isinstance(b, (int, float)) and b == 0:
                result.append(0)
            else:
                result.append(-b)
        elif isinstance(b, (int, float)) and b == 0:
            result.append(a)
        else:
            result.append(a - b)
    return result


def _matmul(a: Any, x: Any) -> Any:
    if isinstance(a, (int, float)):
        if a == 0:
            return 0
        return a * x
    if isinstance(x, (int, float)):
        if x == 0:
            return 0
        return a * x
    if np.isscalar(a):
        return a * x
    return a @ x


def _sub(a: Any, b: Any) -> Any:
    if isinstance(a, (int, float)) and a == 0:
        if isinstance(b, (int, float)) and b == 0:
            return 0
        return -b
    if isinstance(b, (int, float)) and b == 0:
        return a
    return a - b


def _add(a: Any, b: Any) -> Any:
    if isinstance(a, (int, float)) and a == 0:
        return b
    if isinstance(b, (int, float)) and b == 0:
        return a
    return a + b


def _scalar_or_mat_sub(a: Any, b: Any) -> Any:
    if np.isscalar(a) and np.isscalar(b):
        return a - b
    if np.isscalar(a):
        return a - b
    return a - b


def _index_vec(v: Any, ind: int) -> Any:
    """Extract component from vector.

    MATLAB: index(v, ind) in bemretmirror/mldivide.m
    """
    if isinstance(v, (int, float)) and v == 0:
        return 0
    if isinstance(v, np.ndarray):
        if v.ndim == 2:
            return v[:, ind]
        elif v.ndim == 3:
            return v[:, ind, :]
    return v


def _vector(vx: Any, vy: Any, vz: Any) -> np.ndarray:
    """Combine components to vector.

    MATLAB: vector(vx, vy, vz) in bemretmirror/mldivide.m
    """
    if isinstance(vx, np.ndarray):
        if vx.ndim == 1:
            n = vx.shape[0]
            result = np.empty((n, 3), dtype = vx.dtype)
            result[:, 0] = vx
            result[:, 1] = vy
            result[:, 2] = vz
            return result
        elif vx.ndim == 2:
            n = vx.shape[0]
            npol = vx.shape[1]
            result = np.empty((n, 3, npol), dtype = vx.dtype)
            result[:, 0, :] = vx
            result[:, 1, :] = vy
            result[:, 2, :] = vz
            return result
    return np.array([vx, vy, vz])


def _outer_eps(nvec: np.ndarray, phi: Any, eps: np.ndarray) -> Any:
    """Compute outer(nvec, phi) * eps."""
    if isinstance(phi, (int, float)) and phi == 0:
        return 0
    if isinstance(phi, np.ndarray):
        if phi.ndim == 1:
            return nvec * (phi * eps)[:, np.newaxis]
        else:
            npol = phi.shape[1]
            n = nvec.shape[0]
            result = np.empty((n, 3, npol), dtype = complex)
            for ipol in range(npol):
                result[:, :, ipol] = nvec * (phi[:, ipol] * eps)[:, np.newaxis]
            return result
    return 0


def _inner_eps(nvec: np.ndarray, a: Any, eps: np.ndarray) -> Any:
    """Compute inner(nvec, a) * eps."""
    if isinstance(a, (int, float)) and a == 0:
        return 0
    if isinstance(a, np.ndarray) and a.ndim >= 2:
        if a.ndim == 2:
            dot = np.sum(nvec * a, axis = 1)
            return dot * eps
        elif a.ndim == 3:
            npol = a.shape[2]
            n = nvec.shape[0]
            result = np.empty((n, npol), dtype = complex)
            for ipol in range(npol):
                dot = np.sum(nvec * a[:, :, ipol], axis = 1)
                result[:, ipol] = dot * eps
            return result
    return 0


def _matmul_diag(eps: Any, phi_p: Any) -> Any:
    """Compute eps * phi_p (element-wise for diagonal eps)."""
    if isinstance(phi_p, (int, float)) and phi_p == 0:
        return 0
    if isinstance(phi_p, np.ndarray):
        if isinstance(eps, np.ndarray) and eps.ndim == 1:
            if phi_p.ndim == 1:
                return eps * phi_p
            else:
                return eps[:, np.newaxis] * phi_p
        else:
            return eps * phi_p
    return 0


def _matmul_diag_vec(n_comp: np.ndarray, val: Any) -> Any:
    """Compute matmul(n_comp, val) where n_comp is diagonal-like.

    MATLAB: matmul(nx, val) where nx is a component of nvec.
    """
    if isinstance(val, (int, float)) and val == 0:
        return 0
    if isinstance(val, np.ndarray):
        if val.ndim == 1:
            return n_comp * val
        elif val.ndim == 2:
            return n_comp[:, np.newaxis] * val
    return n_comp * val
