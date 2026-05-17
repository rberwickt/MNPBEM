import os
import numpy as np
from typing import Optional, List, Tuple, Any, Union

from ..greenfun import CompStruct
from ..greenfun.compgreen_stat_mirror import CompGreenStatMirror
from ..geometry.comparticle_mirror import CompStructMirror
from ..utils.gpu import lu_factor_dispatch, lu_solve_dispatch, to_host, is_cupy_array


# v1.7.3 Phase 2: BEMStatMirror dense LU loop now honours the same
# MNPBEM_VRAM_SHARE_* env vars + cupy pool drain pattern as BEMStat /
# BEMStatLayer.  Mirror solvers tend to allocate ``n_sym`` LU factors per
# wavelength (one per symmetry block), so per-block drains keep the pool
# from accumulating across the loop.
try:
    import cupy as _cp_mirror  # type: ignore
    _CUPY_OK_MIRROR = True
except Exception:
    _cp_mirror = None  # type: ignore
    _CUPY_OK_MIRROR = False


def _vram_share_lu_kwargs() -> dict:
    """Read MNPBEM_VRAM_SHARE_* env vars and return kwargs for
    lu_factor_dispatch.  Mirrors the helper in ``bem_ret.py``."""
    if os.environ.get('MNPBEM_VRAM_SHARE', '0') != '1':
        return {}
    n_gpus = int(os.environ.get('MNPBEM_VRAM_SHARE_GPUS', '1'))
    if n_gpus <= 1:
        return {}
    backend = os.environ.get('MNPBEM_VRAM_SHARE_BACKEND', 'cusolvermg')
    return {'n_gpus': n_gpus, 'backend': backend}


def _vram_share_active() -> bool:
    """Return True iff the distributed-build path should be taken.

    Mirrors ``BEMStat._vram_share_active``.  Off by default; user opts
    in via ``MNPBEM_VRAM_SHARE_DISTRIBUTED=1``.
    """
    if not _CUPY_OK_MIRROR:
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


def _gpu_pool_cleanup_mirror(apply_limit: bool = False) -> None:
    """Synchronise CUDA stream then drain cupy default + pinned pools."""
    if not _CUPY_OK_MIRROR:
        return
    try:
        mempool = _cp_mirror.get_default_memory_pool()
        pinned = _cp_mirror.get_default_pinned_memory_pool()
        if apply_limit:
            try:
                pool_limit_gb = float(os.environ.get(
                        'MNPBEM_GPU_POOL_LIMIT_GB', '0'))
            except (TypeError, ValueError):
                pool_limit_gb = 0.0
            if pool_limit_gb > 0:
                mempool.set_limit(size = int(pool_limit_gb * (1024 ** 3)))
        _cp_mirror.cuda.runtime.deviceSynchronize()
        mempool.free_all_blocks()
        pinned.free_all_blocks()
    except Exception:
        pass


def _mirror_stat_eval_host(g: Any, key: str) -> List:
    """Mirror-symmetry-contracted quasistatic Green block list as host arrays.

    Same audit fix as ``bem_ret_mirror._mirror_eval_host`` for the
    quasistatic Green function: the underlying base attribute (G, F, H1...)
    may be a cupy ndarray under MNPBEM_GPU=1 and the upstream mirror
    contraction skips silently for non-numpy inputs.  Bring it to host
    here so BEMStatMirror always receives populated numpy blocks.
    """
    tab = g.p.symtable
    n_sym = tab.shape[0]
    out: List = [0.0] * n_sym

    mat = getattr(g.g, key)
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


class BEMStatMirror(object):
    """BEM solver for quasistatic approximation with mirror symmetry.

    Given an external excitation, BEMStatMirror computes the surface
    charges such that the boundary conditions of Maxwell's equations
    in the quasistatic approximation are fulfilled.

    MATLAB: @bemstatmirror

    Parameters
    ----------
    p : ComParticleMirror
        Composite particle with mirror symmetry
    enei : float, optional
        Light wavelength in vacuum for pre-initialization
    """

    name = 'bemsolver'
    needs = {'sim': 'stat', 'sym': True}

    def __init__(self,
            p: Any,
            enei: Optional[float] = None,
            **options: Any) -> None:
        self.p = p
        self.enei = None  # type: Optional[float]

        # Green function
        self.g = CompGreenStatMirror(p, p, **options)

        # surface derivative of Green function (list, one per symmetry value).
        # Use the host-promoting wrapper so MNPBEM_GPU=1 (cupy assembly) does
        # not produce a zero list -- see _mirror_stat_eval_host.
        self.F = _mirror_stat_eval_host(self.g, 'F')

        # v1.7.3 Phase 2: drain the cupy pool after the contracted F-block
        # extraction.  Each block went through a host round-trip (asnumpy)
        # so the cupy view of the underlying full F (potentially N^2) can
        # be released here.
        if _CUPY_OK_MIRROR:
            _gpu_pool_cleanup_mirror()

        # resolvent matrices
        self.mat_lu = None  # type: Optional[List]

        if enei is not None:
            self._init_matrices(enei)

    def _init_matrices(self, enei: float) -> 'BEMStatMirror':
        """Initialize matrices for BEM solver.

        MATLAB: @bemstatmirror/subsref.m case '()'
        """
        if self.enei is not None and np.isclose(self.enei, enei):
            return self

        # B-3 distributed multi-GPU build path: each per-symmetry F[i]
        # is fed to a separate cuSolverMg LU through DistributedMatrix
        # so the host never holds the full block of N x N matrices in
        # memory together with the LU buffers.  Falls back to host
        # build on any failure.
        if _vram_share_active():
            try:
                return self._init_distributed_assemble(enei)
            except Exception as e:  # pragma: no cover
                import warnings
                warnings.warn(
                    '[warn] BEMStatMirror distributed assembly failed ({}); '
                    'falling back to host build'.format(e))

        # v1.7.3 Phase 2: free any previous wavelength's LU list before
        # allocating new device-resident factors.  Mirrors the v1.7.2 BEMStat
        # pattern (cupy holds onto old LU buffers until the rebind below).
        self.mat_lu = None
        _gpu_pool_cleanup_mirror(apply_limit = True)

        # inside and outside dielectric function
        eps1 = self.p.eps1(enei)
        eps2 = self.p.eps2(enei)

        # Lambda [Garcia de Abajo, Eq. (23)]
        lambda_diag = 2 * np.pi * (eps1 + eps2) / (eps1 - eps2)

        # VRAM-share kwargs for multi-GPU dispatch on large meshes.
        _lu_opts = _vram_share_lu_kwargs()

        self.mat_lu = []
        for i in range(len(self.F)):
            # BEM resolvent matrix
            M_full = -(np.diag(lambda_diag) + self.F[i])
            self.mat_lu.append(lu_factor_dispatch(M_full, **_lu_opts))
            # v1.7.3 Phase 2: drop the per-block M_full handle so its N^2
            # buffer can return to the pool before the next iteration's
            # GEMM/LU runs.  Important when n_sym >= 2 on large meshes.
            del M_full
            if _CUPY_OK_MIRROR:
                _gpu_pool_cleanup_mirror()

        self.enei = enei
        # v1.7.3 Phase 2: final pool drain so the next wavelength entry sees
        # a clean device.
        _gpu_pool_cleanup_mirror()
        return self

    def _init_distributed_assemble(self,
            enei: float) -> 'BEMStatMirror':
        """B-3 distributed multi-GPU build for the quasistatic mirror solver.

        For each symmetry block the BEM matrix ``M_i = -(diag(Lambda) +
        F[i])`` is built column-tile distributed via
        :class:`mnpbem.utils.distributed_matrix.DistributedMatrix` and
        factored with cuSolverMg.  ``self.F`` is a list of per-symmetry
        N x N host arrays (already produced by
        :func:`_mirror_stat_eval_host` at construction time), so the
        per-tile callback slices each block directly — no extra Green
        function recompute happens here.

        Result residency
        ----------------
        - ``self.mat_lu`` becomes a list of ``('mgpu', MultiGPULU, None)``
          tuples, one per symmetry.  ``_lu_solve_multi`` already routes
          the tag through :func:`lu_solve_dispatch` so downstream
          ``BEMStatMirror.solve`` consumers don't change.
        """

        import gc as _gc
        from ..utils.distributed_matrix import DistributedMatrix
        cp = _cp_mirror

        dist_kw = _vram_share_distributed_kwargs()
        n_gpus = int(dist_kw['n_gpus'])
        device_ids = dist_kw['device_ids']
        block_size = int(dist_kw['block_size'])
        if device_ids is None:
            device_ids = list(range(n_gpus))
        assert len(device_ids) == n_gpus, \
            '[error] MNPBEM_VRAM_SHARE_DEVICE_IDS length must equal MNPBEM_VRAM_SHARE_GPUS'

        # ---- Per-wavelength cleanup: close stale per-symmetry LUs ----
        old_list = getattr(self, 'mat_lu', None)
        if isinstance(old_list, list):
            for entry in old_list:
                if (isinstance(entry, tuple) and len(entry) == 3
                        and entry[0] == 'mgpu' and entry[1] is not None):
                    handle = entry[1]
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
        _gc.collect()
        _gpu_pool_cleanup_mirror(apply_limit = True)

        # ---- eps + Lambda ----
        eps1 = self.p.eps1(enei)
        eps2 = self.p.eps2(enei)
        lambda_diag = (2 * np.pi * (eps1 + eps2) / (eps1 - eps2)).astype(
            np.complex128)

        # ---- Build per-symmetry M_i distributed and factor ----
        self.mat_lu = []
        for i, F_i in enumerate(self.F):
            if not isinstance(F_i, np.ndarray):
                # Skip empty/zero entries (matches legacy guard for
                # symmetry slots with no contribution).
                self.mat_lu.append(None)
                continue
            N_i = int(F_i.shape[0])
            F_host_i = F_i  # already host numpy (from _mirror_stat_eval_host)

            def _make_eval(F_arr, lam):
                def _eval_M_tile(gpu_idx, c0, c1):
                    # Promote real F slice to complex128 so the lambda
                    # diagonal addition does not truncate / warn.
                    block = F_arr[:, c0:c1].astype(np.complex128, copy=True)
                    ncol = c1 - c0
                    for k in range(ncol):
                        j = c0 + k
                        block[j, k] += lam[j]
                    return -block
                return _eval_M_tile

            eval_fn = _make_eval(F_host_i, lambda_diag)
            M_dm = DistributedMatrix.from_func(
                shape=(N_i, N_i),
                dtype=np.complex128,
                n_gpus=n_gpus,
                device_ids=device_ids,
                block_size=block_size,
                eval_func=eval_fn,
            )
            M_mglu = M_dm.lu_factor(backend='cusolvermg')
            M_mglu._distmat_keepalive = M_dm  # type: ignore[attr-defined]
            self.mat_lu.append(('mgpu', M_mglu, None))

            # Drain pool between symmetries so the next iteration's
            # build sees a clean device.
            try:
                for d in device_ids:
                    cp.cuda.runtime.setDevice(d)
                    cp.cuda.runtime.deviceSynchronize()
                    cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

        self.enei = enei
        return self

    def solve(self, exc: CompStructMirror) -> Tuple[CompStructMirror, 'BEMStatMirror']:
        """Surface charge for given excitation.

        MATLAB: @bemstatmirror/mldivide.m

        Parameters
        ----------
        exc : CompStructMirror
            External excitation with field 'phip'

        Returns
        -------
        sig : CompStructMirror
            Surface charge
        obj : BEMStatMirror
            Updated solver
        """
        self._init_matrices(exc.enei)

        sig = CompStructMirror(self.p, exc.enei, getattr(exc, 'fun', None))

        for i in range(len(exc.val)):
            ind = self.p.symindex(exc.val[i].symval[-1, :])

            sig_val = _lu_solve_multi(self.mat_lu[ind], exc.val[i].phip)

            # v1.7 Phase 1.4: host-materialize for user-facing access.
            if is_cupy_array(sig_val):
                sig_val = to_host(sig_val)

            val = CompStruct(self.p, exc.enei, sig = sig_val)
            val.symval = exc.val[i].symval
            sig.val.append(val)

        # v1.7.3 Phase 2: post-solve pool drain (per-symmetry LU back-
        # substitute leaves O(N) scratch in the pool; release before next
        # wavelength's __truediv__ enters).
        if _CUPY_OK_MIRROR:
            _gpu_pool_cleanup_mirror()

        return sig, self

    def __truediv__(self, exc: CompStructMirror) -> Tuple[CompStructMirror, 'BEMStatMirror']:
        return self.solve(exc)

    def __mul__(self, sig: CompStructMirror) -> CompStructMirror:
        """Induced potential for given surface charge.

        MATLAB: @bemstatmirror/mtimes.m
        """
        pot1 = self.potential(sig, 1)
        pot2 = self.potential(sig, 2)

        result = CompStructMirror(self.p, sig.enei, sig.fun)
        for i in range(len(sig.val)):
            combined = CompStruct(self.p, sig.enei)
            for attr in ('phi1', 'phi1p'):
                v = getattr(pot1.val[i], attr, None)
                if v is not None:
                    setattr(combined, attr, v)
            for attr in ('phi2', 'phi2p'):
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

        MATLAB: @bemstatmirror/potential.m
        """
        return self.g.potential(sig, inout)

    def field(self,
            sig: CompStructMirror,
            inout: int = 2) -> CompStructMirror:
        """Electric field inside/outside of particle surface.

        MATLAB: @bemstatmirror/field.m
        """
        return self.g.field(sig, inout)

    def clear(self) -> 'BEMStatMirror':
        """Clear cached LU factors and force rebuild on next solve.

        v1.7.3 Phase 2: API parity with BEMStat / BEMStatLayer / BEMStatIter.
        Drops the per-symmetry LU list and resets the cache gate so a
        subsequent solve() at the same wavelength does not skip rebuild.
        B-3: close mgpu handles + free their distributed buffers.
        """
        old_list = self.mat_lu
        if isinstance(old_list, list):
            for entry in old_list:
                if (isinstance(entry, tuple) and len(entry) == 3
                        and entry[0] == 'mgpu' and entry[1] is not None):
                    handle = entry[1]
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
        self.enei = None
        if _CUPY_OK_MIRROR:
            _gpu_pool_cleanup_mirror()
        return self

    def __call__(self, enei: float) -> 'BEMStatMirror':
        return self._init_matrices(enei)

    def __repr__(self) -> str:
        status = 'enei={}'.format(self.enei) if self.enei is not None else 'not initialized'
        return 'BEMStatMirror(p={}, {})'.format(self.p, status)


def _lu_solve_multi(lu_piv: Tuple, b: Any) -> Any:
    if isinstance(b, np.ndarray):
        if b.ndim == 1:
            return lu_solve_dispatch(lu_piv, b)
        else:
            return lu_solve_dispatch(lu_piv, b.reshape(b.shape[0], -1)).reshape(b.shape)
    return lu_solve_dispatch(lu_piv, np.asarray(b))


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
    if isinstance(a, np.ndarray) and isinstance(x, np.ndarray):
        if x.ndim == 1:
            return a @ x
        elif x.ndim == 2:
            return a @ x
    return a @ x
