import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from ..utils.gpu import lu_factor_dispatch, lu_solve_dispatch, matmul_dispatch, to_host, is_cupy_array

from ..greenfun import CompGreenRetLayer, CompStruct
from .bem_ret import _vram_share_lu_kwargs


# v1.7.2 memory-pool parity: when cupy is importable and MNPBEM_GPU=1 the
# matrix-assembly routines below dispatch GEMM/LU through cupy via the
# *_dispatch helpers in mnpbem.utils.gpu.  Without an explicit pool
# free_all_blocks() between wavelengths, cupy's caching allocator keeps
# every Sigma/Gamma/m_full residue alive across the sweep — on 12672-face
# Au@Ag dimers this fragments past the 49 GB cap around wl 20-25.  Mirror
# the BEMRet._init_gpu_assemble v1.7.2 pattern: deviceSynchronize then
# free_all_blocks at every reasonable boundary (start-of-wl, after each
# large GEMM/LU group, and at function exit).  Implemented inline at
# each call site (not via a helper) so static grep can audit coverage
# the same way it does for ``bem_ret.py`` — see the MNPBEM_GPU_POOL_LIMIT_GB
# documentation and the BEMRet v1.7.2 commentary for the rationale.
try:
    import cupy as _cp_v172  # type: ignore
    _CUPY_OK_V172 = True
except Exception:
    _cp_v172 = None  # type: ignore
    _CUPY_OK_V172 = False


# ---------------------------------------------------------------------------
# B-3 distributed build gate (Agent E)
# ---------------------------------------------------------------------------
# When ``MNPBEM_VRAM_SHARE_DISTRIBUTED=1`` plus ``MNPBEM_VRAM_SHARE=1`` and
# ``MNPBEM_VRAM_SHARE_GPUS >= 2`` are set, the layer-substrate BEM build
# routes the dense G/H/Sigma/m_full matrices through ``DistributedMatrix``
# so that no N^2 buffer ever lives in pinned host memory simultaneously
# across N GPUs -- each device only ever holds its own column tile.  The
# LU factors (``G1_lu``, ``G2p_lu``, ``Gamma_lu``, ``m_lu``) are produced
# by ``DistributedMatrix.lu_factor()`` (cuSolverMg) and carry the existing
# ``('mgpu', ...)`` tag that ``lu_solve_dispatch`` already understands, so
# the per-wavelength solve path stays the same.
#
# When the gate is OFF (default) the legacy single-GPU/host path runs
# verbatim -- this keeps every regression scenario bit-identical.
# ---------------------------------------------------------------------------


def _vram_share_active() -> bool:
    """Return True iff the distributed-build path should be taken.

    Distinct from ``_vram_share_lu_kwargs``: that helper governs whether
    the LU factor is dispatched to cuSolverMg, while this gate controls
    whether the *build* (Green-function assembly, GEMMs, scatter) also
    runs distributed.  Both gates are read separately so a user who only
    wants the LU split can leave ``MNPBEM_VRAM_SHARE_DISTRIBUTED=0`` and
    keep the legacy host build.
    """
    if not _CUPY_OK_V172:
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
    # cuSolverMg must be loadable for the distributed LU factor to run.
    try:
        from ..utils.multi_gpu_lu import cusolvermg_available
        if not cusolvermg_available():
            return False
    except Exception:
        return False
    return True


def _vram_share_distributed_kwargs() -> dict:
    """Return ``{n_gpus, device_ids, block_size}`` for DistributedMatrix.

    Reads the same ``MNPBEM_VRAM_SHARE_*`` env vars as
    ``_vram_share_lu_kwargs`` so the block-cyclic layout matches what
    ``lu_factor_dispatch(n_gpus=N)`` would use internally.  Block size is
    fixed at 256 to align with the cuSolverMg samples and the
    ``DistributedMatrix`` default.
    """
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


# ---------------------------------------------------------------------------
# Backend alignment helper for cupy/numpy mix safety (v1.6.5 fix)
# ---------------------------------------------------------------------------

def _is_cupy_array(x: Any) -> bool:
    """Return True if x is a cupy ndarray."""
    if not hasattr(x, '__class__'):
        return False
    return 'cupy' in type(x).__module__ and hasattr(x, 'shape')


def _backend_align(A: Any, B: Any) -> Tuple[Any, Any]:
    """Return (A, B) on the same backend (cupy or numpy).

    If one is cupy ndarray and the other is numpy ndarray, promote the
    numpy one to cupy to keep GPU residency. Scalars and non-array values
    are returned untouched.
    """
    a_is_cp = _is_cupy_array(A)
    b_is_cp = _is_cupy_array(B)
    if a_is_cp and not b_is_cp and isinstance(B, np.ndarray):
        import cupy as cp
        return A, cp.asarray(B)
    if b_is_cp and not a_is_cp and isinstance(A, np.ndarray):
        import cupy as cp
        return cp.asarray(A), B
    return A, B


def _to_host_safe(x: Any) -> Any:
    """Materialise x on the host as numpy array.

    Cupy ndarrays are converted via cp.asnumpy.  Non-array types
    (scalars, dicts, etc.) are returned unchanged.
    """
    if _is_cupy_array(x):
        import cupy as cp
        return cp.asnumpy(x)
    return x


# ---------------------------------------------------------------------------
# Helper functions matching MATLAB inner/outer/matmul for bemretlayer
# ---------------------------------------------------------------------------

def _inner(nvec, a):
    # MATLAB: inner(nvec, a) — dot product of nvec (n,3) with a (n,3) or (n,3,npol)
    if not isinstance(a, np.ndarray):
        return 0
    if a.ndim == 2:
        # (n, 3) -> (n,)
        return np.sum(nvec * a, axis = 1)
    else:
        # (n, 3, npol) -> (n, npol)
        return np.einsum('ij,ijk->ik', nvec, a)


def _outer(nvec, val):
    # MATLAB: outer(nvec, val) — nvec (n,3) * val (n,) or (n,npol) -> (n,3) or (n,3,npol)
    if not isinstance(val, np.ndarray):
        if val == 0:
            return 0
        return nvec * val
    if val.ndim == 1:
        # (n,) -> (n, 3)
        return nvec * val[:, np.newaxis]
    else:
        # (n, npol) -> (n, 3, npol)
        return nvec[:, :, np.newaxis] * val[:, np.newaxis, :]


def _matmul(M, x):
    # MATLAB: matmul(M, x) — M can be scalar or (n,n), x can be scalar/1D/2D/3D
    if not isinstance(x, np.ndarray):
        if x == 0:
            return 0
        if np.isscalar(M):
            return M * x
        return M * x

    if np.isscalar(M):
        return M * x

    # M is (n, n), x can be (n,), (n, 3), (n, npol), (n, 3, npol)
    if x.ndim == 1:
        return M @ x
    elif x.ndim == 2:
        # (n, n) @ (n, cols) for each column
        return M @ x
    else:
        # (n, 3, npol): apply M to each (n,) slice
        shape = x.shape
        return (M @ x.reshape(shape[0], -1)).reshape(shape)


class BEMRetLayer(object):

    name = 'bemsolver'
    needs = {'sim': 'ret'}

    def __init__(self,
            p: Any,
            layer: Any,
            enei: Optional[float] = None,
            greentab: Optional[Any] = None,
            **options: Any) -> None:

        self.p = p
        self.layer = layer
        self.greentab = greentab

        self.enei = None
        self.k = None
        self.nvec = None
        self.npar = None
        self.eps1 = None
        self.eps2 = None

        # BEM matrices (MATLAB initmat.m variables)
        self.L1 = None
        self.L2p = None
        self.G1i = None
        self.G2pi = None
        self.G2 = None
        self.G2e = None
        self.Sigma1 = None
        self.Sigma1e = None
        self.Gamma = None
        self.m_lu = None
        self.m_full = None

        # LU factorizations
        self._G1_lu = None
        self._G2p_lu = None
        self._Gamma_lu = None

        # Green function with layer
        self.g = None
        self.options = options

        # Wave 66: opt-in MATLAB Engine route for the dense linear solves of
        # the 2n x 2n block matrix.  Default False keeps numpy lu_factor /
        # lu_solve; True delegates the matrix solve to MATLAB's mldivide,
        # eliminating LU/solve numerical drift versus the MATLAB reference.
        self.use_matlab_engine = options.get('use_matlab_engine', False)

        if enei is not None:
            self.init(enei)

    def init(self,
            enei: float) -> 'BEMRetLayer':

        if self.enei is not None and np.isclose(self.enei, enei):
            return self

        # B-3 (Agent E): when the distributed-build gate is on, route
        # through the cuSolverMg + DistributedMatrix assembly so the
        # ``m_full`` 2n x 2n dense block never has to fit on a single
        # device.  Falls back to the legacy host path on any unexpected
        # failure so existing tests stay green.
        if not self.use_matlab_engine and _vram_share_active():
            try:
                return self._init_distributed_precond(enei)
            except Exception as _dist_exc:
                import warnings as _w
                _w.warn(
                    '[warn] BEMRetLayer distributed build path failed ({}); '
                    'falling back to legacy host path.'.format(_dist_exc),
                    RuntimeWarning,
                    stacklevel=2,
                )

        # v1.7.2 MATLAB-parity: free cupy pools before allocating the new
        # wavelength's BEM matrices.  The previous wavelength's cached
        # LU/Sigma residues (potentially ~10-20 GB on large-substrate dimers
        # when *_dispatch routes through cupy) would otherwise stay pinned
        # until Python rebinds the attribute mid-routine, causing a steady
        # pool growth and eventual OOM 20+ wavelengths into a sweep.  Also
        # drop the cached LU/Sigma references up front so free_all_blocks
        # actually returns the device buffers.
        for _attr in ('_G1_lu', '_G2p_lu', '_Gamma_lu', 'm_lu',
                      'G1i', 'G2pi', 'G2', 'G2e',
                      'L1', 'L2p', 'Sigma1', 'Sigma1e', 'Gamma',
                      'm_full'):
            if hasattr(self, _attr):
                setattr(self, _attr, None)
        if _CUPY_OK_V172:
            try:
                _mempool_pre = _cp_v172.get_default_memory_pool()
                _pinned_pre = _cp_v172.get_default_pinned_memory_pool()
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
                _cp_v172.cuda.runtime.deviceSynchronize()
                _mempool_pre.free_all_blocks()
                _pinned_pre.free_all_blocks()
            except Exception:
                pass

        self.enei = enei

        # Outer surface normals
        nvec = self.p.nvec
        self.nvec = nvec

        # Perpendicular and parallel component of normal vector
        # MATLAB: nperp = nvec(:,3);  npar = nvec - nperp * [0,0,1];
        nperp = nvec[:, 2]
        npar = nvec.copy()
        npar[:, 2] = 0.0
        self.npar = npar
        self.nperp = nperp

        # Wavenumber in vacuum
        k = 2 * np.pi / enei
        self.k = k

        # Dielectric function values
        eps1_vals = self.p.eps1(enei)
        eps2_vals = self.p.eps2(enei)

        if np.allclose(eps1_vals, eps1_vals[0]) and np.allclose(eps2_vals, eps2_vals[0]):
            eps1 = eps1_vals[0]
            eps2 = eps2_vals[0]
        else:
            eps1 = np.diag(eps1_vals)
            eps2 = np.diag(eps2_vals)

        self.eps1 = eps1
        self.eps2 = eps2

        # Create Green function with layer
        if self.g is None:
            opts = dict(self.options)
            if self.greentab is not None:
                # Pass the tabulated Green function's GreenTabLayer
                gt = self.greentab
                if hasattr(gt, 'tab'):
                    # CompGreenTabLayer object - extract its GreenTabLayer
                    opts['greentab_obj'] = gt.tab
                elif hasattr(gt, 'r'):
                    # Direct GreenTabLayer object
                    opts['greentab_obj'] = gt
            self.g = CompGreenRetLayer(self.p, self.p, self.layer, **opts)

        # ---- Green functions for inner surfaces (plain scalar matrices) ----
        # MATLAB: G11 = obj.g{1,1}.G(enei);  G21 = obj.g{2,1}.G(enei);
        G11 = self.g.eval(0, 0, 'G', enei)
        G21 = self.g.eval(1, 0, 'G', enei)
        H11 = self.g.eval(0, 0, 'H1', enei)
        H21 = self.g.eval(1, 0, 'H1', enei)

        # Mixed contributions (plain matrices)
        # MATLAB: G1 = G11 - G21;  G1e = eps1 * G11 - eps2 * G21;
        G1 = self._sub_mat(G11, G21)
        G1e = self._sub_mat(self._mul_eps(eps1, G11), self._mul_eps(eps2, G21))
        H1 = self._sub_mat(H11, H21)
        H1e = self._sub_mat(self._mul_eps(eps1, H11), self._mul_eps(eps2, H21))
        # v1.7.2: release inner-surface Green intermediates as soon as the
        # combined G1/G1e/H1/H1e are formed.  Mirrors BEMRet's per-stage
        # del + free_all_blocks pattern.
        del G11, G21, H11, H21
        if _CUPY_OK_V172:
            try:
                _cp_v172.cuda.runtime.deviceSynchronize()
                _cp_v172.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

        # ---- Green functions for outer surfaces (structured dict) ----
        # MATLAB: G22 = obj.g{2,2}.G(enei) -> structured {ss,hh,p,sh,hs}
        #         G12 = obj.g{1,2}.G(enei) -> plain scalar
        G22 = self.g.eval(1, 1, 'G', enei)
        G12 = self.g.eval(0, 1, 'G', enei)
        H22 = self.g.eval(1, 1, 'H2', enei)
        H12 = self.g.eval(0, 1, 'H2', enei)

        # Build G2 structured dict: G2.ss = G22.ss - G12, etc.
        G2 = self._build_outer_mixed(G22, G12)
        H2 = self._build_outer_mixed(H22, H12)

        # Build G2e structured dict: G2e.ss = eps2*G22.ss - eps1*G12, etc.
        G2e = self._build_outer_mixed_eps(G22, G12, eps2, eps1)
        H2e = self._build_outer_mixed_eps(H22, H12, eps2, eps1)
        # v1.7.2: release outer-surface Green intermediates after the
        # structured G2/G2e/H2/H2e dicts have been built.  G22 is a dict
        # of N^2 substrate-table evaluations (~5 buffers for an outer
        # block); freeing it now keeps the pool from carrying double the
        # peak through the LU factorizations below.
        del G22, G12, H22, H12
        if _CUPY_OK_V172:
            try:
                _cp_v172.cuda.runtime.deviceSynchronize()
                _cp_v172.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

        n = G1.shape[0]

        if self.use_matlab_engine:
            # ---- Wave 67: delegate the entire BEM matrix construction
            # (initmat.m sequence) to MATLAB to inherit MATLAB's exact BLAS
            # ordering and rounding behavior on each matmul/inv. ----
            from .matlab_bem import matlab_bem_init

            eps1_diag = (np.full(n, eps1, dtype=complex) if np.isscalar(eps1)
                         else np.asarray(np.diag(eps1) if eps1.ndim == 2 else eps1, dtype=complex))
            eps2_diag = (np.full(n, eps2, dtype=complex) if np.isscalar(eps2)
                         else np.asarray(np.diag(eps2) if eps2.ndim == 2 else eps2, dtype=complex))

            ml_in_G22 = G22 if isinstance(G22, dict) else {'ss': G22, 'hh': G22, 'p': G22}
            ml_in_H22 = H22 if isinstance(H22, dict) else {'ss': H22, 'hh': H22, 'p': H22}

            mout = matlab_bem_init(
                G11, G21 if isinstance(G21, np.ndarray) else np.zeros((n, n), dtype=complex),
                H11, H21 if isinstance(H21, np.ndarray) else np.zeros((n, n), dtype=complex),
                ml_in_G22, G12 if isinstance(G12, np.ndarray) else np.zeros((n, n), dtype=complex),
                ml_in_H22, H12 if isinstance(H12, np.ndarray) else np.zeros((n, n), dtype=complex),
                eps1_diag, eps2_diag, k, nvec)

            G1 = mout['G1']
            G1i = mout['G1i']
            G2pi = mout['G2pi']
            G2 = {
                'ss': mout['G2_ss'], 'hh': mout['G2_hh'], 'p': mout['G2_p'],
                'sh': mout['G2_sh'], 'hs': mout['G2_hs'],
            }
            G2e = {
                'ss': mout['G2e_ss'], 'hh': mout['G2e_hh'], 'p': mout['G2e_p'],
                'sh': mout['G2e_sh'], 'hs': mout['G2e_hs'],
            }
            Sigma1 = mout['Sigma1']
            Sigma1e = mout['Sigma1e']
            L1 = mout['L1']
            L2p = mout['L2p']
            Gamma = mout['Gamma']
            m_full = mout['m_full']

            self.m_full = m_full
            self.m_lu = None
            self._G1_lu = None
            self._G2p_lu = None
            self._Gamma_lu = None
        else:
            # ---- Auxiliary matrices (MATLAB initmat.m lines 51-68) ----
            # Inverse of G1 and of parallel component G2.p
            self._G1_lu = lu_factor_dispatch(G1, **_vram_share_lu_kwargs())
            G1i = lu_solve_dispatch(self._G1_lu, np.eye(G1.shape[0]))

            self._G2p_lu = lu_factor_dispatch(G2['p'], **_vram_share_lu_kwargs())
            G2pi = lu_solve_dispatch(self._G2p_lu, np.eye(G2['p'].shape[0]))
            # v1.7.2: free pools after the two G LU factorizations + their
            # inverses (each LU is ~N^2 complex; the eye-product N^2 too).
            if _CUPY_OK_V172:
                try:
                    _cp_v172.cuda.runtime.deviceSynchronize()
                    _cp_v172.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass

            # Sigma matrices [Eq.(21)]
            Sigma1 = matmul_dispatch(H1, G1i)
            Sigma1e = matmul_dispatch(H1e, G1i)
            Sigma2p = matmul_dispatch(H2['p'], G2pi)

            # Auxiliary dielectric function matrices
            L1 = matmul_dispatch(G1e, G1i)
            L2p = matmul_dispatch(G2e['p'], G2pi)
            # v1.7.2: G1e/H1/H1e and the H/H1/G part of H2 are no longer
            # needed after Sigma/L are formed.  Drop the local refs so the
            # cupy pool can recycle them before the m_full GEMMs below.
            del H1, H1e, G1e
            if _CUPY_OK_V172:
                try:
                    _cp_v172.cuda.runtime.deviceSynchronize()
                    _cp_v172.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass

            # Gamma matrix
            self._Gamma_lu = lu_factor_dispatch(Sigma1 - Sigma2p, **_vram_share_lu_kwargs())
            Gamma = lu_solve_dispatch(self._Gamma_lu, np.eye(Sigma1.shape[0]))
            del Sigma2p
            if _CUPY_OK_V172:
                try:
                    _cp_v172.cuda.runtime.deviceSynchronize()
                    _cp_v172.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass

            # Gammapar = ik*(L1-L2p)*Gamma .* (npar*npar')
            # Element-wise multiply with outer product of parallel normals
            npar_outer = npar @ npar.T  # (n, n)
            Gammapar = 1j * k * matmul_dispatch(L1 - L2p, Gamma) * npar_outer
            del npar_outer

            # ---- Set up 2x2 block response matrix (MATLAB initmat.m lines 72-77) ----
            # m{1,1} = Sigma1e*G2.ss - H2e.ss - ik*(Gammapar*(L1*G2.ss - G2e.ss)
            #          + bsxfun(@times, L1*G2.sh - G2e.sh, nperp))
            diff_ss = matmul_dispatch(L1, G2['ss']) - G2e['ss']
            diff_sh = matmul_dispatch(L1, G2['sh']) - G2e['sh']
            diff_hh = matmul_dispatch(L1, G2['hh']) - G2e['hh']

            m11 = (matmul_dispatch(Sigma1e, G2['ss']) - H2e['ss']
                - 1j * k * (matmul_dispatch(Gammapar, diff_ss) + diff_sh * nperp[:, np.newaxis]))
            m12 = (matmul_dispatch(Sigma1e, G2['sh']) - H2e['sh']
                - 1j * k * (matmul_dispatch(Gammapar, diff_sh) + diff_hh * nperp[:, np.newaxis]))
            m21 = (matmul_dispatch(Sigma1, G2['hs']) - H2['hs']
                - 1j * k * diff_ss * nperp[:, np.newaxis])
            m22 = (matmul_dispatch(Sigma1, G2['hh']) - H2['hh']
                - 1j * k * diff_sh * nperp[:, np.newaxis])
            # v1.7.3 (VRAM-share path): matmul_dispatch may return cupy
            # arrays under MNPBEM_GPU=1.  Materialise the m11..m22 blocks
            # on the host before assembling m_full so the 4*N^2 buffer
            # lives in pinned host memory (the cuSolverMg LU input is
            # uploaded internally per-tile).  This also lets the per-block
            # device buffers be released before the LU factor below.
            if is_cupy_array(m11):
                m11 = to_host(m11)
            if is_cupy_array(m12):
                m12 = to_host(m12)
            if is_cupy_array(m21):
                m21 = to_host(m21)
            if is_cupy_array(m22):
                m22 = to_host(m22)
            # v1.7.2: diff_* and Gammapar are consumed by the m11..m22
            # forms; drop them so the m_full assemble below sees max
            # headroom.  H2/H2e remain only via their already-formed terms.
            del diff_ss, diff_sh, diff_hh, Gammapar, H2, H2e
            if _CUPY_OK_V172:
                try:
                    _cp_v172.cuda.runtime.deviceSynchronize()
                    _cp_v172.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass

            # Assemble 2x2 block matrix (2n x 2n) and LU factorize
            m_full = np.empty((2 * n, 2 * n), dtype = complex)
            m_full[:n, :n] = m11
            m_full[:n, n:] = m12
            m_full[n:, :n] = m21
            m_full[n:, n:] = m22
            del m11, m12, m21, m22
            if _CUPY_OK_V172:
                try:
                    _cp_v172.cuda.runtime.deviceSynchronize()
                    _cp_v172.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass

            self.m_full = None
            self.m_lu = lu_factor_dispatch(m_full, **_vram_share_lu_kwargs())
            # v1.7.2: the dense m_full (2n x 2n) is now captured by the LU
            # factor; drop the local handle so its 4*N^2 complex buffer
            # returns to the pool before we exit init().
            del m_full
            if _CUPY_OK_V172:
                try:
                    _cp_v172.cuda.runtime.deviceSynchronize()
                    _cp_v172.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass

        # Store all needed matrices.  v1.7.3 (VRAM-share path): when the
        # m_full LU factor is owned by cuSolverMg (multi-GPU), keeping
        # G1i/G2pi/L1/L2p/Sigma1/Sigma1e/Gamma on device merely doubles
        # the auxiliary-matrix footprint without benefit — solve()/
        # _solve_single uses matmul_dispatch which uploads on demand.
        # Materialise on host so the device pool can be compacted before
        # the per-wavelength solve loop.
        if is_cupy_array(G1i):
            G1i = to_host(G1i)
        if is_cupy_array(G2pi):
            G2pi = to_host(G2pi)
        if is_cupy_array(L1):
            L1 = to_host(L1)
        if is_cupy_array(L2p):
            L2p = to_host(L2p)
        if is_cupy_array(Sigma1):
            Sigma1 = to_host(Sigma1)
        if is_cupy_array(Sigma1e):
            Sigma1e = to_host(Sigma1e)
        if is_cupy_array(Gamma):
            Gamma = to_host(Gamma)
        if isinstance(G2, dict):
            for _k in list(G2.keys()):
                if is_cupy_array(G2[_k]):
                    G2[_k] = to_host(G2[_k])
        if isinstance(G2e, dict):
            for _k in list(G2e.keys()):
                if is_cupy_array(G2e[_k]):
                    G2e[_k] = to_host(G2e[_k])

        self.G1i = G1i
        self.G2pi = G2pi
        self.G2 = G2
        self.G2e = G2e
        self.L1 = L1
        self.L2p = L2p
        self.Sigma1 = Sigma1
        self.Sigma1e = Sigma1e
        self.Gamma = Gamma

        # v1.7.2 final cleanup at wavelength-end: ensure every transient
        # LU/Sigma/Gamma allocation has been compacted out of the cupy
        # pool before the BEM solver returns to the caller.  Mirrors
        # BEMRet._init_gpu_assemble's closing free_all_blocks pair.
        if _CUPY_OK_V172:
            try:
                _cp_v172.cuda.runtime.deviceSynchronize()
                _cp_v172.get_default_memory_pool().free_all_blocks()
                _cp_v172.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

        return self

    def _sub_mat(self,
            A: Any,
            B: Any) -> Any:
        if isinstance(B, (int, float)) and B == 0:
            return _to_host_safe(A)
        if isinstance(A, (int, float)) and A == 0:
            return _to_host_safe(-B if not _is_cupy_array(B) else -B)
        A, B = _backend_align(A, B)
        result = A - B
        return _to_host_safe(result)

    def _mul_eps(self,
            eps: Any,
            M: Any) -> Any:
        if isinstance(M, (int, float)) and M == 0:
            return 0
        if np.isscalar(eps):
            return _to_host_safe(eps * M)
        eps, M = _backend_align(eps, M)
        result = eps @ M
        return _to_host_safe(result)

    def _build_outer_mixed(self,
            G_struct: Any,
            G_plain: Any) -> Dict[str, Any]:
        # MATLAB: G2.ss = G22.ss - G12;  G2.hh = G22.hh - G12;  G2.p = G22.p - G12;
        #         G2.sh = G22.sh;  G2.hs = G22.hs;
        if isinstance(G_struct, dict):
            result = {}
            for key in ('ss', 'hh', 'p'):
                result[key] = self._sub_mat(G_struct[key], G_plain)
            result['sh'] = G_struct.get('sh', 0)
            result['hs'] = G_struct.get('hs', 0)
            return result
        else:
            # If G_struct is not structured, treat as plain: all components are G_struct - G_plain
            val = self._sub_mat(G_struct, G_plain)
            return {'ss': val, 'hh': val, 'p': val, 'sh': 0, 'hs': 0}

    def _build_outer_mixed_eps(self,
            G_struct: Any,
            G_plain: Any,
            eps_outer: Any,
            eps_inner: Any) -> Dict[str, Any]:
        # MATLAB: G2e.ss = eps2*G22.ss - eps1*G12;  etc.
        #         G2e.sh = eps2*G22.sh;  G2e.hs = eps2*G22.hs;
        if isinstance(G_struct, dict):
            result = {}
            for key in ('ss', 'hh', 'p'):
                result[key] = self._sub_mat(
                    self._mul_eps(eps_outer, G_struct[key]),
                    self._mul_eps(eps_inner, G_plain))
            result['sh'] = self._mul_eps(eps_outer, G_struct.get('sh', 0))
            result['hs'] = self._mul_eps(eps_outer, G_struct.get('hs', 0))
            return result
        else:
            val = self._sub_mat(
                self._mul_eps(eps_outer, G_struct),
                self._mul_eps(eps_inner, G_plain))
            return {'ss': val, 'hh': val, 'p': val, 'sh': 0, 'hs': 0}

    def _excitation(self,
            exc: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        enei = exc.enei if hasattr(exc, 'enei') else exc['enei']
        nfaces = self.p.nfaces if hasattr(self.p, 'nfaces') else self.p.n

        def get_field(name: str) -> Any:
            if hasattr(exc, name):
                val = getattr(exc, name)
                if isinstance(val, np.ndarray):
                    return val
                return val
            elif isinstance(exc, dict) and name in exc:
                val = exc[name]
                if isinstance(val, np.ndarray):
                    return val
                return val
            return 0

        phi1 = get_field('phi1')
        phi1p = get_field('phi1p')
        a1 = get_field('a1')
        a1p = get_field('a1p')
        phi2 = get_field('phi2')
        phi2p = get_field('phi2p')
        a2 = get_field('a2')
        a2p = get_field('a2p')

        k = 2 * np.pi / enei

        eps1 = self.p.eps1(enei)
        eps2 = self.p.eps2(enei)
        nvec = self.nvec

        # Potential jumps: Eqs. (10,11)
        phi = self._subtract(phi2, phi1)
        a = self._subtract(a2, a1)

        # Eq. (15): alpha = a2p - a1p - ik*(outer(nvec, phi2, eps2) - outer(nvec, phi1, eps1))
        outer_term2 = self._outer_eps(nvec, phi2, eps2)
        outer_term1 = self._outer_eps(nvec, phi1, eps1)
        alpha = self._subtract(a2p, a1p) - 1j * k * self._subtract(outer_term2, outer_term1)

        # Eq. (18): De = matmul(eps2, phi2p) - matmul(eps1, phi1p)
        #               - ik*(inner(nvec, a2, eps2) - inner(nvec, a1, eps1))
        matmul_term2 = self._matmul_eps(eps2, phi2p)
        matmul_term1 = self._matmul_eps(eps1, phi1p)
        inner_term2 = self._inner_eps(nvec, a2, eps2)
        inner_term1 = self._inner_eps(nvec, a1, eps1)

        De = self._subtract(matmul_term2, matmul_term1) - 1j * k * self._subtract(inner_term2, inner_term1)

        return phi, a, alpha, De

    def _subtract(self,
            a: Any,
            b: Any) -> Any:

        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return a - b
        elif isinstance(a, np.ndarray):
            return a if b == 0 else a - b
        elif isinstance(b, np.ndarray):
            return -b if a == 0 else a - b
        else:
            return a - b

    def _outer_eps(self,
            nvec: np.ndarray,
            phi: Any,
            eps: np.ndarray) -> Any:

        if isinstance(phi, np.ndarray):
            if phi.ndim == 1:
                return nvec * (phi * eps)[:, np.newaxis]
            else:
                npol = phi.shape[1]
                n = len(nvec)
                result = np.zeros((n, 3, npol), dtype = complex)
                for ipol in range(npol):
                    result[:, :, ipol] = nvec * (phi[:, ipol] * eps)[:, np.newaxis]
                return result
        elif phi == 0:
            return 0
        else:
            return nvec * (phi * eps)

    def _inner_eps(self,
            nvec: np.ndarray,
            a: Any,
            eps: np.ndarray) -> Any:

        if isinstance(a, np.ndarray) and a.ndim >= 2:
            if a.ndim == 2:
                dot = np.sum(nvec * a, axis = 1)
                return dot * eps
            else:
                npol = a.shape[2]
                n = len(nvec)
                result = np.zeros((n, npol), dtype = complex)
                for ipol in range(npol):
                    dot = np.sum(nvec * a[:, :, ipol], axis = 1)
                    result[:, ipol] = dot * eps
                return result
        elif not isinstance(a, np.ndarray) and a == 0:
            return 0
        else:
            return 0

    def _matmul_eps(self,
            eps: np.ndarray,
            phi_p: Any) -> Any:

        if isinstance(phi_p, np.ndarray):
            if phi_p.ndim == 1:
                return eps * phi_p
            else:
                return eps[:, np.newaxis] * phi_p
        elif phi_p == 0:
            return 0
        else:
            return eps * phi_p

    def solve(self,
            exc: Any) -> Tuple[CompStruct, 'BEMRetLayer']:

        enei = exc.enei if hasattr(exc, 'enei') else exc['enei']
        self.init(enei)

        phi, a, alpha, De = self._excitation(exc)

        k = self.k
        nvec = self.nvec
        npar = self.npar
        nperp = self.nperp
        L1 = self.L1
        L2p = self.L2p
        G1i = self.G1i
        G2pi = self.G2pi
        G2 = self.G2
        G2e = self.G2e
        Sigma1 = self.Sigma1
        Sigma1e = self.Sigma1e
        Gamma = self.Gamma
        m_lu = self.m_lu

        nfaces = self.p.nfaces if hasattr(self.p, 'nfaces') else self.p.n

        # Ensure proper shapes
        if not isinstance(phi, np.ndarray) or phi.size == 0:
            phi = np.zeros(nfaces, dtype = complex)
        if not isinstance(a, np.ndarray) or a.size == 0:
            a = np.zeros((nfaces, 3), dtype = complex)
        if not isinstance(alpha, np.ndarray):
            alpha = np.zeros((nfaces, 3), dtype = complex)
        if not isinstance(De, np.ndarray):
            De = np.zeros(nfaces, dtype = complex)

        # Determine number of polarizations
        npol = 1
        if isinstance(a, np.ndarray) and a.ndim == 3:
            npol = a.shape[2]
        elif isinstance(alpha, np.ndarray) and alpha.ndim == 3:
            npol = alpha.shape[2]
        elif isinstance(phi, np.ndarray) and phi.ndim == 2:
            npol = phi.shape[1]
        elif isinstance(De, np.ndarray) and De.ndim == 2:
            npol = De.shape[1]

        if npol == 1:
            if isinstance(a, np.ndarray) and a.ndim == 3:
                a = a[:, :, 0]
            if isinstance(alpha, np.ndarray) and alpha.ndim == 3:
                alpha = alpha[:, :, 0]
            if isinstance(phi, np.ndarray) and phi.ndim == 2:
                phi = phi[:, 0]
            if isinstance(De, np.ndarray) and De.ndim == 2:
                De = De[:, 0]

        n = nfaces

        # Unit vector in z-direction
        zunit = np.zeros((n, 3))
        zunit[:, 2] = 1.0

        m_full = self.m_full

        if npol == 1:
            sig1, sig2, h1, h2 = self._solve_single(
                phi, a, alpha, De, k, n, nvec, npar, nperp, zunit,
                L1, L2p, G1i, G2pi, G2, G2e, Sigma1, Sigma1e, Gamma,
                m_lu, m_full)
        else:
            sig1 = np.zeros((n, npol), dtype = complex)
            sig2 = np.zeros((n, npol), dtype = complex)
            h1 = np.zeros((n, 3, npol), dtype = complex)
            h2 = np.zeros((n, 3, npol), dtype = complex)

            for ipol in range(npol):
                phi_i = phi[:, ipol] if phi.ndim > 1 else phi
                a_i = a[:, :, ipol] if a.ndim > 2 else a
                alpha_i = alpha[:, :, ipol] if alpha.ndim > 2 else alpha
                De_i = De[:, ipol] if De.ndim > 1 else De

                s1, s2, hh1, hh2 = self._solve_single(
                    phi_i, a_i, alpha_i, De_i, k, n, nvec, npar, nperp, zunit,
                    L1, L2p, G1i, G2pi, G2, G2e, Sigma1, Sigma1e, Gamma,
                    m_lu, m_full)

                sig1[:, ipol] = s1
                sig2[:, ipol] = s2
                h1[:, :, ipol] = hh1
                h2[:, :, ipol] = hh2

        # v1.7 Phase 1.4: host-materialize before returning to user.
        if is_cupy_array(sig1):
            sig1 = to_host(sig1)
        if is_cupy_array(sig2):
            sig2 = to_host(sig2)
        if is_cupy_array(h1):
            h1 = to_host(h1)
        if is_cupy_array(h2):
            h2 = to_host(h2)

        # v1.7.2: free any cupy LU-solve scratch buffers allocated during
        # the per-polarization _solve_single calls (each polarization
        # allocates a 2n RHS + GEMM scratch).  Without this the solve-side
        # pool can balloon by N^2 per wavelength on multi-polarization
        # sweeps even when init() is cached.
        if _CUPY_OK_V172:
            try:
                _cp_v172.cuda.runtime.deviceSynchronize()
                _cp_v172.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

        sig = CompStruct(self.p, enei, sig1 = sig1, sig2 = sig2,
            h1 = h1, h2 = h2)

        return sig, self

    def _solve_single(self,
            phi: np.ndarray,
            a: np.ndarray,
            alpha: np.ndarray,
            De: np.ndarray,
            k: float,
            n: int,
            nvec: np.ndarray,
            npar: np.ndarray,
            nperp: np.ndarray,
            zunit: np.ndarray,
            L1: np.ndarray,
            L2p: np.ndarray,
            G1i: np.ndarray,
            G2pi: np.ndarray,
            G2: Dict[str, Any],
            G2e: Dict[str, Any],
            Sigma1: np.ndarray,
            Sigma1e: np.ndarray,
            Gamma: np.ndarray,
            m_lu: Any,
            m_full: Any = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # MATLAB mldivide.m: Decompose vector potential into parallel and perpendicular
        aperp = _inner(zunit, a)  # (n,)
        apar = a - _outer(zunit, aperp)  # (n, 3)

        # MATLAB: alpha = alpha - matmul(Sigma1, a) + ik * outer(nvec, matmul(L1, phi))
        alpha = alpha - _matmul(Sigma1, a) + 1j * k * _outer(nvec, _matmul(L1, phi))

        # MATLAB: De = De - matmul(Sigma1e, phi) + ik*inner(nvec, matmul(L1, a))
        #             + ik*inner(npar, matmul((L1-L2p)*Gamma, alpha))
        De = (De
            - _matmul(Sigma1e, phi)
            + 1j * k * _inner(nvec, _matmul(L1, a))
            + 1j * k * _inner(npar, _matmul((L1 - L2p) @ Gamma, alpha)))

        # Decompose alpha into parallel and perpendicular
        alphaperp = _inner(zunit, alpha)  # (n,)
        alphapar = alpha - _outer(zunit, alphaperp)  # (n, 3)

        # Solve 2x2 block matrix equation: [sig2; h2perp] = m \ [De; alphaperp]
        rhs = np.empty(2 * n, dtype = complex)
        rhs[:n] = De
        rhs[n:] = alphaperp

        if self.use_matlab_engine and m_full is not None:
            from .matlab_bem import matlab_solve
            xi2 = matlab_solve(m_full, rhs)
        else:
            # B-3 (Agent E): also accept the ('mgpu', ...) tag produced by
            # the distributed-build path -- ``lu_solve_dispatch`` knows how
            # to route the cuSolverMg distributed solve.
            if (isinstance(m_lu, tuple) and len(m_lu) == 3
                    and m_lu[0] in ("cpu", "gpu", "mgpu")):
                xi2 = lu_solve_dispatch(m_lu, rhs)
            else:
                xi2 = lu_solve(m_lu, rhs, check_finite=False, overwrite_b=True)
        sig2 = xi2[:n]
        h2perp = xi2[n:]

        # Parallel component of surface current (MATLAB mldivide.m line 60-62)
        # h2par = matmul(G2pi*Gamma, alphapar + ik*outer(npar,
        #           matmul(L1*G2.ss - G2e.ss, sig2) + matmul(L1*G2.sh - G2e.sh, h2perp)))
        diff_ss = matmul_dispatch(L1, G2['ss']) - G2e['ss']
        diff_sh = matmul_dispatch(L1, G2['sh']) - G2e['sh']
        inner_par = _matmul(diff_ss, sig2) + _matmul(diff_sh, h2perp)
        h2par = _matmul(matmul_dispatch(G2pi, Gamma), alphapar + 1j * k * _outer(npar, inner_par))

        # Surface current h2 = h2par + outer(zunit, h2perp)
        h2 = h2par + _outer(zunit, h2perp)

        # Surface charges at inner interface (MATLAB mldivide.m line 67)
        # sig1 = matmul(G1i, matmul(G2.ss, sig2) + matmul(G2.sh, h2perp) + phi)
        sig1 = _matmul(G1i, _matmul(G2['ss'], sig2) + _matmul(G2['sh'], h2perp) + phi)

        # Surface currents at inner interface (MATLAB mldivide.m lines 69-71)
        # h1perp = matmul(G1i, matmul(G2.hs, sig2) + matmul(G2.hh, h2perp) + aperp)
        h1perp = _matmul(G1i, _matmul(G2['hs'], sig2) + _matmul(G2['hh'], h2perp) + aperp)
        # h1par = matmul(G1i, matmul(G2.p, h2par) + apar)
        h1par = _matmul(G1i, _matmul(G2['p'], h2par) + apar)
        # h1 = h1par + outer(zunit, h1perp)
        h1 = h1par + _outer(zunit, h1perp)

        return sig1, sig2, h1, h2

    def __truediv__(self,
            exc: Any) -> Tuple[CompStruct, 'BEMRetLayer']:

        return self.solve(exc)

    def __mul__(self,
            sig: Any) -> CompStruct:

        pot1 = self.potential(sig, 1)
        pot2 = self.potential(sig, 2)

        enei = sig.enei if hasattr(sig, 'enei') else sig['enei']

        return CompStruct(self.p, enei,
            phi1 = pot1.phi1, phi1p = pot1.phi1p,
            a1 = pot1.a1, a1p = pot1.a1p,
            phi2 = pot2.phi2, phi2p = pot2.phi2p,
            a2 = pot2.a2, a2p = pot2.a2p)

    def potential(self,
            sig: Any,
            inout: int = 2) -> CompStruct:

        return self.g.potential(sig, inout)

    def field(self,
            sig: Any,
            inout: int = 2) -> CompStruct:

        return self.g.field(sig, inout)

    def setup_tabulation(self, nr = 30, nz = 20):

        if self.g is None:
            self.g = CompGreenRetLayer(self.p, self.p, self.layer, **self.options)
        self.g.setup_tabulation(nr = nr, nz = nz)

    # ------------------------------------------------------------------
    # B-3 distributed build path (Agent E)
    # ------------------------------------------------------------------

    def _init_distributed_precond(self, enei: float) -> 'BEMRetLayer':
        """Distributed BEM build for the substrate (layer) solver.

        Mirrors :meth:`init` but routes every dense N^2 matrix through
        ``DistributedMatrix`` so the host never holds more than one of
        them at a time.  Each per-GPU tile carries the column slice of
        the global matrix that ``cusolverMg`` will end up owning, so the
        LU factor consumes the distributed buffers in place.

        Layout choices
        --------------
        - The ``2n x 2n`` block-response matrix ``m_full`` is built as a
          single :class:`DistributedMatrix` (column block-cyclic across N
          GPUs).  Its LU factor uses the existing ``('mgpu', ...)`` tag.
        - The auxiliary scalar-Green N x N matrices (``G1``, ``G2.p``,
          ``Gamma = Sigma1 - Sigma2p``) are also block-cyclically
          distributed for their LU factors.
        - The structured-Green block dict (``G2``, ``G2e``, ``H2``,
          ``H2e``) is built once on the host (the layer Sommerfeld table
          assembly is intrinsically per-wavelength CPU-bound; columnar
          distribution would not save host memory because the tile
          callback would still need the table per call).  Each
          structured component is materialized one at a time so peak
          host residency stays at a single ``N^2`` complex buffer.

        Numerical contract
        ------------------
        The combined matrices ``m11..m22`` and ``Gamma``/``Sigma`` are
        the same products as the legacy path -- only the storage class
        changes.  cuBLAS / cuSolverMg differ from MKL by floating-point
        rounding bounded by ``N * eps_machine`` (~1e-12 for dimer-scale
        meshes), well below the BEM solver's downstream tolerance.
        """

        from ..utils.distributed_matrix import DistributedMatrix

        cp = _cp_v172
        dist_kw = _vram_share_distributed_kwargs()
        n_gpus = int(dist_kw['n_gpus'])
        device_ids = dist_kw['device_ids']
        block_size = int(dist_kw['block_size'])

        # ---- Per-wavelength fragmentation cleanup (same as legacy) ----
        # Close cuSolverMg handles from the previous wavelength FIRST so
        # the next factor() calls do not collide with stale state.
        for _attr in ('_G1_lu', '_G2p_lu', '_Gamma_lu', 'm_lu'):
            _entry = getattr(self, _attr, None)
            if isinstance(_entry, tuple) and len(_entry) == 3 and _entry[0] == 'mgpu':
                try:
                    _entry[1].close()
                except Exception:
                    pass
        # Release the matching distributed buffers (kept the LU pointers).
        for _attr in ('_G1_dm', '_G2p_dm', '_Gamma_dm', '_m_full_dm'):
            _dm = getattr(self, _attr, None)
            if _dm is not None:
                try:
                    _dm.free()
                except Exception:
                    pass
            setattr(self, _attr, None)
        for _attr in ('_G1_lu', '_G2p_lu', '_Gamma_lu', 'm_lu',
                      'G1i', 'G2pi', 'G2', 'G2e',
                      'L1', 'L2p', 'Sigma1', 'Sigma1e', 'Gamma',
                      'm_full'):
            if hasattr(self, _attr):
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

        # Outer surface normals
        nvec = self.p.nvec
        self.nvec = nvec
        nperp = nvec[:, 2]
        npar = nvec.copy()
        npar[:, 2] = 0.0
        self.npar = npar
        self.nperp = nperp

        # Wavenumber in vacuum
        k = 2 * np.pi / enei
        self.k = k

        # Dielectric function values
        eps1_vals = self.p.eps1(enei)
        eps2_vals = self.p.eps2(enei)
        if np.allclose(eps1_vals, eps1_vals[0]) and np.allclose(eps2_vals, eps2_vals[0]):
            eps1 = eps1_vals[0]
            eps2 = eps2_vals[0]
        else:
            eps1 = np.diag(eps1_vals)
            eps2 = np.diag(eps2_vals)
        self.eps1 = eps1
        self.eps2 = eps2

        # Green-function object: build once
        if self.g is None:
            opts = dict(self.options)
            if self.greentab is not None:
                gt = self.greentab
                if hasattr(gt, 'tab'):
                    opts['greentab_obj'] = gt.tab
                elif hasattr(gt, 'r'):
                    opts['greentab_obj'] = gt
            self.g = CompGreenRetLayer(self.p, self.p, self.layer, **opts)

        # ---- Inner-surface Green (plain scalar matrices) ----
        G11 = self.g.eval(0, 0, 'G', enei)
        G21 = self.g.eval(1, 0, 'G', enei)
        H11 = self.g.eval(0, 0, 'H1', enei)
        H21 = self.g.eval(1, 0, 'H1', enei)

        G1 = self._sub_mat(G11, G21)
        G1e = self._sub_mat(self._mul_eps(eps1, G11), self._mul_eps(eps2, G21))
        H1 = self._sub_mat(H11, H21)
        H1e = self._sub_mat(self._mul_eps(eps1, H11), self._mul_eps(eps2, H21))
        del G11, G21, H11, H21
        try:
            cp.cuda.runtime.deviceSynchronize()
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

        # ---- Outer-surface Green (structured dict) ----
        G22 = self.g.eval(1, 1, 'G', enei)
        G12 = self.g.eval(0, 1, 'G', enei)
        H22 = self.g.eval(1, 1, 'H2', enei)
        H12 = self.g.eval(0, 1, 'H2', enei)

        G2 = self._build_outer_mixed(G22, G12)
        H2 = self._build_outer_mixed(H22, H12)
        G2e = self._build_outer_mixed_eps(G22, G12, eps2, eps1)
        H2e = self._build_outer_mixed_eps(H22, H12, eps2, eps1)
        del G22, G12, H22, H12
        try:
            cp.cuda.runtime.deviceSynchronize()
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

        n = G1.shape[0]

        # ============================================================
        # Step 1: distributed LU of G1 and inverse construction
        # ============================================================
        # Scatter G1 to N GPUs, factor in place, then solve against I to
        # recover G1i.  G1i is gathered to host because L1/Sigma1
        # downstream are formed via host GEMM (with structured G2 dict).
        G1_dm = DistributedMatrix.from_host(
            np.ascontiguousarray(G1),
            n_gpus=n_gpus,
            device_ids=device_ids,
            block_size=block_size,
        )
        del G1
        G1_lu_handle = G1_dm.lu_factor(backend='cusolvermg')
        # The DistributedMatrix tiles now hold the L/U factors; keep
        # both refs alive so the lu_handle's pointer array stays valid.
        self._G1_lu = ('mgpu', G1_lu_handle, None)
        self._G1_dm = G1_dm  # keep distributed buffers alive
        # Recover full inverse on host (one cuSolverMg gather).
        G1i = G1_lu_handle.solve(np.eye(n, dtype=complex))

        # ============================================================
        # Step 2: distributed LU of G2.p (substrate parallel block)
        # ============================================================
        G2p = G2['p']
        G2p_dm = DistributedMatrix.from_host(
            np.ascontiguousarray(G2p),
            n_gpus=n_gpus,
            device_ids=device_ids,
            block_size=block_size,
        )
        del G2p
        G2p_lu_handle = G2p_dm.lu_factor(backend='cusolvermg')
        self._G2p_lu = ('mgpu', G2p_lu_handle, None)
        self._G2p_dm = G2p_dm
        G2pi = G2p_lu_handle.solve(np.eye(n, dtype=complex))

        # ============================================================
        # Step 3: Sigma / L matrices on host (structured G2 dict)
        # ============================================================
        Sigma1 = H1 @ G1i
        Sigma1e = H1e @ G1i
        Sigma2p = H2['p'] @ G2pi
        L1 = G1e @ G1i
        L2p = G2e['p'] @ G2pi
        del H1, H1e, G1e

        # ============================================================
        # Step 4: distributed LU of Gamma = Sigma1 - Sigma2p
        # ============================================================
        Gamma_host = np.ascontiguousarray(Sigma1 - Sigma2p)
        del Sigma2p
        Gamma_dm = DistributedMatrix.from_host(
            Gamma_host,
            n_gpus=n_gpus,
            device_ids=device_ids,
            block_size=block_size,
        )
        Gamma_lu_handle = Gamma_dm.lu_factor(backend='cusolvermg')
        self._Gamma_lu = ('mgpu', Gamma_lu_handle, None)
        self._Gamma_dm = Gamma_dm
        Gamma = Gamma_lu_handle.solve(np.eye(n, dtype=complex))
        del Gamma_host

        # Gammapar = ik*(L1-L2p)*Gamma .* (npar*npar')
        npar_outer = npar @ npar.T
        Gammapar = 1j * k * ((L1 - L2p) @ Gamma) * npar_outer
        del npar_outer

        # ============================================================
        # Step 5: assemble the 2n x 2n block matrix on host then scatter
        # ============================================================
        # We allocate m_full on the host (single 4*N^2 complex buffer)
        # before the structured-G2 ``diff_*`` intermediates are released,
        # which is the same peak as the legacy path.  The DistributedMatrix
        # then scatters columns block-cyclic and the host copy is freed
        # immediately so subsequent wavelengths do not double-occupy.
        diff_ss = L1 @ G2['ss'] - G2e['ss']
        diff_sh = L1 @ G2['sh'] - G2e['sh']
        diff_hh = L1 @ G2['hh'] - G2e['hh']

        m11 = (Sigma1e @ G2['ss'] - H2e['ss']
            - 1j * k * (Gammapar @ diff_ss + diff_sh * nperp[:, np.newaxis]))
        m12 = (Sigma1e @ G2['sh'] - H2e['sh']
            - 1j * k * (Gammapar @ diff_sh + diff_hh * nperp[:, np.newaxis]))
        m21 = (Sigma1 @ G2['hs'] - H2['hs']
            - 1j * k * diff_ss * nperp[:, np.newaxis])
        m22 = (Sigma1 @ G2['hh'] - H2['hh']
            - 1j * k * diff_sh * nperp[:, np.newaxis])
        del diff_ss, diff_sh, diff_hh, Gammapar, H2, H2e

        m_full = np.empty((2 * n, 2 * n), dtype=complex)
        m_full[:n, :n] = m11
        m_full[:n, n:] = m12
        m_full[n:, :n] = m21
        m_full[n:, n:] = m22
        del m11, m12, m21, m22

        m_full_dm = DistributedMatrix.from_host(
            m_full,
            n_gpus=n_gpus,
            device_ids=device_ids,
            block_size=block_size,
        )
        del m_full  # host copy freed; tiles now hold the data
        try:
            cp.cuda.runtime.deviceSynchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

        m_lu_handle = m_full_dm.lu_factor(backend='cusolvermg')
        self.m_full = None
        self.m_lu = ('mgpu', m_lu_handle, None)
        self._m_full_dm = m_full_dm

        # Store auxiliary matrices on host (same as legacy path) so
        # ``_solve_single`` can reuse the existing numpy code path.
        self.G1i = G1i
        self.G2pi = G2pi
        self.G2 = G2
        self.G2e = G2e
        self.L1 = L1
        self.L2p = L2p
        self.Sigma1 = Sigma1
        self.Sigma1e = Sigma1e
        self.Gamma = Gamma

        # Sync ALL devices in the distributed grid before returning.
        # cuSolverMg leaves async work queued on each device and the next
        # ``solve()`` (running on a different handle) can otherwise see
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

    def clear(self) -> 'BEMRetLayer':

        self.L1 = None
        self.L2p = None
        self.G1i = None
        self.G2pi = None
        self.G2 = None
        self.G2e = None
        self.Sigma1 = None
        self.Sigma1e = None
        self.Gamma = None
        self.m_lu = None
        self.m_full = None
        self._G1_lu = None
        self._G2p_lu = None
        self._Gamma_lu = None
        # B-3 (Agent E): release distributed buffers when present.
        for _attr in ('_G1_dm', '_G2p_dm', '_Gamma_dm', '_m_full_dm'):
            _dm = getattr(self, _attr, None)
            if _dm is not None:
                try:
                    _dm.free()
                except Exception:
                    pass
                setattr(self, _attr, None)
        self.enei = None
        return self

    def __call__(self,
            enei: float) -> 'BEMRetLayer':

        return self.init(enei)

    def __repr__(self) -> str:
        status = 'enei={:.1f}nm'.format(self.enei) if self.enei is not None else 'not initialized'
        n = self.p.nfaces if hasattr(self.p, 'nfaces') else self.p.n if hasattr(self.p, 'n') else '?'
        return 'BEMRetLayer(p: {} faces, {})'.format(n, status)
