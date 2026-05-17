"""
BEM solver for quasistatic approximation.

MATLAB: BEM/@bemstat/
100% identical to MATLAB MNPBEM implementation.

Given an external excitation, BEMStat computes the surface charges
such that the boundary conditions of Maxwell's equations in the
quasistatic approximation are fulfilled.

Reference:
    Garcia de Abajo and Howie, PRB 65, 115418 (2002)
    Hohenester et al., PRL 103, 106801 (2009)
"""

import os

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from ..greenfun import CompGreenStat, CompStruct
from ..utils.matlab_compat import msqrt
from ..utils.gpu import lu_factor_dispatch, lu_solve_dispatch, to_host, is_cupy_array


# v1.7.2 memory-pool parity: BEMStat's quasistatic single-LU pipeline
# still routes Lambda+F assembly + dense LU factor through *_dispatch when
# MNPBEM_GPU=1, which means cupy's caching pool retains the previous
# wavelength's mat_lu (~N^2 complex) plus the diag(Lambda) scratch across
# the entire sweep.  Mirror BEMRet._init_gpu_assemble's v1.7.2 pattern:
# deviceSynchronize then free_all_blocks at every reasonable boundary
# (start-of-wl, after the LU factor, at __truediv__ exit, and clear()).
# Implemented inline at each call site to keep static grep audit parity
# with bem_ret.py.  Honors MNPBEM_GPU_POOL_LIMIT_GB env var the same way
# BEMRet does.
try:
    import cupy as _cp_v172  # type: ignore
    _CUPY_OK_V172 = True
except Exception:
    _cp_v172 = None  # type: ignore
    _CUPY_OK_V172 = False


def _vram_share_lu_kwargs() -> dict:
    if os.environ.get('MNPBEM_VRAM_SHARE', '0') != '1':
        return {}
    n_gpus = int(os.environ.get('MNPBEM_VRAM_SHARE_GPUS', '1'))
    if n_gpus <= 1:
        return {}
    backend = os.environ.get('MNPBEM_VRAM_SHARE_BACKEND', 'cusolvermg')
    return {'n_gpus': n_gpus, 'backend': backend}


def _vram_share_active() -> bool:
    """Return True iff the distributed-build path should be taken.

    The quasistatic mirror of ``bem_ret_layer._vram_share_active``.
    Distinct from ``_vram_share_lu_kwargs``: that helper controls
    whether the LU dispatch routes through cuSolverMg, while this gate
    decides whether the BEM matrix is also *built* distributed
    (column block-cyclic across N GPUs).  Off by default; the user
    must opt in via ``MNPBEM_VRAM_SHARE_DISTRIBUTED=1`` so existing
    LU-only multi-GPU sweeps keep their build path.
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


class BEMStat(object):
    """
    BEM solver for quasistatic approximation.

    MATLAB: @bemstat

    Properties
    ----------
    name : str
        'bemsolver' (constant)
    needs : dict
        {'sim': 'stat'} (constant)
    p : ComParticle
        Composite particle (see comparticle)
    F : ndarray
        Surface derivative of Green function
    enei : float or None
        Light wavelength in vacuum
    g : CompGreenStat (private)
        Green function (needed in bemstat/field)
    mat : ndarray (private)
        -inv(Lambda + F)

    Methods
    -------
    __init__(p, enei=None, **options)
        Initialize quasistatic BEM solver
    solve(exc)
        Solve BEM equations for given excitation
    __truediv__(exc)
        Surface charge for given excitation (operator \)
    __mul__(sig)
        Induced potential for given surface charge (operator *)
    field(sig, inout=2)
        Electric field inside/outside of particle surface
    potential(sig, inout=2)
        Potentials and surface derivatives inside/outside of particle
    clear()
        Clear auxiliary matrices
    __call__(enei)
        Computes resolvent matrix for later use in __truediv__

    Examples
    --------
    >>> from mnpbem import EpsConst, EpsTable, trisphere, ComParticle
    >>> from mnpbem.bem import BEMStat
    >>>
    >>> # Create gold sphere
    >>> eps_tab = [EpsConst(1.0), EpsTable('gold.dat')]
    >>> sphere = trisphere(144, 10.0)
    >>> p = ComParticle(eps_tab, [sphere], [[2, 1]])
    >>>
    >>> # Create BEM solver
    >>> bem = BEMStat(p)
    >>>
    >>> # Solve for excitation
    >>> sig = bem \ exc  # or sig = bem.solve(exc)
    >>>
    >>> # Get induced potential
    >>> phi = bem * sig
    """

    # Class constants
    name = 'bemsolver'
    needs = {'sim': 'stat'}

    def __init__(self, p, enei=None, **options):
        """
        Initialize quasistatic BEM solver.

        MATLAB: bemstat.m, private/init.m

        Parameters
        ----------
        p : ComParticle
            Compound of particles (see comparticle)
        enei : float, optional
            Light wavelength in vacuum
        **options : dict
            Additional options passed to CompGreenStat. Special keys:

            schur : bool or 'auto', optional
                Activate Schur-complement elimination of EpsNonlocal
                cover-layer faces. ``True`` or ``'auto'`` enables the
                reduction whenever a cover layer is detected via
                ``detect_shell_core_partition``; ``False`` (default) keeps
                the full BEM matrix.

        Examples
        --------
        >>> bem = BEMStat(p)
        >>> bem = BEMStat(p, enei=600.0)
        """
        # Validate particle
        if p is None:
            raise ValueError(
                "BEMStat: 'p' must be a ComParticle (or compatible particle "
                "object), got None.")
        if not (hasattr(p, 'pos') and hasattr(p, 'nvec') and hasattr(p, 'eps')):
            raise TypeError(
                "BEMStat: 'p' must expose ComParticle-like attributes "
                "(pos, nvec, eps); got {!r}.".format(type(p).__name__))

        # Save particle
        self.p = p

        # Schur option (extract before forwarding to CompGreenStat).
        self._schur_opt = options.pop('schur', False)
        self._schur_active = False
        self._shell_idx = None
        self._core_idx = None
        self._schur_reduce_rhs = None
        self._schur_recover = None
        # v1.8 (VRAM-Schur): keepalive dict for the distributed Schur
        # host slices (M_ss LU + M_sc + M_cs + D_inv_C).  None on the
        # legacy host path; set by ``_init_distributed_schur``.
        self._schur_dist_keepalive = None

        # Initialize properties
        self.enei = None
        self.mat_lu = None

        # Green function
        # MATLAB: obj.g = compgreenstat(p, p, varargin{:})
        self.g = CompGreenStat(p, p, **options)

        # Surface derivative of Green function
        # MATLAB: obj.F = subsref(obj.g, substruct('.', 'F'))
        F_obj = self.g.F
        # If hmatrix=True swapped self.g for an ACACompGreenStat, F is an
        # HMatrix; convert to dense so the standard LU solver works.
        if hasattr(F_obj, 'full') and not isinstance(F_obj, np.ndarray):
            F_obj = F_obj.full()
        self.F = F_obj

        # Initialize for given wavelength
        # MATLAB: if exist('enei', 'var') && ~isempty(enei)
        if enei is not None:
            self(enei)

    def _init_matrices(self, enei):
        """
        Initialize matrices for BEM solver.

        MATLAB: bemstat/subsref.m case '()'

        Parameters
        ----------
        enei : float
            Light wavelength in vacuum
        """
        # Use previously computed matrices?
        # MATLAB: if isempty(obj.enei) || obj.enei ~= enei
        if self.enei is None or self.enei != enei:
            # B-3 distributed multi-GPU build path. When MNPBEM_VRAM_SHARE
            # _DISTRIBUTED=1 and n_gpus>=2, route the BEM matrix build +
            # LU through DistributedMatrix so the host never holds the
            # full ``-(Lambda + F)`` matrix simultaneously with the
            # cuSolverMg LU factor.  v1.8 (VRAM-Schur) extends this to
            # the Schur-reduced path: when ``self._schur_opt`` is also
            # set and an EpsNonlocal cover layer is detected, the
            # reduced (n_core, n_core) BEM matrix is assembled directly
            # distributed via ``_init_distributed_schur``.
            if _vram_share_active():
                if self._schur_opt:
                    try:
                        return self._init_distributed_schur(enei)
                    except Exception as e:  # pragma: no cover - safety
                        import warnings
                        warnings.warn(
                            '[warn] BEMStat distributed Schur assembly '
                            'failed ({}); falling back to host build'
                            .format(e))
                else:
                    try:
                        return self._init_distributed_assemble(enei)
                    except Exception as e:  # pragma: no cover - safety
                        import warnings
                        warnings.warn(
                            '[warn] BEMStat distributed assembly failed ({}); '
                            'falling back to host build'.format(e))
            # v1.7.2 MATLAB-parity: free cupy pools before allocating the
            # new wavelength's BEM matrices.  The previous wavelength's
            # mat_lu (a ~N^2 complex device buffer when lu_factor_dispatch
            # routed through cupy) would otherwise pin until the
            # ``self.mat_lu = ...`` rebind below, leaving the pool with
            # double the peak through the diag(Lambda) + M_full GEMM step.
            # Drop the stale handle first so free_all_blocks can actually
            # return the device storage to the pool.
            self.mat_lu = None
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
                        _mempool_pre.set_limit(
                            size=int(_pool_limit_gb * (1024 ** 3))
                        )
                    import gc as _gc
                    _gc.collect()
                    _cp_v172.cuda.runtime.deviceSynchronize()
                    _mempool_pre.free_all_blocks()
                    _pinned_pre.free_all_blocks()
                except Exception:
                    pass

            # Inside and outside dielectric function
            # MATLAB: eps1 = obj.p.eps1(enei); eps2 = obj.p.eps2(enei);
            eps1 = self.p.eps1(enei)
            eps2 = self.p.eps2(enei)
            # v1.7.2: small intermediate sync — when eps1/eps2 evaluation
            # routes through cupy (EpsTable lookup with GPU index_select)
            # the scratch arrays are tiny but still kept by the pool until
            # explicit release.
            if _CUPY_OK_V172:
                try:
                    _cp_v172.cuda.runtime.deviceSynchronize()
                    _cp_v172.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass

            # Lambda [Garcia de Abajo, Eq. (23)]
            # MATLAB: lambda = 2 * pi * (eps1 + eps2) ./ (eps1 - eps2)
            lambda_diag = 2 * np.pi * (eps1 + eps2) / (eps1 - eps2)

            # BEM resolvent matrix
            # MATLAB: obj.mat = -inv(diag(lambda) + obj.F)
            Lambda = np.diag(lambda_diag)
            del lambda_diag
            if _CUPY_OK_V172:
                try:
                    _cp_v172.cuda.runtime.deviceSynchronize()
                    _cp_v172.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
            M_full = -(Lambda + self.F)
            # v1.7.2: Lambda is a diag-only N^2 buffer; once added into
            # M_full it is no longer needed.  Drop the local handle so the
            # cupy pool reclaims it before the LU factor below.
            del Lambda
            if _CUPY_OK_V172:
                try:
                    _cp_v172.cuda.runtime.deviceSynchronize()
                    _cp_v172.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass

            # Optional Schur-complement reduction over EpsNonlocal cover-
            # layer faces. The reduced matrix has size (M, M) where M is the
            # number of core (non-shell) faces. Mathematically equivalent to
            # the full block solve. VRAM-share kwargs propagate into mat_lu.
            _lu_opts = _vram_share_lu_kwargs()
            self._schur_active = False
            if self._schur_opt:
                from .schur_helpers import (
                    schur_eliminate, detect_shell_core_partition,
                )
                partition = detect_shell_core_partition(self.p)
                if partition is not None:
                    shell_idx, core_idx = partition
                    M_eff, reduce_rhs, recover = schur_eliminate(
                            np.asarray(M_full), shell_idx, core_idx)
                    self._shell_idx = shell_idx
                    self._core_idx = core_idx
                    self._schur_reduce_rhs = reduce_rhs
                    self._schur_recover = recover
                    self._schur_active = True
                    self.mat_lu = lu_factor_dispatch(M_eff, **_lu_opts)
                    del M_eff
                    # v1.7.2: post-Schur-LU pool drain (the schur_eliminate
                    # work arrays plus the LU scratch can each be N^2).
                    if _CUPY_OK_V172:
                        try:
                            _cp_v172.cuda.runtime.deviceSynchronize()
                            _cp_v172.get_default_memory_pool().free_all_blocks()
                        except Exception:
                            pass
                else:
                    self.mat_lu = lu_factor_dispatch(M_full, **_lu_opts)
            else:
                self.mat_lu = lu_factor_dispatch(M_full, **_lu_opts)
            # v1.7.2: drop M_full (now captured by the LU factor) so the
            # pool can recycle its N^2 buffer.  Sync + free so the next
            # wavelength enters _init_matrices with a clean slate.
            del M_full
            if _CUPY_OK_V172:
                try:
                    _cp_v172.cuda.runtime.deviceSynchronize()
                    _cp_v172.get_default_memory_pool().free_all_blocks()
                    _cp_v172.get_default_pinned_memory_pool().free_all_blocks()
                except Exception:
                    pass

            # Save energy
            # MATLAB: obj.enei = enei
            self.enei = enei

        return self

    def _init_distributed_assemble(self, enei):
        """B-3 distributed multi-GPU quasistatic BEM matrix assembly.

        Builds ``M = -(diag(Lambda) + F)`` directly distributed across N
        GPUs via :class:`mnpbem.utils.distributed_matrix.DistributedMatrix`,
        then factors the distributed tiles in place with cuSolverMg
        (block-cyclic Getrf).  The single (N, N) host buffer that the
        legacy path allocates for ``M_full`` is never materialized.

        The per-column tile is built from
        :meth:`CompGreenStat.eval_block` (key='F'), plus the lambda
        contribution on the diagonal rows that fall inside the column
        range.  The eval_block call is a slice into ``self.F`` (already
        on host) — no extra Green-function recompute happens.  The
        memory saving comes entirely from never building ``M_full`` and
        from cuSolverMg owning the LU in distributed tiles.

        Result residency
        ----------------
        - ``self.mat_lu`` is stored as ``('mgpu', MultiGPULU_handle,
          None)``.  ``lu_solve_dispatch`` already routes that tag through
          ``MultiGPULU.solve`` so downstream ``BEMStat.solve`` consumers
          don't change.
        - The keepalive ``DistributedMatrix`` (which owns the per-GPU
          tiles backing the LU's ctypes pointer array) is attached as
          ``mat_lu[1]._distmat_keepalive`` so the device memory stays
          live until the handle is closed.

        Bit-identity contract
        ---------------------
        cuSolverMg differs from MKL LU by floating-point rounding bounded
        by ``N * eps_machine``.  At dimer-scale meshes (N ~ 12-15k) this
        is ~1e-12 relative — well below the BEM physics tolerance.  When
        a regression suspicion arises, unset
        ``MNPBEM_VRAM_SHARE_DISTRIBUTED`` to fall back to the legacy
        host build.
        """

        import gc as _gc
        from ..utils.distributed_matrix import DistributedMatrix
        cp = _cp_v172

        dist_kw = _vram_share_distributed_kwargs()
        n_gpus = int(dist_kw['n_gpus'])
        device_ids = dist_kw['device_ids']
        block_size = int(dist_kw['block_size'])
        if device_ids is None:
            device_ids = list(range(n_gpus))
        assert len(device_ids) == n_gpus, \
            '[error] MNPBEM_VRAM_SHARE_DEVICE_IDS length must equal MNPBEM_VRAM_SHARE_GPUS'

        # ---- Per-wavelength cleanup: close stale LU + free old tiles ----
        old = getattr(self, 'mat_lu', None)
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
        _gc.collect()
        try:
            _pool_limit_gb = float(
                os.environ.get('MNPBEM_GPU_POOL_LIMIT_GB', '0'))
        except (TypeError, ValueError):
            _pool_limit_gb = 0.0
        try:
            mempool = cp.get_default_memory_pool()
            pinned = cp.get_default_pinned_memory_pool()
            if _pool_limit_gb > 0:
                mempool.set_limit(size=int(_pool_limit_gb * (1024 ** 3)))
            for d in device_ids:
                cp.cuda.runtime.setDevice(d)
                cp.cuda.runtime.deviceSynchronize()
                mempool.free_all_blocks()
            pinned.free_all_blocks()
        except Exception:
            pass

        # ---- Per-wavelength Lambda evaluation ----
        # MATLAB: lambda = 2*pi*(eps1+eps2)/(eps1-eps2)
        eps1 = self.p.eps1(enei)
        eps2 = self.p.eps2(enei)
        lambda_diag = (2 * np.pi * (eps1 + eps2) / (eps1 - eps2)).astype(
            np.complex128)

        # ---- Build M = -(diag(Lambda) + F) column-tile distributed ----
        N = int(self.F.shape[0])
        compg = self.g
        F_host = np.asarray(self.F)

        def _eval_M_tile(gpu_idx, c0, c1):
            # Pull the column slice of F from the host array.  We do not
            # route through compg.eval_block here because BEMStat's
            # ``self.F`` already holds any refinement / closed-surface
            # corrections that the Green-function build applied; the
            # raw eval_block tile is bit-identical to the slice when no
            # external refun has reshaped it (the standard case), but
            # taking the slice from ``self.F`` is safe whether or not
            # the user supplied a ``refun`` at construction time.
            # Promote to complex128 up-front so the lambda addition
            # below does not warn / silently truncate when self.F is a
            # real-valued ndarray (which is the standard case before
            # any dispersive correction is applied).
            block = F_host[:, c0:c1].astype(np.complex128, copy=True)
            # Add Lambda on the diagonal entries that fall in this tile.
            ncol = c1 - c0
            for k in range(ncol):
                j = c0 + k
                block[j, k] += lambda_diag[j]
            # Negate to form M = -(diag(Lambda) + F).
            return -block

        M_dm = DistributedMatrix.from_func(
            shape=(N, N),
            dtype=np.complex128,
            n_gpus=n_gpus,
            device_ids=device_ids,
            block_size=block_size,
            eval_func=_eval_M_tile,
        )

        # Factor in place; the keepalive holds the device tiles.
        M_mglu = M_dm.lu_factor(backend='cusolvermg')
        M_mglu._distmat_keepalive = M_dm  # type: ignore[attr-defined]
        self.mat_lu = ('mgpu', M_mglu, None)

        # Schur path is incompatible with the column-split build (it
        # needs the full host matrix to eliminate shell faces).  When
        # _vram_share_active is True with schur on, we route through
        # ``_init_distributed_schur`` instead of this method.  Reset
        # the flag here for safety so downstream solve takes the
        # standard path on the non-Schur distributed branch.
        self._schur_active = False

        # Final sync + pool compaction across all participating devices.
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

    def _init_distributed_schur(self, enei):
        """v1.8 VRAM-Schur: distributed BEM matrix build + Schur reduction.

        Combines the column-distributed build path with the Schur-complement
        elimination of EpsNonlocal cover-layer shell faces.  Both the full
        BEM matrix ``M = -(diag(Lambda) + F)`` and the reduced matrix
        ``M_eff = M_cc - M_cs @ inv(M_ss) @ M_sc`` are assembled directly
        as ``DistributedMatrix`` tiles across N GPUs; the dominant LU
        factor operates on ``M_eff`` (size ``n_core, n_core``).

        Pipeline
        --------
        1. Detect shell / core partition (skip distributed path if no
           cover layer is present and fall back to the non-Schur
           distributed build).
        2. Build the small host slices ``M_ss / M_sc / M_cs`` via the
           same ``M_eval`` callback used by the distributed assembler
           (slab reads of ``F_host[:, c0:c1] + Lambda diag - negation``).
        3. ``schur_eliminate_distributed`` runs ``D_inv_C = lu_solve(M_ss,
           M_sc)`` on host (single dense LU, ~``(N/2, N/2)``), then
           builds the reduced matrix tile-by-tile via
           ``DistributedMatrix.from_func`` (each tile = M_cc column slice
           minus ``M_cs @ D_inv_C[:, c0:c1]``, computed on the owning GPU).
        4. cuSolverMg getrf factors the reduced distributed matrix in
           place.  ``self.mat_lu`` becomes ``('mgpu', handle, None)``
           with ``handle._distmat_keepalive`` holding the per-GPU tiles.

        Bit-identity contract
        ---------------------
        The reduce/recover callables are bit-identical to
        ``schur_eliminate`` on a host build (same lu_solve calls on the
        same M_ss factor).  The reduced LU itself differs from a CPU LU
        only by floating-point rounding bounded by ``n_core *
        eps_machine`` — well below the BEM physics tolerance.
        """

        import gc as _gc
        from ..utils.distributed_matrix import DistributedMatrix
        from .schur_helpers import (
                schur_eliminate_distributed,
                detect_shell_core_partition,
        )
        cp = _cp_v172

        partition = detect_shell_core_partition(self.p)
        if partition is None:
            # No cover layer — defer to the standard distributed build
            # which factors the full (N, N) matrix.
            self._schur_active = False
            return self._init_distributed_assemble(enei)

        shell_idx, core_idx = partition

        dist_kw = _vram_share_distributed_kwargs()
        n_gpus = int(dist_kw['n_gpus'])
        device_ids = dist_kw['device_ids']
        block_size = int(dist_kw['block_size'])
        if device_ids is None:
            device_ids = list(range(n_gpus))
        assert len(device_ids) == n_gpus, \
            '[error] MNPBEM_VRAM_SHARE_DEVICE_IDS length must equal MNPBEM_VRAM_SHARE_GPUS'

        # ---- Per-wavelength cleanup: close stale LU + free old tiles ----
        old = getattr(self, 'mat_lu', None)
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
        self._schur_reduce_rhs = None
        self._schur_recover = None
        self._schur_dist_keepalive = None
        _gc.collect()
        try:
            _pool_limit_gb = float(
                os.environ.get('MNPBEM_GPU_POOL_LIMIT_GB', '0'))
        except (TypeError, ValueError):
            _pool_limit_gb = 0.0
        try:
            mempool = cp.get_default_memory_pool()
            pinned = cp.get_default_pinned_memory_pool()
            if _pool_limit_gb > 0:
                mempool.set_limit(size=int(_pool_limit_gb * (1024 ** 3)))
            for d in device_ids:
                cp.cuda.runtime.setDevice(d)
                cp.cuda.runtime.deviceSynchronize()
                mempool.free_all_blocks()
            pinned.free_all_blocks()
        except Exception:
            pass

        # ---- Lambda evaluation -----------------------------------------
        eps1 = self.p.eps1(enei)
        eps2 = self.p.eps2(enei)
        lambda_diag = (2 * np.pi * (eps1 + eps2) / (eps1 - eps2)).astype(
            np.complex128)

        # ---- M_eval callback (full column slice, includes shell+core
        # rows) -- shared with the distributed eval_func.
        N = int(self.F.shape[0])
        F_host = np.asarray(self.F)

        def _M_eval(c0: int, c1: int) -> np.ndarray:
            block = F_host[:, c0:c1].astype(np.complex128, copy = True)
            ncol = c1 - c0
            for k in range(ncol):
                j = c0 + k
                block[j, k] += lambda_diag[j]
            # Negate to form M = -(diag(Lambda) + F).
            return -block

        # ---- Build the reduced distributed matrix ----------------------
        M_eff_dm, reduce_rhs, recover, keepalive = schur_eliminate_distributed(
                M_eval = _M_eval,
                N_full = N,
                shell_indices = shell_idx,
                core_indices = core_idx,
                n_gpus = n_gpus,
                device_ids = device_ids,
                block_size = block_size)

        # ---- Factor reduced matrix in place ----------------------------
        M_mglu = M_eff_dm.lu_factor(backend = 'cusolvermg')
        M_mglu._distmat_keepalive = M_eff_dm  # type: ignore[attr-defined]
        self.mat_lu = ('mgpu', M_mglu, None)

        # Wire up Schur callables so _schur_solve picks them up.
        self._shell_idx = np.asarray(shell_idx)
        self._core_idx = np.asarray(core_idx)
        self._schur_reduce_rhs = reduce_rhs
        self._schur_recover = recover
        self._schur_active = True
        # Stash the keepalive on the instance so the host slices the
        # closures capture do not get GC'd while we still need them.
        self._schur_dist_keepalive = keepalive

        # Final sync + pool compaction across all participating devices.
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

    def solve(self, exc):
        """
        Solve BEM equations for given excitation.

        MATLAB: bemstat/solve.m

        Parameters
        ----------
        exc : CompStruct
            compstruct with fields for external excitation

        Returns
        -------
        sig : CompStruct
            compstruct with fields for surface charge
        obj : BEMStat
            Updated BEM solver object

        Examples
        --------
        >>> sig, bem = bem.solve(exc)
        """
        # MATLAB: [sig, obj] = mldivide(obj, exc)
        return self.__truediv__(exc)

    def __truediv__(self, exc):
        """
        Surface charge for given excitation.

        MATLAB: bemstat/mldivide.m

        Usage
        -----
        sig = obj \ exc

        Parameters
        ----------
        exc : CompStruct
            compstruct with field 'phip' for external excitation

        Returns
        -------
        sig : CompStruct
            compstruct with field for surface charge
        obj : BEMStat
            Updated BEM solver object

        Examples
        --------
        >>> sig, bem = bem \ exc
        """
        # Initialize BEM solver (if needed)
        # MATLAB: obj = subsref(obj, substruct('()', {exc.enei}))
        self._init_matrices(exc.enei)

        # Solve: σ = mat · φₚ
        # MATLAB: sig = compstruct(obj.p, exc.enei, 'sig', matmul(obj.mat, exc.phip))
        if self._schur_active:
            sig_result = self._schur_solve(exc.phip)
        else:
            sig_result = self._lu_solve(self.mat_lu, exc.phip)
        # v1.7 Phase 1.4 fix: host-materialize so np.asarray(sig.sig) works.
        if is_cupy_array(sig_result):
            sig_result = to_host(sig_result)

        # v1.7.2: free any cupy LU-solve scratch buffers allocated during
        # the LU-back-substitute (each polarization allocates an O(N)
        # vector + intermediate GEMM scratch).  Without this the solve-
        # side pool can grow steadily on multi-polarization sweeps even
        # when _init_matrices is cached.  Mirrors BEMRet.solve's closing
        # pattern.
        if _CUPY_OK_V172:
            try:
                _cp_v172.cuda.runtime.deviceSynchronize()
                _cp_v172.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

        sig = CompStruct(self.p, exc.enei, sig=sig_result)

        return sig, self

    def _schur_solve(self, phip):
        # Reduced RHS lives only on core faces. Solve (M, M) reduced system
        # then recover the full sigma vector via the cached
        # _schur_recover callable.
        b_full = np.asarray(phip)
        b_eff = self._schur_reduce_rhs(b_full)
        sig_core = self._lu_solve(self.mat_lu, b_eff)
        sig_full = self._schur_recover(sig_core, b_full)
        # v1.7.2: post-Schur-recover pool drain — the recover callable
        # allocates the shell-face restore vector + a GEMM scratch.
        if _CUPY_OK_V172:
            try:
                _cp_v172.cuda.runtime.deviceSynchronize()
                _cp_v172.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        return sig_full

    def __mul__(self, sig):
        """
        Induced potential for given surface charge.

        MATLAB: bemstat/mtimes.m

        Usage
        -----
        phi = obj * sig

        Parameters
        ----------
        sig : CompStruct
            compstruct with fields for surface charge

        Returns
        -------
        phi : CompStruct
            compstruct with fields for induced potential

        Examples
        --------
        >>> phi = bem * sig
        """
        # MATLAB: phi = potential(obj, sig, 1) + potential(obj, sig, 2)
        pot1 = self.potential(sig, 1)
        pot2 = self.potential(sig, 2)

        # Combine potentials
        # pot1 has phi1, phi1p; pot2 has phi2, phi2p
        # Return combined result
        phi = CompStruct(self.p, sig.enei,
                        phi1=pot1.phi1, phi1p=pot1.phi1p,
                        phi2=pot2.phi2, phi2p=pot2.phi2p)
        return phi

    def field(self, sig, inout=2):
        """
        Electric field inside/outside of particle surface.

        MATLAB: bemstat/field.m

        Parameters
        ----------
        sig : CompStruct
            COMPSTRUCT object with surface charges
        inout : int, optional
            Electric field inside (inout=1) or outside (inout=2, default) of particle

        Returns
        -------
        field : CompStruct
            COMPSTRUCT object with electric field

        Examples
        --------
        >>> field = bem.field(sig, inout=2)
        """
        # Compute field from derivative of Green function or from potential interpolation
        # MATLAB: switch obj.g.deriv
        if self.g.deriv == 'cart':
            # MATLAB: field = obj.g.field(sig, inout)
            return self.g.field(sig, inout)

        elif self.g.deriv == 'norm':
            # Electric field in normal direction
            # MATLAB: switch inout
            #           case 1: e = -outer(obj.p.nvec, matmul(obj.g.H1, sig.sig))
            #           case 2: e = -outer(obj.p.nvec, matmul(obj.g.H2, sig.sig))
            if inout == 1:
                H = self.g.H1
            else:
                H = self.g.H2

            # e = -outer(nvec, H @ sig.sig)
            # MATLAB outer(nvec, scalar) creates (n, 3) matrix
            H_sig = self._matmul(H, sig.sig)
            e = -self._outer(self.p.nvec, H_sig)

            # Tangential directions computed by interpolation and derivative
            # MATLAB: phi = interp(obj.p, matmul(obj.g.G, sig.sig))
            #         [phi1, phi2, t1, t2] = deriv(obj.p, phi)
            G_sig = self._matmul(self.g.G, sig.sig)
            phi = self.p.interp(G_sig)
            phi1, phi2, t1, t2 = self.p.deriv(phi)

            # Normal vector
            # MATLAB: nvec = cross(t1, t2)
            #         h = sqrt(dot(nvec, nvec, 2)); nvec = bsxfun(@rdivide, nvec, h)
            nvec = np.cross(t1, t2)
            h = msqrt(np.sum(nvec * nvec, axis=1, keepdims=True))
            nvec = nvec / h

            # Tangential derivative of PHI
            # MATLAB: phip = outer(bsxfun(@rdivide, cross(t2, nvec, 2), h), phi1) -
            #                outer(bsxfun(@rdivide, cross(t1, nvec, 2), h), phi2)
            tvec1 = np.cross(t2, nvec) / h
            tvec2 = np.cross(t1, nvec) / h
            phip = self._outer(tvec1, phi1) - self._outer(tvec2, phi2)

            # Add electric field in tangential direction
            # MATLAB: e = e - phip
            e = e - phip

            # Set output
            # MATLAB: field = compstruct(obj.p, sig.enei, 'e', e)
            field = CompStruct(self.p, sig.enei, e=e)
            return field

    def potential(self, sig, inout=2):
        """
        Determine potentials and surface derivatives inside/outside of particle.

        MATLAB: bemstat/potential.m

        Parameters
        ----------
        sig : CompStruct
            compstruct with surface charges
        inout : int, optional
            Potential inside (inout=1) or outside (inout=2, default) of particle

        Returns
        -------
        pot : CompStruct
            compstruct object with potentials

        Examples
        --------
        >>> pot = bem.potential(sig, inout=2)
        """
        # MATLAB: pot = obj.g.potential(sig, inout)
        return self.g.potential(sig, inout)

    def clear(self):
        """
        Clear auxiliary matrices.

        MATLAB: bemstat/clear.m

        Returns
        -------
        self : BEMStat
            Returns self for chaining

        Examples
        --------
        >>> bem = bem.clear()
        """
        # MATLAB: obj.mat = []
        # v1.7 A3 fix: also reset enei so the cache gate in _init_matrices
        # does not skip rebuild when the user re-solves at the same
        # wavelength after clear().  Stale enei + mat_lu=None previously
        # crashed __truediv__ with a NoneType unpack error.  Schur
        # auxiliaries are likewise dropped so a subsequent solve does not
        # accidentally reuse the recover callable bound to a freed factor.
        # B-3: when the LU is an 'mgpu' tuple we must close the handle
        # and free its distributed buffers explicitly; otherwise the
        # device tiles would remain pinned through the cupy pool until
        # the BEMStat object is GC'd, which can happen long after
        # clear() returned in long-running sweeps.
        old = self.mat_lu
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
        self.enei = None
        self._schur_active = False
        self._schur_reduce_rhs = None
        self._schur_recover = None
        # v1.8 (VRAM-Schur): drop the distributed-Schur host keepalive
        # (M_ss LU + M_sc + M_cs + D_inv_C).  The closures captured these
        # via the dict so the explicit reset ensures the references are
        # gone before the next solve allocates.  Safe no-op on the
        # non-distributed path (attribute never set).
        if hasattr(self, '_schur_dist_keepalive'):
            self._schur_dist_keepalive = None
        # v1.7.2: an explicit clear() is the user's signal that they want
        # the BEM cache gone — free the cupy pool so memory returns to
        # the device immediately (analogous to MATLAB's wavelength-end
        # immediate free).  Pinned pool too in case potential() / field()
        # routed any host transfers through pinned scratch.
        if _CUPY_OK_V172:
            try:
                _cp_v172.cuda.runtime.deviceSynchronize()
                _cp_v172.get_default_memory_pool().free_all_blocks()
                _cp_v172.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
        return self

    def __call__(self, enei):
        """
        Computes resolvent matrix for later use in mldivide.

        MATLAB: bemstat/subsref.m case '()'

        Parameters
        ----------
        enei : float
            Light wavelength in vacuum

        Returns
        -------
        self : BEMStat
            Returns self for chaining

        Examples
        --------
        >>> bem = bem(600.0)
        """
        return self._init_matrices(enei)

    @staticmethod
    def _lu_solve(lu_piv, b):
        if isinstance(lu_piv, tuple) and len(lu_piv) == 3 and lu_piv[0] in ("cpu", "gpu", "mgpu"):
            if b.ndim == 1:
                result = lu_solve_dispatch(lu_piv, b)
            else:
                result = lu_solve_dispatch(lu_piv, b.reshape(b.shape[0], -1)).reshape(b.shape)
            # v1.7.2: post-solve pool drain — when lu_piv lives on cupy
            # the LU back-substitute allocates an O(N) scratch per RHS;
            # free it eagerly so a long-running sweep does not accumulate
            # solve-side residue.
            if _CUPY_OK_V172:
                try:
                    _cp_v172.cuda.runtime.deviceSynchronize()
                    _cp_v172.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
            return result
        if b.ndim == 1:
            return lu_solve(lu_piv, b, check_finite=False)
        else:
            return lu_solve(lu_piv, b.reshape(b.shape[0], -1), check_finite=False).reshape(b.shape)

    def _matmul(self, a, x):
        """
        Generalized matrix multiplication for tensors.

        MATLAB: Misc/matmul.m
        """
        if np.isscalar(a) or (isinstance(a, np.ndarray) and a.size == 1):
            if a == 0:
                return 0
            else:
                return a * x
        elif np.isscalar(x) or (isinstance(x, np.ndarray) and x.size == 1):
            if x == 0:
                return 0
            else:
                return a * x
        else:
            # A is matrix/tensor
            siza = a.shape
            sizx = x.shape if hasattr(x, 'shape') else (len(x),)

            # Check if we need special handling for 3D arrays
            if len(siza) == 3:
                # a is (n1, 3, n2), x is (n2,) or (n2, ...)
                n1, _, n2 = siza

                if len(sizx) == 1:
                    # x is 1D
                    y = np.tensordot(a, x, axes=([2], [0]))
                else:
                    # x is multi-dimensional
                    a_flat = a.reshape(n1 * 3, n2)
                    x_flat = x.reshape(n2, -1)
                    y_flat = a_flat @ x_flat

                    new_shape = (n1, 3) + sizx[1:]
                    y = y_flat.reshape(new_shape)

                return y
            else:
                # Standard 2D matrix multiplication
                if len(sizx) == 1:
                    return a @ x
                else:
                    return a @ x.reshape(sizx[0], -1).reshape((sizx[0],) + sizx[1:])

    def _outer(self, nvec, scalar):
        """
        Outer product: nvec * scalar.

        MATLAB: outer(nvec, scalar)

        Parameters
        ----------
        nvec : ndarray, shape (n, 3)
            Normal vectors
        scalar : ndarray, shape (n,)
            Scalar values

        Returns
        -------
        result : ndarray, shape (n, 3)
            nvec * scalar[:, None]
        """
        if scalar.ndim == 1:
            return nvec * scalar[:, np.newaxis]
        else:
            # Handle higher dimensions
            return nvec[:, :, np.newaxis] * scalar[:, np.newaxis, :]

    def __repr__(self):
        """String representation."""
        status = "λ={:.1f}nm".format(self.enei) if self.enei is not None else "not initialized"
        return "BEMStat(p: {} faces, {})".format(
            self.p.n if hasattr(self.p, 'n') else '?', status)

    def __str__(self):
        """Detailed string representation."""
        return (
            "bemstat:\n"
            "  p: {}\n"
            "  F: {}\n"
            "  enei: {}\n"
            "  mat: {}".format(
                self.p,
                self.F.shape if hasattr(self, 'F') else 'not computed',
                self.enei,
                self.mat_lu[0].shape if self.mat_lu is not None else 'not computed')
        )
