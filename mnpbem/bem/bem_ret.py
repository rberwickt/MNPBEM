"""
BEM solver for full Maxwell equations (retarded).

Given an external excitation, BEMRet computes the surface charges
and currents such that the boundary conditions of Maxwell's equations
are fulfilled.

Reference:
    Garcia de Abajo and Howie, PRB 65, 115418 (2002)

Matches MATLAB MNPBEM implementation exactly.
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve

import os
from ..utils.gpu import (
    lu_factor_dispatch, lu_solve_dispatch, lu_solve_native, matmul_dispatch,
    eye_like_lu, lu_backend, to_host, is_cupy_array,
)
from ..greenfun import CompGreenRet, CompStruct

# Lane A2 (M4 GPU Phase 2): when MNPBEM_GPU=1 and cupy is importable, the
# BEM matrix assembly (G/H differences, dense inverse, L/Sigma/Delta GEMMs,
# nvec*nvec^T, k^2 magnetic-coupling) is performed end-to-end on the GPU
# without round-tripping intermediates back to host.  Falls back to CPU
# numpy when cupy is unavailable or MNPBEM_GPU != 1.
try:
    import cupy as _cp_a2  # type: ignore
    _CUPY_OK_A2 = True
except Exception:
    _cp_a2 = None  # type: ignore
    _CUPY_OK_A2 = False


def _bem_assembly_use_gpu() -> bool:
    if not _CUPY_OK_A2:
        return False
    return os.environ.get('MNPBEM_GPU', '0') == '1'


def _clear_green_distance_cache(green) -> None:
    """Drop GreenRetRefined ``_d_cache`` arrays so the GPU is freed.

    Each per-block GreenRetRefined caches up to 7 full (N, N) distance
    arrays (~25 GB total at 15072 faces).  After the Green-eval phase
    of ``_init_gpu_assemble`` they are dead weight that collides with
    the LU/Sigma pipeline (the historical 47.8 GB OOM).  Clearing them
    lets the next wavelength rebuild the cache cheaply — the distance
    kernels are O(N^2) memory-bandwidth-bound (~0.5 s at 15072 faces),
    negligible next to the O(N^3) LU.
    """
    g2d = getattr(green, 'g', None)
    if g2d is None:
        return
    # ACACompGreenRet wraps a CompGreenRet at ``.g``; unwrap one level.
    if not isinstance(g2d, (list, tuple)) and hasattr(g2d, 'g'):
        g2d = g2d.g
    if not isinstance(g2d, (list, tuple)):
        return
    for row in g2d:
        if not isinstance(row, (list, tuple)):
            continue
        for blk in row:
            refined = getattr(blk, 'refined', None) if blk is not None else None
            if refined is not None and getattr(refined, '_d_cache', None) is not None:
                refined._d_cache = None


def _vram_share_lu_kwargs() -> dict:
    """Read MNPBEM_VRAM_SHARE_* env vars and return kwargs for lu_factor_dispatch.

    Returns ``{}`` when VRAM-share is not enabled (n_gpus<=1).
    """
    if os.environ.get('MNPBEM_VRAM_SHARE', '0') != '1':
        return {}
    n_gpus = int(os.environ.get('MNPBEM_VRAM_SHARE_GPUS', '1'))
    if n_gpus <= 1:
        return {}
    backend = os.environ.get('MNPBEM_VRAM_SHARE_BACKEND', 'cusolvermg')
    return {'n_gpus': n_gpus, 'backend': backend}


def _bem_native_gpu() -> bool:
    """Phase 3: keep BEMRet matrices (Sigma1, L1/L2) on device when set.

    Default ON when MNPBEM_GPU=1 (Phase 3 native path verified bit-identical).
    Set MNPBEM_GPU_NATIVE=0 to opt out (escape hatch for regression suspicion).
    """
    if not _CUPY_OK_A2:
        return False
    if os.environ.get('MNPBEM_GPU', '0') != '1':
        return False
    return os.environ.get('MNPBEM_GPU_NATIVE', '1') != '0'


def _vram_share_active() -> bool:
    """Detect whether the distributed multi-GPU build path is active.

    The distributed *build* path (not just LU dispatch) is gated by all
    of these conditions together:
    - ``MNPBEM_GPU=1``                       (GPU mode enabled)
    - ``MNPBEM_VRAM_SHARE=1``                (master switch on, default '0')
    - ``MNPBEM_VRAM_SHARE_DISTRIBUTED=1``    (distributed-build gate, default '0')
    - ``MNPBEM_VRAM_SHARE_GPUS>=2``           (>=2 GPUs requested)
    - cupy is importable                      (device available at all)

    The DISTRIBUTED gate matches the pattern other solvers
    (bem_ret_iter, bem_ret_layer_iter, bem_ret_layer, bem_ret_mirror)
    use, so users on the existing LU-only multi-GPU path
    (``MNPBEM_VRAM_SHARE=1`` + ``MNPBEM_VRAM_SHARE_GPUS=N`` without the
    DISTRIBUTED flag) do not accidentally fall into the column-split
    Green-function build.

    When False, callers fall back to ``_init_gpu_assemble`` (single-GPU
    cupy-eager build) or ``_init_host_assemble`` (the pure-CPU legacy
    path below).
    """
    if not _CUPY_OK_A2:
        return False
    if os.environ.get('MNPBEM_GPU', '0') != '1':
        return False
    if os.environ.get('MNPBEM_VRAM_SHARE', '0').strip() != '1':
        return False
    if os.environ.get('MNPBEM_VRAM_SHARE_DISTRIBUTED', '0').strip() != '1':
        return False
    raw = os.environ.get('MNPBEM_VRAM_SHARE_GPUS', '').strip()
    if not raw:
        return False
    try:
        n_gpus = int(raw)
    except ValueError:
        return False
    return n_gpus >= 2


class BEMRet(object):
    """
    BEM solver for full Maxwell equations (retarded).

    Solves the boundary element method equations in the retarded regime
    to find surface charges and currents given external fields.

    Parameters
    ----------
    p : ComParticle
        Composite particle with geometry and material properties
    enei : float, optional
        Photon energy (eV) or wavelength (nm) for pre-initialization

    Attributes
    ----------
    p : ComParticle
        The particle object
    enei : float or None
        Current wavelength/energy (None if not initialized)
    k : float or None
        Wavenumber in vacuum (2π/λ)

    Notes
    -----
    The BEM equations for retarded case are more complex than quasistatic,
    involving coupled equations for surface charges (σ) and currents (h).

    The implementation follows Garcia de Abajo and Howie, PRB 65, 115418 (2002),
    Equations (19-22).

    MATLAB convention (bemret/private/initmat.m):
        G1 = g{1,1}.G(enei) - g{2,1}.G(enei)  # inside Green function
        G2 = g{2,2}.G(enei) - g{1,2}.G(enei)  # outside Green function
        H1 = g{1,1}.H1(enei) - g{2,1}.H1(enei)
        H2 = g{2,2}.H2(enei) - g{1,2}.H2(enei)

    For single particle, g{1,2} and g{2,1} are typically 0.

    Key matrices:
        G1, G2 : Green functions (inside/outside) with different wavenumbers
        H1, H2 : Surface derivatives with ±2π terms
        L1, L2 : G * ε * G^(-1) matrices (or just ε for single particle)
        Sigma1, Sigma2 : H * G^(-1) matrices
        Deltai : inv(Sigma1 - Sigma2)
        Sigmai : Inverse of combined Sigma matrix

    Examples
    --------
    >>> from mnpbem import EpsConst, EpsTable, trisphere, ComParticle
    >>> from mnpbem.bem import BEMRet
    >>>
    >>> # Create gold sphere
    >>> eps_tab = [EpsConst(1.0), EpsTable('gold.dat')]
    >>> sphere = trisphere(144, 10.0)
    >>> p = ComParticle(eps_tab, [sphere], [[2, 1]])
    >>>
    >>> # Create BEM solver
    >>> bem = BEMRet(p)
    >>>
    >>> # Initialize at specific wavelength
    >>> bem.init(600.0)
    """

    # Class constants
    # MATLAB: @bemret line 10-13
    name = 'bemsolver'
    needs = {'sim': 'ret'}

    def __init__(self, p, enei=None, **options):
        """
        Initialize BEM solver for retarded approximation.

        Parameters
        ----------
        p : ComParticle
            Composite particle
        enei : float, optional
            Photon energy (eV) or wavelength (nm) for pre-initialization
        **options : dict
            Additional options forwarded to CompGreenRet. Special keys:

            refun : callable, optional
                User-supplied Green-function refinement hook with signature
                ``fun(obj, g, f) -> (g, f)``. Applied to the assembled
                ``(G1, H1)`` and ``(G2, H2)`` Green matrix pairs after each
                ``init(enei)`` so cover-layer / nonlocal effective-layer
                geometries can inject polar-integration corrections.
                MATLAB equivalent: ``bemsolver(p, op, 'refun', ...)``
                (see ``Greenfun/@greenret/private/init.m`` line 187).
            schur : bool or 'auto', optional
                Activate Schur-complement elimination of EpsNonlocal
                cover-layer faces from the final Sigma matrix (v1.2.0).
                ``True`` / ``'auto'`` enables when a cover layer is
                detected; ``False`` (default) keeps the full BEM matrix.
                Only the dense path supports Schur in v1.2.0; iterative
                solvers raise NotImplementedError.
        """
        if p is None:
            raise ValueError(
                "BEMRet: 'p' must be a ComParticle (or compatible particle "
                "object), got None.")
        if not (hasattr(p, 'pos') and hasattr(p, 'nvec') and hasattr(p, 'eps')):
            raise TypeError(
                "BEMRet: 'p' must expose ComParticle-like attributes "
                "(pos, nvec, eps); got {!r}.".format(type(p).__name__))

        self.p = p
        self.enei = None

        # BEM matrices (initialized on demand)
        self.k = None
        self.nvec = None
        self.eps1 = None
        self.eps2 = None
        self.G1_lu = None
        self.G2_lu = None
        self.L1 = None
        self.L2 = None
        self.Sigma1 = None
        self.Delta_lu = None
        self.Sigma_lu = None

        # Optional user-supplied refinement hook (e.g. coverlayer.refine).
        # Pulled out of `options` so it is not passed down to CompGreenRet
        # (CompGreenRet itself does not yet handle 'refun'; refinement is
        # applied at the BEM-matrix level in init() / _init_gpu_assemble()).
        self.options = dict(options)
        self._refun = self.options.pop('refun', None)

        # Optional Schur-complement reduction (v1.2.0).
        # When enabled and an EpsNonlocal cover layer is detected, the final
        # Sigma matrix (the dominant LU factor in BEMRet.solve) is reduced
        # by eliminating the shell-face block; the full sigma vector is
        # reconstructed at the end of the solve.
        self._schur_opt = self.options.pop('schur', False)
        self._schur_active = False
        self._shell_idx = None
        self._core_idx = None
        self._schur_reduce_rhs = None
        self._schur_recover = None

        # Green function object (for field/potential computation).
        # MATLAB bemret/private/init.m line 26 builds compgreenret immediately
        # in the constructor, snapshotting the particle's quadrature state at
        # bemsolver-creation time. Python must do the same; otherwise an
        # excitation object created afterwards (e.g. EELSRet with refine=2)
        # could mutate p.quad and the BEM Green matrix would silently use the
        # refined rule. See Wave 22 Track A for full diagnosis.
        self.g = CompGreenRet(self.p, self.p, **self.options)

        # Initialize at specific energy if provided
        if enei is not None:
            self.init(enei)

    def init(self, enei):
        """
        Initialize BEM solver for specific wavelength/energy.

        Computes all necessary matrices for solving the BEM equations.
        Follows MATLAB bemret/private/initmat.m exactly.

        Parameters
        ----------
        enei : float
            Photon energy (eV) or wavelength (nm)

        Returns
        -------
        self : BEMRet
            Returns self for chaining
        """
        # Skip if already initialized at this energy
        if self.enei is not None and np.isclose(self.enei, enei):
            return self

        # B-3 distributed multi-GPU build (v1.8): when MNPBEM_VRAM_SHARE_GPUS>=2
        # the BEM matrices are partitioned across N GPUs from the build phase
        # itself, never materializing a single-GPU (N, N) tile. This is the
        # real path for >12k-face dimers where the 47 GB Sigma exceeds the
        # 49 GB single-GPU cap. Falls back to the single-GPU fast path
        # below when distributed assembly is not requested or unavailable.
        # Schur option (v1.2.0) only supported on the CPU path today.
        if _vram_share_active() and not self._schur_opt:
            try:
                self._init_distributed_assemble(enei)
                return self
            except Exception as _dist_exc:
                import warnings as _w
                _w.warn(
                    '[warn] BEMRet distributed assembly failed ({}); '
                    'falling back to single-GPU / CPU.'.format(_dist_exc),
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Lane A2: route the heavy assembly through a cupy-eager fast path
        # when the GPU is enabled.  This keeps every intermediate
        # (G/H/inverse/L/Sigma/Delta/Sigma_combined/nvec_outer) on device,
        # avoiding the dimer-scale 6336^2 host<->device transfers that
        # dominated the baseline GPU run.  The bit-identical numpy path is
        # preserved as fallback below.
        # Schur option (v1.2.0) currently only supported on the CPU path;
        # the GPU path keeps Sigma_lu as a ('gpu', lu, piv) tuple which the
        # Schur reduce/recover pipeline does not yet round-trip through.
        if _bem_assembly_use_gpu() and not self._schur_opt:
            try:
                self._init_gpu_assemble(enei)
                return self
            except Exception as _gpu_exc:
                # Fallback: log via warning and resume the CPU path.
                import warnings as _w
                _w.warn(
                    '[warn] BEMRet GPU assembly path failed ({}); '
                    'falling back to CPU.'.format(_gpu_exc),
                    RuntimeWarning,
                    stacklevel=2,
                )

        self.enei = enei

        # Outer surface normals
        self.nvec = self.p.nvec

        # Wavenumber in vacuum
        self.k = 2 * np.pi / enei

        # Get dielectric functions and wavenumbers for each material
        # MATLAB: [~, k] = cellfun(@(eps)(eps(enei)), obj.p1.eps)
        eps_vals = []
        k_vals = []
        for eps_func in self.p.eps:
            eps, k = eps_func(enei)
            eps_vals.append(eps)
            k_vals.append(k)

        # For single particle: eps[0] = outside, eps[1] = inside
        # k_out = k_vals[0], k_in = k_vals[1]
        k_out = k_vals[0]  # outside (vacuum)
        k_in = k_vals[1]   # inside (metal)

        # Dielectric function values
        eps1_vals = self.p.eps1(enei)  # inside (nfaces,)
        eps2_vals = self.p.eps2(enei)  # outside (nfaces,)

        # Check if all values are the same (can use scalar)
        if np.allclose(eps1_vals, eps1_vals[0]) and np.allclose(eps2_vals, eps2_vals[0]):
            self.eps1 = eps1_vals[0]
            self.eps2 = eps2_vals[0]
        else:
            self.eps1 = np.diag(eps1_vals)
            self.eps2 = np.diag(eps2_vals)

        # Create Green function (single object for all material combinations)
        # MATLAB: obj.g = compgreenret(p, p, op)
        if not hasattr(self, 'g') or self.g is None:
            self.g = CompGreenRet(self.p, self.p)

        # Compute Green function matrices at this wavelength
        # MATLAB: G1 = g{1,1}.G(enei) - g{2,1}.G(enei)
        #         G2 = g{2,2}.G(enei) - g{1,2}.G(enei)
        # Use region-based indexing: 0=inside, 1=outside
        def _to_dense(x):
            x = x.full() if hasattr(x, 'full') and not isinstance(x, np.ndarray) else x
            # Bug 1 fix: if upstream Green-function returned a cupy array
            # (Lane A GPU path), coerce to host here.  The CPU init path
            # below mixes G1/H1 with host eps1/eps2/nvec; a cupy/numpy mix
            # would otherwise blow up at the first GEMM.
            return to_host(x) if is_cupy_array(x) else x
        G11 = _to_dense(self.g.eval(0, 0, 'G', enei))  # inside → inside
        G21 = _to_dense(self.g.eval(1, 0, 'G', enei))  # outside → inside
        G22 = _to_dense(self.g.eval(1, 1, 'G', enei))  # outside → outside
        G12 = _to_dense(self.g.eval(0, 1, 'G', enei))  # inside → outside

        # Compute differences (cross-terms are 0 for closed surface)
        G1 = G11 - G21 if not (isinstance(G21, int) and G21 == 0) else G11
        G2 = G22 - G12 if not (isinstance(G12, int) and G12 == 0) else G22

        # Same for H1 and H2
        H11 = _to_dense(self.g.eval(0, 0, 'H1', enei))
        H21 = _to_dense(self.g.eval(1, 0, 'H1', enei))
        H22 = _to_dense(self.g.eval(1, 1, 'H2', enei))
        H12 = _to_dense(self.g.eval(0, 1, 'H2', enei))

        H1_mat = H11 - H21 if not (isinstance(H21, int) and H21 == 0) else H11
        H2_mat = H22 - H12 if not (isinstance(H12, int) and H12 == 0) else H22

        # Optional user-supplied refinement (coverlayer.refine for
        # nonlocal cover-layer effects).  Applied to the assembled BEM
        # matrices BEFORE LU factorization so downstream solves use the
        # refined operators.  Mirrors MATLAB's per-greenret refun call at
        # Greenfun/@greenret/private/init.m line 187, but lifted to the
        # combined (G1, H1) / (G2, H2) pairs.
        if self._refun is not None:
            G1, H1_mat = self._refun(self.g, G1, H1_mat)
            G2, H2_mat = self._refun(self.g, G2, H2_mat)

        # LU factorizations of Green functions. Honor MNPBEM_VRAM_SHARE_* for
        # multi-GPU dispatch on large meshes (1 worker pools VRAM across GPUs).
        _lu_opts = _vram_share_lu_kwargs()
        self.G1_lu = lu_factor_dispatch(G1, **_lu_opts)
        self.G2_lu = lu_factor_dispatch(G2, **_lu_opts)

        # Compute inverses for intermediate matrix construction.
        # Bug 1 fix: when the LU lives on GPU, build the identity RHS on the
        # same device so lu_solve_native returns a cupy array; we then bring
        # the result back to host for matmul with G1/eps1 (host arrays).
        # This avoids the cupy/numpy mix that caused the matmul failure on
        # cusolverMg paths.
        eye_g1 = eye_like_lu(self.G1_lu, G1.shape[0])
        eye_g2 = eye_like_lu(self.G2_lu, G2.shape[0])
        G1i = to_host(lu_solve_native(self.G1_lu, eye_g1))
        G2i = to_host(lu_solve_native(self.G2_lu, eye_g2))

        # L matrices [Eq. (22)]
        # MATLAB: if all(obj.g.con{1,2} == 0), L1 = eps1; L2 = eps2
        # Depending on the connectivity matrix, L1 and L2 can be
        # full matrices, diagonal matrices, or scalars.
        # When cross-connectivity is zero, L simplifies to eps directly.
        # When cross-connectivity is non-zero AND eps is non-uniform,
        # the full G * eps * G^{-1} product is needed.
        # Note: when eps is scalar, G * eps * G^{-1} = eps * I, so L = eps
        # regardless of connectivity.
        if np.all(self.g.con[0][1] == 0) or np.isscalar(self.eps1):
            self.L1 = self.eps1
            self.L2 = self.eps2
        else:
            # Full case: L1 = G1 * eps1 * G1^(-1)
            self.L1 = G1 @ self.eps1 @ G1i
            self.L2 = G2 @ self.eps2 @ G2i

        # Sigma matrices [Eq. (21)]
        # Sigma1 = H1 * G1^(-1)
        # Sigma2 = H2 * G2^(-1)
        self.Sigma1 = H1_mat @ G1i
        Sigma2 = H2_mat @ G2i

        # LU factorization of Delta matrix
        Delta = self.Sigma1 - Sigma2
        self.Delta_lu = lu_factor_dispatch(Delta, **_lu_opts)
        eye_d = eye_like_lu(self.Delta_lu, Delta.shape[0])
        Deltai = to_host(lu_solve_native(self.Delta_lu, eye_d))

        # Combined Sigma matrix [Eq. (21,22)]
        # Sigma = Sigma1*L1 - Sigma2*L2 + k²*(L*Deltai)*(nvec*nvec')*L
        L = self.L1 - self.L2

        if np.isscalar(L):
            # Simplified case for uniform materials
            Sigma = self.Sigma1 * self.L1 - Sigma2 * self.L2
            # Add magnetic coupling term
            nvec_outer = self.nvec @ self.nvec.T  # (nfaces, nfaces)
            Sigma = Sigma + self.k**2 * L * (Deltai * nvec_outer) * L
        else:
            # Full matrix case
            nvec_outer = self.nvec @ self.nvec.T
            Sigma = (self.Sigma1 @ self.L1 - Sigma2 @ self.L2 +
                     self.k**2 * ((L @ Deltai) * nvec_outer) @ L)

        # Optional Schur-complement reduction over the EpsNonlocal cover-
        # layer face block of Sigma (v1.2.0). When active, Sigma_lu factors
        # the (M, M) reduced matrix (M = number of core faces) and the
        # cached _schur_reduce_rhs / _schur_recover callables are used in
        # solve() to keep the rest of the algorithm operating on full-size
        # vectors. VRAM-share kwargs (_lu_opts) propagate into the Sigma_lu
        # factor regardless of Schur path.
        self._schur_active = False
        if self._schur_opt:
            from .schur_helpers import (
                schur_eliminate, detect_shell_core_partition,
            )
            partition = detect_shell_core_partition(self.p)
            if partition is not None:
                shell_idx, core_idx = partition
                Sigma_eff, reduce_rhs, recover = schur_eliminate(
                        np.asarray(Sigma), shell_idx, core_idx)
                self._shell_idx = shell_idx
                self._core_idx = core_idx
                self._schur_reduce_rhs = reduce_rhs
                self._schur_recover = recover
                self._schur_active = True
                self.Sigma_lu = lu_factor_dispatch(Sigma_eff, **_lu_opts)
            else:
                self.Sigma_lu = lu_factor_dispatch(Sigma, **_lu_opts)
        else:
            self.Sigma_lu = lu_factor_dispatch(Sigma, **_lu_opts)

        return self

    def _init_gpu_assemble(self, enei):
        """Cupy-eager BEM matrix assembly (Lane A2 fast path).

        Builds G1/G2/H1/H2 on the host (Green-function evaluation still
        happens through ``self.g``), uploads them to the GPU once, and
        keeps every subsequent dense op (inverse, GEMM, nvec*nvec^T,
        Sigma combine, LU factor) on the device.  Only the small
        ``self.eps1``/``eps2`` scalars and the LU package metadata cross
        the PCIe boundary; ``self.G1_lu``/``self.G2_lu``/``self.Sigma_lu``
        are returned as ``('gpu', lu, piv)`` tuples that
        ``lu_solve_dispatch`` already understands, so downstream
        ``BEMRet.solve`` consumers don't have to change.

        Numerical contract: cuBLAS/cuSOLVER GEMMs differ from MKL only by
        floating-point rounding; the relative Frobenius error vs the CPU
        path is bounded by N * eps_machine ~ 1e-12 for dimer-scale meshes.
        """
        cp = _cp_a2
        from cupyx.scipy.linalg import lu_factor as _cp_lu_factor
        from cupyx.scipy.linalg import lu_solve as _cp_lu_solve

        # v1.7.1: per-wavelength fragmentation cleanup.  When the same
        # BEMRet instance is reused across a wavelength loop (the dominant
        # case for sweep runs), the cached GPU residents from the previous
        # wavelength stay alive until Python rebinds them mid-routine.  On
        # 12672-face Au@Ag dimers this leaves ~13 GB of stale LU/Sigma
        # buffers on the device just as the new wavelength's 8 G/H
        # intermediates (~20 GB) get uploaded — the resulting fragmentation
        # would push cupy's pool past the 49 GB cap around wl 20-25 and
        # trigger an OOM fallback to CPU.  Releasing the prior state up
        # front (and forcing the pool to compact) keeps the high-water
        # mark stable across the whole loop.
        #
        # v1.7.2: v1.7.1 held the high-water mark stable for ~10 wl on
        # 12672-face Au@Ag dimers, but the pool kept accumulating small
        # CUDA-async free residuals (each ``del`` returns blocks to the
        # pool only AFTER device sync, so a ``free_all_blocks`` call that
        # races ahead of the CUDA stream returns nothing).  By wl ~11-13
        # the device was 41 GB in use trying to add 2.6 GB → OOM.  The
        # strengthened cleanup: (1) ``deviceSynchronize`` BEFORE every
        # explicit ``free_all_blocks`` so the CUDA stream has finished
        # the prior ops and the blocks are truly idle.  (2) ``gc.collect``
        # to drop any lingering Python-level refs (tracebacks, numba
        # scratch) that hold cupy buffers.  No ``set_limit`` cap — the
        # legitimate wl-1 peak on 12672-face dimers is ~41 GB; capping
        # the pool below that just shifts the OOM forward to wl 1.
        _mempool_pre = cp.get_default_memory_pool()
        _pinned_pre = cp.get_default_pinned_memory_pool()
        for _attr in ('G1_lu', 'G2_lu', 'Delta_lu', 'Sigma_lu',
                      'Sigma1', 'L1', 'L2', 'nvec', 'eps1', 'eps2'):
            if hasattr(self, _attr):
                setattr(self, _attr, None)
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
        cp.cuda.runtime.deviceSynchronize()
        _mempool_pre.free_all_blocks()
        _pinned_pre.free_all_blocks()

        self.enei = enei
        self.nvec = self.p.nvec
        self.k = 2 * np.pi / enei

        # Path A instrumentation: per-stage wall-clock timing so we can
        # see exactly where the per-wavelength cost goes (green eval vs
        # LU factor vs Sigma assembly).  Printed at the end of init.
        import time as _time
        _t_stage = {'start': _time.perf_counter()}

        # Dielectric scalars / arrays on host (cheap).
        eps1_vals = self.p.eps1(enei)
        eps2_vals = self.p.eps2(enei)
        if np.allclose(eps1_vals, eps1_vals[0]) and np.allclose(
                eps2_vals, eps2_vals[0]):
            self.eps1 = eps1_vals[0]
            self.eps2 = eps2_vals[0]
            scalar_eps = True
        else:
            self.eps1 = np.diag(eps1_vals)
            self.eps2 = np.diag(eps2_vals)
            scalar_eps = False

        if not hasattr(self, 'g') or self.g is None:
            self.g = CompGreenRet(self.p, self.p)

        def _to_host(x):
            return x.full() if hasattr(x, 'full') and not isinstance(x, np.ndarray) else x

        # Pull G/H from the Green-function object (CPU-side eval) then
        # transfer to GPU.  Lane A's GPU Green-function path returns cupy
        # arrays directly; we accept either.
        def _to_dev(x):
            if x is None or (isinstance(x, int) and x == 0):
                return None
            if isinstance(x, cp.ndarray):
                return x
            return cp.asarray(_to_host(x))

        _mempool = cp.get_default_memory_pool()

        # Path A green-eval (v4): evaluate G1/G2/H1/H2 as FULL matrices
        # on the GPU.  The green path reuses its wavelength-independent
        # (N, N) distance cache — far cheaper per wavelength than
        # rebuilding column-tile slice caches 64x.  Each combined
        # matrix is moved to the host the instant it is formed, so the
        # device holds only {distance cache + one eval working set}
        # during this phase, never the cache AND all four accumulating
        # G/H (the old 47.8 GB OOM).  The distance cache is then dropped
        # so the LU/Sigma phase gets the whole device; the next
        # wavelength rebuilds it in ~0.5 s (O(N^2) bandwidth-bound).
        def _to_host_arr(x):
            if x is None or (isinstance(x, int) and x == 0):
                return None
            if isinstance(x, cp.ndarray):
                return cp.asnumpy(x)
            return _to_host(x)

        def _diff_host(a_raw, b_raw):
            a = _to_host_arr(a_raw)
            b = _to_host_arr(b_raw)
            if a is None and b is None:
                return None
            if b is None:
                return a
            if a is None:
                return -b
            return a - b

        # MATLAB: G1 = g{1,1}.G - g{2,1}.G ; G2 = g{2,2}.G - g{1,2}.G
        #         H1 = g{1,1}.H1 - g{2,1}.H1 ; H2 = g{2,2}.H2 - g{1,2}.H2
        G1_h = _diff_host(self.g.eval(0, 0, 'G', enei),
                          self.g.eval(1, 0, 'G', enei))
        cp.cuda.runtime.deviceSynchronize()
        _mempool.free_all_blocks()
        G2_h = _diff_host(self.g.eval(1, 1, 'G', enei),
                          self.g.eval(0, 1, 'G', enei))
        cp.cuda.runtime.deviceSynchronize()
        _mempool.free_all_blocks()
        H1_h = _diff_host(self.g.eval(0, 0, 'H1', enei),
                          self.g.eval(1, 0, 'H1', enei))
        cp.cuda.runtime.deviceSynchronize()
        _mempool.free_all_blocks()
        H2_h = _diff_host(self.g.eval(1, 1, 'H2', enei),
                          self.g.eval(0, 1, 'H2', enei))
        cp.cuda.runtime.deviceSynchronize()
        _mempool.free_all_blocks()

        # Drop the green distance cache (7 N×N arrays, ~25 GB) so the
        # LU/Sigma phase has the full device.
        _clear_green_distance_cache(self.g)
        cp.cuda.runtime.deviceSynchronize()
        _mempool.free_all_blocks()

        # Optional user-supplied refun (coverlayer.refine) — host numpy.
        if self._refun is not None:
            G1_h, H1_h = self._refun(self.g, G1_h, H1_h)
            G2_h, H2_h = self._refun(self.g, G2_h, H2_h)

        # Phase 2: upload the four host matrices for the on-device
        # LU/Sigma pipeline.  The Green distance cache is gone, so the
        # device is empty and the ~36 GB linear-algebra peak fits.
        G1 = cp.asarray(G1_h); del G1_h
        G2 = cp.asarray(G2_h); del G2_h
        H1_mat = cp.asarray(H1_h); del H1_h
        H2_mat = cp.asarray(H2_h); del H2_h
        # Path A precision lever: when MNPBEM_GPU_LOWPREC=1 the LU/Sigma
        # pipeline runs in complex64.  On the RTX A6000 fp64 is ~1.2
        # TFLOPS vs ~38 TFLOPS fp32 (1:32) — the measured Sigma stage
        # (687 s of a 750 s wavelength) is almost pure fp64 GEMM/solve,
        # so complex64 cuts it ~10-30x.  The four LU factors / Sigma1 /
        # L1 / L2 are cast back to complex128 before return so the
        # downstream solve() is unchanged; only the *values* carry
        # complex64 precision (verified against a complex128 reference).
        _lowprec = os.environ.get('MNPBEM_GPU_LOWPREC', '0') == '1'
        _wd = np.complex64 if _lowprec else np.complex128
        _wd_real = np.float32 if _lowprec else np.float64
        if _lowprec:
            G1 = G1.astype(_wd)
            G2 = G2.astype(_wd)
            H1_mat = H1_mat.astype(_wd)
            H2_mat = H2_mat.astype(_wd)
        cp.cuda.runtime.deviceSynchronize()
        _mempool.free_all_blocks()
        _t_stage['green'] = _time.perf_counter()

        # LU factor on device.  Keep G1/G2 on device for L1/L2 product.
        G1_dev = G1
        G2_dev = G2
        # cupyx lu_factor expects a fresh array (overwrite_a=True).
        G1c = G1_dev.copy()
        G2c = G2_dev.copy()
        lu1, piv1 = _cp_lu_factor(G1c, overwrite_a=True)
        lu2, piv2 = _cp_lu_factor(G2c, overwrite_a=True)
        self.G1_lu = ('gpu', lu1, piv1)
        self.G2_lu = ('gpu', lu2, piv2)
        # G1c/G2c are now consumed by the LU factor (overwrite_a); drop the
        # Python references so the pool can immediately recycle the buffer.
        del G1c, G2c
        cp.cuda.runtime.deviceSynchronize()
        _t_stage['lu_gh'] = _time.perf_counter()

        n = G1_dev.shape[0]
        I_dev = cp.eye(n, dtype=_wd)

        # L matrices
        # ACA wrappers proxy the underlying CompGreenRet via .g
        _gobj = self.g.g if hasattr(self.g, 'g') and hasattr(self.g.g, 'con') else self.g
        native = _bem_native_gpu()
        # v1.7 A1 fix: the prior code set ``L1_is_scalar=True`` whenever
        # ``con[0][1]==0`` OR ``scalar_eps``.  The former is also true
        # for a disjoint-particle dimer with NON-uniform eps, in which
        # case ``self.L1 = self.eps1`` is a host numpy DIAG matrix
        # (n,n), not a true scalar; downstream ``Sigma1_dev * self.L1``
        # then mixes cupy and numpy and raises TypeError.  The fix
        # restricts the scalar Sigma path to genuine Python scalars and
        # uploads diag L1/L2 to device when native mode is active so
        # all subsequent @-products stay on the GPU backend.
        if np.all(_gobj.con[0][1] == 0) or scalar_eps:
            L1_true_scalar = scalar_eps and np.isscalar(self.eps1)
            if native and not L1_true_scalar:
                self.L1 = cp.asarray(self.eps1).astype(_wd)
                self.L2 = cp.asarray(self.eps2).astype(_wd)
            else:
                self.L1 = self.eps1  # host scalar OR host numpy diag
                self.L2 = self.eps2
            # Avoid computing the full G1i / G2i inverses when L is scalar:
            # Sigma1 = H1 @ G1^-1 can be obtained directly via lu_solve on
            # the right (solve G1^T x^T = H1^T).  This trims two N^3 GEMMs
            # per wavelength.
            G1i_dev = None
            G2i_dev = None
            # G1/G2 dense matrices are no longer needed (only their LU
            # factors are used downstream); release ~5 GB on 12672-face dimer.
            del G1, G2, G1_dev, G2_dev
            cp.cuda.runtime.deviceSynchronize()
            _mempool.free_all_blocks()
        else:
            # Path A peak fix: process the G1-side and G2-side L matrices
            # sequentially instead of holding G1, G2, G1i, G2i, L1, L2 all
            # at once.  Building L1 (then freeing G1/G1i before touching
            # the G2 side) cuts the L-stage peak from ~11 to ~9 N x N
            # matrices — ~7 GB headroom on a 15072-face complex128 dimer.
            eps1_dev = cp.asarray(self.eps1).astype(_wd)
            G1i_dev = _cp_lu_solve((lu1, piv1), I_dev)
            L1_dev_full = G1_dev @ eps1_dev @ G1i_dev
            del eps1_dev, G1i_dev, G1, G1_dev
            cp.cuda.runtime.deviceSynchronize()
            _mempool.free_all_blocks()

            eps2_dev = cp.asarray(self.eps2).astype(_wd)
            G2i_dev = _cp_lu_solve((lu2, piv2), I_dev)
            L2_dev_full = G2_dev @ eps2_dev @ G2i_dev
            del eps2_dev, G2i_dev, G2, G2_dev
            cp.cuda.runtime.deviceSynchronize()
            _mempool.free_all_blocks()

            if native:
                # Phase 3: keep L1/L2 on device.
                self.L1 = L1_dev_full
                self.L2 = L2_dev_full
            else:
                self.L1 = cp.asnumpy(L1_dev_full)
                self.L2 = cp.asnumpy(L2_dev_full)
                del L1_dev_full, L2_dev_full
            # G1i/G2i were freed above; the Sigma stage must take the
            # ``lu_solve`` route (G1i_dev is None) rather than the explicit
            # inverse multiply.
            G1i_dev = None
            G2i_dev = None
            cp.cuda.runtime.deviceSynchronize()
            _mempool.free_all_blocks()
            L1_true_scalar = False

        # Sigma matrices: Sigma_i = H_i @ G_i^-1.
        # Use right-hand-side lu_solve when possible to skip the full
        # inverse construction.
        if G1i_dev is not None:
            Sigma1_dev = H1_mat @ G1i_dev
            Sigma2_dev = H2_mat @ G2i_dev
            # G1i/G2i (each ~2.5 GB) drop here; downstream uses only LU and
            # the Sigma products.
            del G1i_dev, G2i_dev
        else:
            # Sigma1 @ G1 = H1  =>  G1^T @ Sigma1^T = H1^T
            Sigma1_dev = _cp_lu_solve((lu1, piv1), H1_mat.T, trans=1).T
            Sigma2_dev = _cp_lu_solve((lu2, piv2), H2_mat.T, trans=1).T
        # H1_mat/H2_mat are no longer needed after Sigma1/Sigma2 are formed.
        del H1_mat, H2_mat
        cp.cuda.runtime.deviceSynchronize()
        _mempool.free_all_blocks()
        # Phase 3: keep Sigma1 on device when GPU_NATIVE is enabled so the
        # downstream solve() can do Sigma1 @ a entirely on the GPU.
        if native:
            self.Sigma1 = Sigma1_dev
        else:
            self.Sigma1 = cp.asnumpy(Sigma1_dev)

        # Delta = Sigma1 - Sigma2  (still on device)
        Delta_dev = Sigma1_dev - Sigma2_dev
        Dc = Delta_dev.copy()
        del Delta_dev
        lu_d, piv_d = _cp_lu_factor(Dc, overwrite_a=True)
        self.Delta_lu = ('gpu', lu_d, piv_d)
        del Dc
        Deltai_dev = _cp_lu_solve((lu_d, piv_d), I_dev)
        del I_dev

        if L1_true_scalar:
            # nvec*nvec^T built just before use to keep it off the peak.
            nvec_dev = cp.asarray(self.nvec).astype(_wd_real)
            nvec_outer_dev = nvec_dev @ nvec_dev.T
            del nvec_dev
            Sigma_dev = (Sigma1_dev * self.L1) - (Sigma2_dev * self.L2)
            L_scalar = self.L1 - self.L2
            Sigma_dev = Sigma_dev + (self.k ** 2) * L_scalar * (
                Deltai_dev * nvec_outer_dev) * L_scalar
            # Sigma2/Deltai/nvec_outer done; Sigma1 stays only if native.
            if not native:
                del Sigma1_dev
            del Sigma2_dev, Deltai_dev, nvec_outer_dev
        else:
            # Path A peak fix: accumulate Sigma term-by-term so the full
            # ``Sigma1@L1 - Sigma2@L2 + k²((L@Deltai)*nvec)@L`` expression
            # never holds all 7 operand matrices + cupy GEMM temporaries
            # at once.  Each source matrix is freed as soon as its term is
            # folded in, capping the Sigma-stage peak near 36 GB instead
            # of the ~50 GB single-expression peak that OOM'd at 15072
            # faces.  self.L1/self.L2 may be host numpy diag or device
            # cupy; ``cp.asarray`` is a no-op on cupy, an upload on numpy.
            L1_dev = cp.asarray(self.L1).astype(_wd)
            L2_dev = cp.asarray(self.L2).astype(_wd)
            Sigma_dev = Sigma1_dev @ L1_dev
            Sigma_dev -= Sigma2_dev @ L2_dev
            del Sigma2_dev
            if not native:
                del Sigma1_dev
            cp.cuda.runtime.deviceSynchronize()
            _mempool.free_all_blocks()
            # magnetic term: k² ((L @ Deltai) * nvec_outer) @ L
            L_dev = L1_dev - L2_dev
            del L1_dev, L2_dev
            mag = L_dev @ Deltai_dev
            del Deltai_dev
            # nvec*nvec^T built here (just before use) so it is not
            # resident through the Sigma1@L1 / Sigma2@L2 terms above.
            nvec_dev = cp.asarray(self.nvec).astype(_wd_real)
            nvec_outer_dev = nvec_dev @ nvec_dev.T
            del nvec_dev
            mag *= nvec_outer_dev
            del nvec_outer_dev
            mag = mag @ L_dev
            del L_dev
            Sigma_dev += (self.k ** 2) * mag
            del mag
        cp.cuda.runtime.deviceSynchronize()
        _mempool.free_all_blocks()

        # Final Sigma LU factor on device.
        Sc = Sigma_dev.copy()
        del Sigma_dev
        lu_s, piv_s = _cp_lu_factor(Sc, overwrite_a=True)
        self.Sigma_lu = ('gpu', lu_s, piv_s)
        del Sc
        # v1.7.2: the local lu/piv handles for G1/G2/Delta now alias the
        # device buffers stored in self.{G1,G2,Delta,Sigma}_lu, but the
        # Python frame still keeps them alive until function return.
        # Drop them explicitly so free_all_blocks below can compact the
        # pool down to the genuinely live set (4 LU factors + Sigma1 +
        # L1/L2 + eps + nvec) before the next wavelength enters
        # _init_gpu_assemble and re-runs the upfront cleanup.
        del lu1, piv1, lu2, piv2, lu_d, piv_d, lu_s, piv_s
        cp.cuda.runtime.deviceSynchronize()
        _mempool.free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        if _lowprec:
            # Cast the complex64 results back into complex128 storage so
            # the downstream solve() sees its expected dtype with no
            # dtype-mix errors.  The *values* keep complex64 precision —
            # this only widens the container.
            for _luattr in ('G1_lu', 'G2_lu', 'Delta_lu', 'Sigma_lu'):
                _tag = getattr(self, _luattr, None)
                if (isinstance(_tag, tuple) and len(_tag) == 3
                        and _tag[1] is not None
                        and hasattr(_tag[1], 'astype')):
                    setattr(self, _luattr,
                            (_tag[0], _tag[1].astype(np.complex128), _tag[2]))
            for _mattr in ('Sigma1', 'L1', 'L2'):
                _v = getattr(self, _mattr, None)
                if (_v is not None and hasattr(_v, 'astype')
                        and hasattr(_v, 'dtype')
                        and _v.dtype == np.complex64):
                    setattr(self, _mattr, _v.astype(np.complex128))
            cp.cuda.runtime.deviceSynchronize()
            _mempool.free_all_blocks()
        _t_stage['sigma'] = _time.perf_counter()

        _t_green = _t_stage['green'] - _t_stage['start']
        _t_lu = _t_stage['lu_gh'] - _t_stage['green']
        _t_sig = _t_stage['sigma'] - _t_stage['lu_gh']
        _t_tot = _t_stage['sigma'] - _t_stage['start']
        print('[timing] _init_gpu_assemble enei={:.1f}: green-eval={:.1f}s '
              'G1G2-LU={:.1f}s Sigma-stage={:.1f}s total={:.1f}s'.format(
                  enei, _t_green, _t_lu, _t_sig, _t_tot), flush=True)
        return self

    def _init_distributed_assemble(self, enei):
        """B-3 distributed multi-GPU BEM matrix assembly.

        Builds G1/G2/H1/H2 directly distributed across N GPUs (using
        ``DistributedMatrix.from_func`` + ``CompGreenRet.eval_block`` so
        each device computes only its own column tile). The four LU
        factorizations (G1, G2, Delta, Sigma) are then performed on the
        distributed tiles via ``cuSolverMg`` block-cyclic Getrf — no
        single-GPU (N, N) buffer is ever allocated.

        Memory characteristic (n_gpus=N):
        - Per-GPU peak: ~4 * N_faces^2 * 16 / n_gpus bytes (build phase
          holds G_ij + H_ij residuals before merge).
        - Host peak: ``Sigma1`` and ``Sigma`` materialized once for the
          host @ L1 / L2 products and the final Sigma factor input.

        Result residency
        ----------------
        - ``self.G1_lu``, ``self.G2_lu``, ``self.Delta_lu``, ``self.Sigma_lu``
          are stored as ``('mgpu', MultiGPULU_handle, None)`` tuples.
          ``lu_solve_dispatch`` already routes that tag through
          ``MultiGPULU.solve`` so downstream ``BEMRet.solve`` consumers
          don't change.
        - ``self.Sigma1``, ``self.L1``, ``self.L2``, ``self.nvec``,
          ``self.eps1``, ``self.eps2`` are host numpy arrays (no
          ``GPU_NATIVE`` shortcut on this path; the matrices are too
          large for a single device).

        Bit-identity contract (vs single-GPU ``_init_gpu_assemble``):
        cuBLAS/cuSolverMg arithmetic differs from the legacy CPU path by
        floating-point rounding (~N * eps_machine relative). The
        distributed path adds no extra rounding beyond what the
        per-block GEMM/LU already do, so the result matches the
        cuSolverMg single-GPU build to within Frobenius ~1e-12.
        """
        import gc as _gc
        from ..utils.distributed_matrix import DistributedMatrix
        cp = _cp_a2

        n_gpus = int(os.environ.get('MNPBEM_VRAM_SHARE_GPUS', '2'))
        # device_ids: honour MNPBEM_VRAM_SHARE_DEVICE_IDS, else default [0..n_gpus).
        devs_raw = os.environ.get('MNPBEM_VRAM_SHARE_DEVICE_IDS', '').strip()
        if devs_raw:
            try:
                device_ids = [int(x) for x in devs_raw.split(',') if x.strip()]
            except ValueError:
                device_ids = list(range(n_gpus))
        else:
            device_ids = list(range(n_gpus))
        assert len(device_ids) == n_gpus, \
            '[error] MNPBEM_VRAM_SHARE_DEVICE_IDS length must equal MNPBEM_VRAM_SHARE_GPUS'

        # v1.7-style upfront cleanup: when the same BEMRet is reused across
        # a wavelength sweep, release the previous wavelength's distributed
        # LU handles and host matrices before re-allocating.
        for _attr in ('G1_lu', 'G2_lu', 'Delta_lu', 'Sigma_lu',
                      'Sigma1', 'L1', 'L2', 'nvec', 'eps1', 'eps2'):
            old = getattr(self, _attr, None)
            if old is None:
                continue
            # 'mgpu' tag: close the MultiGPULU handle AND drop the
            # keepalive DistributedMatrix that owned the device tiles.
            if (isinstance(old, tuple) and len(old) == 3
                    and old[0] == 'mgpu' and old[1] is not None):
                handle = old[1]
                try:
                    handle.close()
                except Exception:
                    pass
                # The handle's tiles were owned by a DistributedMatrix
                # attached as ``_distmat_keepalive``. Releasing it now
                # actually frees the per-GPU memory; the prior close()
                # only released IPIV / workspace / cusolverMg objects.
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
            setattr(self, _attr, None)
        _gc.collect()
        for d in device_ids:
            cp.cuda.runtime.setDevice(d)
            cp.cuda.runtime.deviceSynchronize()
            cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        self.enei = enei
        self.nvec = self.p.nvec
        self.k = 2 * np.pi / enei

        # Dielectric scalars / arrays on host.
        eps1_vals = self.p.eps1(enei)
        eps2_vals = self.p.eps2(enei)
        if (np.allclose(eps1_vals, eps1_vals[0])
                and np.allclose(eps2_vals, eps2_vals[0])):
            self.eps1 = eps1_vals[0]
            self.eps2 = eps2_vals[0]
            scalar_eps = True
        else:
            self.eps1 = np.diag(eps1_vals)
            self.eps2 = np.diag(eps2_vals)
            scalar_eps = False

        if not hasattr(self, 'g') or self.g is None:
            self.g = CompGreenRet(self.p, self.p)

        N = self.p.nfaces
        dtype = np.complex128

        # ------------------------------------------------------------------
        # cuSolverMg.solve numerical fix: the binary loses precision when
        # the RHS column count approaches N (rel ~ 5e-3 at ncol/N ~ 0.95).
        # Below ~2048 columns the result is bit-identical to CPU. To get
        # an accurate N x N inverse (needed for Sigma_i = H_i @ G_i^{-1}
        # and Deltai), we chunk the solve into smaller column blocks.
        # Configurable via MNPBEM_VRAM_SHARE_SOLVE_CHUNK (default 1024).
        # ------------------------------------------------------------------
        try:
            solve_chunk = int(os.environ.get(
                'MNPBEM_VRAM_SHARE_SOLVE_CHUNK', '1024'))
        except (TypeError, ValueError):
            solve_chunk = 1024
        if solve_chunk < 16:
            solve_chunk = 16

        def _mg_solve_chunked(mg_handle, B_full):
            # B_full: numpy ndarray shape (N,) or (N, nrhs). For nrhs<=
            # solve_chunk we hand it through directly; otherwise we split
            # along columns and concatenate.
            if B_full.ndim == 1 or B_full.shape[1] <= solve_chunk:
                return mg_handle.solve(B_full)
            n_rows, n_cols = B_full.shape
            out = np.empty_like(B_full)
            for c0 in range(0, n_cols, solve_chunk):
                c1 = min(c0 + solve_chunk, n_cols)
                out[:, c0:c1] = mg_handle.solve(
                    np.ascontiguousarray(B_full[:, c0:c1]))
            return out

        # --------------------------------------------------------------
        # Per-block eval_func helpers. Each builds one column tile by
        # calling CompGreenRet.eval_block for the two cross-region terms
        # that contribute to the corresponding BEM matrix and subtracting
        # in place. ``eval_block`` returns either a host numpy ndarray or
        # a cupy ndarray (when MNPBEM_GPU_NATIVE=1 and the green-function
        # path supports it). DistributedMatrix.from_func handles either
        # — it uploads numpy returns to the owning GPU automatically.
        # The eval_func runs INSIDE ``cp.cuda.Device(gpu_idx)``, so the
        # returned cupy array (when present) ends up on the right device.
        # --------------------------------------------------------------
        compg = self.g
        # Use a helper that handles the closed-surface case where the
        # cross-region eval returns 0 (single particle, no junction).
        # When CompGreenRet returns cupy for one term and numpy zeros for
        # the other (e.g. con[0,0]!=0 produces cupy, con[1,0]==0 produces
        # numpy zeros), normalize both to the same backend before the
        # subtraction. The eval_func runs inside ``cp.cuda.Device(gpu_idx)``
        # so the upload lands on the right device.
        def _safe_eval_block(i, j, key, c0, c1):
            blk = compg.eval_block(i, j, key, enei, c0, c1)
            if isinstance(blk, int) and blk == 0:
                return np.zeros((N, c1 - c0), dtype=dtype)
            return blk

        def _diff(a, b):
            # Coerce to the cupy backend when either operand is cupy.
            if isinstance(a, cp.ndarray) or isinstance(b, cp.ndarray):
                if not isinstance(a, cp.ndarray):
                    a = cp.asarray(a, dtype=dtype)
                if not isinstance(b, cp.ndarray):
                    b = cp.asarray(b, dtype=dtype)
            return a - b

        def _eval_G1(gpu_idx, c0, c1):
            a = _safe_eval_block(0, 0, 'G', c0, c1)
            b = _safe_eval_block(1, 0, 'G', c0, c1)
            return _diff(a, b)

        def _eval_G2(gpu_idx, c0, c1):
            a = _safe_eval_block(1, 1, 'G', c0, c1)
            b = _safe_eval_block(0, 1, 'G', c0, c1)
            return _diff(a, b)

        def _eval_H1(gpu_idx, c0, c1):
            a = _safe_eval_block(0, 0, 'H1', c0, c1)
            b = _safe_eval_block(1, 0, 'H1', c0, c1)
            return _diff(a, b)

        def _eval_H2(gpu_idx, c0, c1):
            a = _safe_eval_block(1, 1, 'H2', c0, c1)
            b = _safe_eval_block(0, 1, 'H2', c0, c1)
            return _diff(a, b)

        # --------------------------------------------------------------
        # 1) Build G1, G2 distributed -> factor in place.
        # --------------------------------------------------------------
        G1_dm = DistributedMatrix.from_func(
            shape=(N, N), dtype=dtype, n_gpus=n_gpus,
            device_ids=device_ids, eval_func=_eval_G1)

        # Optional user refun (coverlayer.refine). Refun is host numpy
        # only; we have to gather/refine/scatter. Memory cost: one host
        # (N, N) ~ 3.6 GB at 15k faces — fine on 503 GB host RAM.
        if self._refun is not None:
            G1_h = G1_dm.to_host()
            # H1 needed too for refun. Build H1 first below.

        # Factor G1 in place. After this call, the distributed tiles of
        # G1_dm hold the L/U factors; do not reuse G1_dm as a Green
        # function thereafter (the math equations use only G1^{-1}).
        # IMPORTANT: ``G1_mglu`` only holds a ctypes pointer array into
        # ``G1_dm.local_arrays``. We must keep ``G1_dm`` alive (or its
        # cupy tiles will be GC'd and the LU pointers become dangling).
        # Attach ``G1_dm`` to ``G1_mglu`` so the lifetimes are joined.
        G1_mglu = G1_dm.lu_factor()
        G1_mglu._distmat_keepalive = G1_dm  # type: ignore[attr-defined]
        self.G1_lu = ('mgpu', G1_mglu, None)

        G2_dm = DistributedMatrix.from_func(
            shape=(N, N), dtype=dtype, n_gpus=n_gpus,
            device_ids=device_ids, eval_func=_eval_G2)
        if self._refun is not None:
            G2_h = G2_dm.to_host()
        G2_mglu = G2_dm.lu_factor()
        G2_mglu._distmat_keepalive = G2_dm  # type: ignore[attr-defined]
        self.G2_lu = ('mgpu', G2_mglu, None)

        # --------------------------------------------------------------
        # 2) Build H1, H2 distributed -> gather to host for Sigma_i.
        #    Sigma_i = H_i @ G_i^{-1}; computed via right-side solve:
        #        Sigma_i^T = (G_i^T)^{-1} @ H_i^T
        #    cuSolverMg supports trans='T' on the solve, so we feed
        #    H_i^T as the RHS (gathered to host) and recover Sigma_i
        #    via another transpose.
        # --------------------------------------------------------------
        H1_dm = DistributedMatrix.from_func(
            shape=(N, N), dtype=dtype, n_gpus=n_gpus,
            device_ids=device_ids, eval_func=_eval_H1)
        H1_host = H1_dm.to_host()
        H1_dm.free()
        del H1_dm

        if self._refun is not None:
            G1_h, H1_host = self._refun(self.g, G1_h, H1_host)
            # Re-factor G1 with refined data (overwrite mgpu handle).
            # Free the old LU first to release distributed memory.
            try:
                G1_mglu.close()
            except Exception:
                pass
            # Also free the original G1_dm tiles (its ndarrays kept the
            # device buffers alive while the old LU pointed at them).
            try:
                G1_dm.free()
            except Exception:
                pass
            G1_dm_re = DistributedMatrix.from_host(
                G1_h, n_gpus=n_gpus, device_ids=device_ids)
            G1_mglu = G1_dm_re.lu_factor()
            G1_mglu._distmat_keepalive = G1_dm_re  # keep buffers alive
            self.G1_lu = ('mgpu', G1_mglu, None)
            del G1_h

        H2_dm = DistributedMatrix.from_func(
            shape=(N, N), dtype=dtype, n_gpus=n_gpus,
            device_ids=device_ids, eval_func=_eval_H2)
        H2_host = H2_dm.to_host()
        H2_dm.free()
        del H2_dm

        if self._refun is not None:
            G2_h, H2_host = self._refun(self.g, G2_h, H2_host)
            try:
                G2_mglu.close()
            except Exception:
                pass
            try:
                G2_dm.free()
            except Exception:
                pass
            G2_dm_re = DistributedMatrix.from_host(
                G2_h, n_gpus=n_gpus, device_ids=device_ids)
            G2_mglu = G2_dm_re.lu_factor()
            G2_mglu._distmat_keepalive = G2_dm_re
            self.G2_lu = ('mgpu', G2_mglu, None)
            del G2_h

        # Sigma1 = H1 @ G1^{-1}.
        # cusolverMgGetrs supports only trans='N' in practice (the 'T' /
        # 'C' codes return CUSOLVER_STATUS_INVALID_VALUE on the public
        # ABI). So compute G1^{-1} explicitly via solve(I) once and then
        # host-multiply with H1. Memory: G1_inv host ~ 3.6 GB at 15k
        # faces; H1 host already materialized; on a 503 GB host RAM
        # node both fit comfortably and the (n,n)@(n,n) matmul lands on
        # MKL with ~7 TFLOPS = ~6 s at N=15k.
        I_host = np.eye(N, dtype=dtype)
        G1_inv = _mg_solve_chunked(G1_mglu, I_host)
        Sigma1_host = np.ascontiguousarray(H1_host @ G1_inv)
        del G1_inv, H1_host

        G2_inv = _mg_solve_chunked(G2_mglu, I_host)
        Sigma2_host = np.ascontiguousarray(H2_host @ G2_inv)
        del G2_inv, H2_host, I_host
        _gc.collect()

        # G1_dm / G2_dm are now attached to G1_mglu / G2_mglu via the
        # ``_distmat_keepalive`` attribute (the LU handle's ctypes
        # pointer array references the cupy tiles in-place; dropping
        # the DistributedMatrix wrapper before the LU handle is freed
        # would dangle the pointers). Leave the keepalives wired and
        # the local Python refs go out of scope on function return.
        del G1_dm, G2_dm

        # Store Sigma1 as host for downstream solve (matches the CPU
        # path's residency contract; the dimer-scale matmuls Sigma1 @ a
        # in solve() run on host where there is ample RAM).
        self.Sigma1 = Sigma1_host

        # L matrices [Eq. (22)]. For closed surfaces (con[0,1]==0) or
        # scalar eps, L_i = eps_i directly (no Sigma1-style invert).
        # Use the underlying CompGreenRet via the ACA-aware proxy.
        _gobj = self.g.g if (hasattr(self.g, 'g')
                             and hasattr(self.g.g, 'con')) else self.g
        if np.all(_gobj.con[0][1] == 0) or scalar_eps:
            self.L1 = self.eps1
            self.L2 = self.eps2
        else:
            # Full case: L1 = G1 @ eps1 @ G1^{-1}.  Compute via two
            # distributed solves: X = G1^{-1} @ eps1 (n,n) then
            # L1 = G1 @ X. Gather G1 once to host for the second matmul
            # (we lost the dense G1 to the in-place LU; rebuild it via
            # one extra distributed assembly when refun did not retain
            # a host copy).
            if self._refun is not None:
                G1_h_for_L = G1_h
                G2_h_for_L = G2_h
            else:
                _tmp_G1 = DistributedMatrix.from_func(
                    shape=(N, N), dtype=dtype, n_gpus=n_gpus,
                    device_ids=device_ids, eval_func=_eval_G1)
                G1_h_for_L = _tmp_G1.to_host()
                _tmp_G1.free()
                _tmp_G2 = DistributedMatrix.from_func(
                    shape=(N, N), dtype=dtype, n_gpus=n_gpus,
                    device_ids=device_ids, eval_func=_eval_G2)
                G2_h_for_L = _tmp_G2.to_host()
                _tmp_G2.free()
            X1 = _mg_solve_chunked(G1_mglu, np.ascontiguousarray(self.eps1))
            X2 = _mg_solve_chunked(G2_mglu, np.ascontiguousarray(self.eps2))
            self.L1 = G1_h_for_L @ X1
            self.L2 = G2_h_for_L @ X2
            del G1_h_for_L, G2_h_for_L, X1, X2
            _gc.collect()

        # --------------------------------------------------------------
        # 3) Delta = Sigma1 - Sigma2 -> distributed LU factor.
        # --------------------------------------------------------------
        Delta_host = Sigma1_host - Sigma2_host
        # Move Delta to distributed tiles and factor.
        Delta_dm = DistributedMatrix.from_host(
            Delta_host, n_gpus=n_gpus, device_ids=device_ids)
        Delta_mglu = Delta_dm.lu_factor()
        Delta_mglu._distmat_keepalive = Delta_dm
        self.Delta_lu = ('mgpu', Delta_mglu, None)
        del Delta_dm  # local ref dropped; keepalive holds device tiles

        # Deltai needed for Sigma combine. Solve on host (one (N, N)
        # RHS, chunked to work around the MG-LU multi-column solve
        # precision regression — see _mg_solve_chunked).
        Deltai_host = _mg_solve_chunked(Delta_mglu, np.eye(N, dtype=dtype))
        Deltai_host = np.ascontiguousarray(Deltai_host)

        # --------------------------------------------------------------
        # 4) Combine Sigma matrix (host arithmetic).
        # --------------------------------------------------------------
        nvec_outer = self.nvec @ self.nvec.T

        if np.isscalar(self.L1) and np.isscalar(self.L2):
            L_scalar = self.L1 - self.L2
            Sigma_host = (Sigma1_host * self.L1) - (Sigma2_host * self.L2)
            Sigma_host = Sigma_host + (self.k ** 2) * L_scalar * (
                Deltai_host * nvec_outer) * L_scalar
        else:
            L_diff_host = self.L1 - self.L2
            Sigma_host = (Sigma1_host @ self.L1
                          - Sigma2_host @ self.L2
                          + (self.k ** 2)
                            * ((L_diff_host @ Deltai_host) * nvec_outer)
                            @ L_diff_host)
            del L_diff_host

        del Sigma2_host, Deltai_host, nvec_outer
        _gc.collect()

        # --------------------------------------------------------------
        # 5) Distribute Sigma -> LU factor; this is the dominant solver
        #    matrix in BEMRet.solve.
        # --------------------------------------------------------------
        Sigma_dm = DistributedMatrix.from_host(
            Sigma_host, n_gpus=n_gpus, device_ids=device_ids)
        del Sigma_host
        _gc.collect()
        Sigma_mglu = Sigma_dm.lu_factor()
        Sigma_mglu._distmat_keepalive = Sigma_dm
        self.Sigma_lu = ('mgpu', Sigma_mglu, None)
        del Sigma_dm  # local ref dropped; keepalive holds device tiles

        # Final sync + pool compaction across all participating devices.
        for d in device_ids:
            cp.cuda.runtime.setDevice(d)
            cp.cuda.runtime.deviceSynchronize()
            cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        return self

    def _excitation(self, exc):
        """
        Process excitation to get phi, a, alpha, De.

        MATLAB: bemret/private/excitation.m

        Parameters
        ----------
        exc : dict
            Excitation with fields phi1, phi2, a1, a2, phi1p, phi2p, a1p, a2p

        Returns
        -------
        phi, a, alpha, De : ndarray
            Processed excitation variables for BEM equations
        """
        enei = exc['enei']

        # Default values for potentials
        nfaces = self.p.nfaces

        # Helper to get field with default of 0
        def get_field(name, default_shape=None):
            val = exc.get(name, 0)
            if isinstance(val, np.ndarray):
                return val
            elif val == 0 and default_shape is not None:
                return np.zeros(default_shape, dtype=complex)
            return val

        # Get potential values with defaults of 0
        phi1 = get_field('phi1')
        phi1p = get_field('phi1p')
        a1 = get_field('a1')
        a1p = get_field('a1p')
        phi2 = get_field('phi2')
        phi2p = get_field('phi2p')
        a2 = get_field('a2')
        a2p = get_field('a2p')

        # Wavenumber of light in vacuum
        k = 2 * np.pi / enei

        # Dielectric functions
        eps1 = self.p.eps1(enei)  # (nfaces,)
        eps2 = self.p.eps2(enei)  # (nfaces,)

        # Outer surface normal
        nvec = self.nvec

        # External excitation - Garcia de Abajo and Howie, PRB 65, 115418 (2002)

        # Eqs. (10,11): potential jumps
        phi = self._subtract(phi2, phi1)
        a = self._subtract(a2, a1)

        # Eq. (15): alpha = a2p - a1p - 1i*k*(outer(nvec, phi2)*eps2 - outer(nvec, phi1)*eps1)
        outer_term2 = self._outer_eps(nvec, phi2, eps2)
        outer_term1 = self._outer_eps(nvec, phi1, eps1)
        alpha = self._subtract(a2p, a1p) - 1j * k * self._subtract(outer_term2, outer_term1)

        # Eq. (18): De = eps2*phi2p - eps1*phi1p - 1i*k*(inner(nvec,a2)*eps2 - inner(nvec,a1)*eps1)
        matmul_term2 = self._matmul_eps(eps2, phi2p)
        matmul_term1 = self._matmul_eps(eps1, phi1p)
        inner_term2 = self._inner_eps(nvec, a2, eps2)
        inner_term1 = self._inner_eps(nvec, a1, eps1)

        De = self._subtract(matmul_term2, matmul_term1) - 1j * k * self._subtract(inner_term2, inner_term1)

        return phi, a, alpha, De

    def _subtract(self, a, b):
        """Subtract b from a, handling scalars and arrays."""
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return a - b
        elif isinstance(a, np.ndarray):
            if b == 0:
                return a
            return a - b
        elif isinstance(b, np.ndarray):
            if a == 0:
                return -b
            return a - b
        else:
            return a - b

    def _outer_eps(self, nvec, phi, eps):
        """Compute outer(nvec, phi) * eps. Returns (nfaces, 3, ...) matching phi trailing dims."""
        if isinstance(phi, np.ndarray):
            if phi.ndim == 1:
                return nvec * (phi * eps)[:, np.newaxis]  # (nfaces, 3)
            else:
                # phi is (nfaces, ...) with arbitrary trailing dims
                # Vectorized: nvec[:, :, None] * (phi*eps[:, None])[:, None, :]
                phi_eps = phi * eps[:, np.newaxis] if phi.ndim == 2 else phi * eps.reshape(-1, *([1] * (phi.ndim - 1)))
                return nvec[:, :, np.newaxis] * phi_eps[:, np.newaxis, :] if phi.ndim == 2 \
                    else nvec.reshape(nvec.shape[0], 3, *([1] * (phi.ndim - 1))) * phi_eps[:, np.newaxis]
        elif phi == 0:
            return 0
        else:
            return nvec * (phi * eps)

    def _inner_eps(self, nvec, a, eps):
        """Compute inner(nvec, a) * eps. Returns (nfaces, ...) matching a trailing dims after axis=1."""
        if isinstance(a, np.ndarray) and a.ndim >= 2:
            if a.ndim == 2:
                dot = np.sum(nvec * a, axis=1)  # (nfaces,)
                return dot * eps
            else:
                # a is (nfaces, 3, *trailing) — dot over axis=1, vectorized
                dot = np.einsum('ij,ij...->i...', nvec, a)
                return dot * eps.reshape(-1, *([1] * (a.ndim - 2)))
        elif isinstance(a, np.ndarray) and a.size == 0:
            return 0
        elif not isinstance(a, np.ndarray) and a == 0:
            return 0
        else:
            return 0

    def _matmul_eps(self, eps, phi_p):
        """Compute eps * phi_p (element-wise for diagonal eps)."""
        if isinstance(phi_p, np.ndarray):
            if phi_p.ndim == 1:
                return eps * phi_p
            else:
                # (nfaces, npol)
                return eps[:, np.newaxis] * phi_p
        elif phi_p == 0:
            return 0
        else:
            return eps * phi_p

    def __truediv__(self, exc):
        """
        Surface charges and currents for given excitation.

        MATLAB: bemret/mldivide.m

        Usage
        -----
        sig = obj \ exc

        Parameters
        ----------
        exc : CompStruct
            compstruct with fields for external excitation

        Returns
        -------
        sig : CompStruct
            compstruct with fields for surface charges and currents
        obj : BEMRet
            Updated BEM solver object

        Examples
        --------
        >>> sig, bem = bem \ exc
        """
        # MATLAB: [sig, obj] = mldivide(obj, exc)
        return self.solve(exc)

    def __mul__(self, sig):
        """
        Induced potential for given surface charge.

        MATLAB: bemret/mtimes.m

        Usage
        -----
        phi = obj * sig

        Parameters
        ----------
        sig : dict or CompStruct
            Surface charges and currents

        Returns
        -------
        phi : dict
            Combined potentials from inside and outside

        Examples
        --------
        >>> phi = bem * sig
        """
        # MATLAB: phi = potential(obj, sig, 1) + potential(obj, sig, 2)
        pot1 = self.potential(sig, 1)
        pot2 = self.potential(sig, 2)

        # Combine potentials from inside and outside
        # pot1 has phi1, phi1p, a1, a1p
        # pot2 has phi2, phi2p, a2, a2p
        # Extract enei from sig (dict or CompStruct)
        enei = sig['enei'] if isinstance(sig, dict) else sig.enei

        phi = {
            'phi1': pot1['phi1'],
            'phi1p': pot1['phi1p'],
            'a1': pot1['a1'],
            'a1p': pot1['a1p'],
            'phi2': pot2['phi2'],
            'phi2p': pot2['phi2p'],
            'a2': pot2['a2'],
            'a2p': pot2['a2p'],
            'enei': enei,
            'p': self.p
        }
        return phi

    def solve(self, exc):
        """
        Solve BEM equations for given excitation.

        Computes surface charges and currents from external fields.
        MATLAB: bemret/mldivide.m

        Parameters
        ----------
        exc : dict
            Excitation dictionary with fields:
                - enei : wavelength/energy
                - phi1, phi2, a1, a2 : potentials (optional, default 0)
                - phi1p, phi2p, a1p, a2p : potential derivatives

        Returns
        -------
        sig : dict
            Dictionary containing:
                - sig1, sig2 : surface charge distributions (inside/outside)
                - h1, h2 : surface current distributions (inside/outside)
                - enei : wavelength/energy
                - p : particle object
        obj : BEMRet
            Updated BEM solver object

        Examples
        --------
        >>> sig, bem = bem.solve(exc)

        Notes
        -----
        This implements Equations (19-20) from Garcia de Abajo & Howie (2002).
        """
        enei = exc['enei']

        # Initialize at excitation energy if needed
        self.init(enei)

        # Compute excitation variables from raw inputs
        # MATLAB: [phi, a, alpha, De] = excitation(obj, exc)
        phi, a, alpha, De = self._excitation(exc)

        # Get stored variables
        k = self.k
        nvec = self.nvec
        G1_lu = self.G1_lu
        G2_lu = self.G2_lu
        L1 = self.L1
        L2 = self.L2
        Sigma1 = self.Sigma1
        Delta_lu = self.Delta_lu
        Sigma_lu = self.Sigma_lu
        nfaces = self.p.nfaces

        # Phase 3: GPU_NATIVE — keep tensors on device end-to-end.
        # v1.7 A1 fix: ``_bem_native_gpu()`` only reads the env var; it
        # cannot detect the case where ``_init_gpu_assemble`` raised and
        # fell back to the CPU path (leaving Sigma1/L1 on host).  Cross-
        # check the actual residency of the cached state before promoting
        # the solve to device — otherwise a numpy L1 mixed with cupy phi
        # produces a TypeError mid-solve.
        native = _bem_native_gpu()
        if native and (not _CUPY_OK_A2 or not is_cupy_array(self.Sigma1)):
            native = False
        if native:
            cp = _cp_a2
            xp = cp
            nvec = cp.asarray(nvec)
        else:
            xp = np

        def _to_xp(x):
            if native and not isinstance(x, cp.ndarray):
                return cp.asarray(x)
            return x

        def _ls(lu_piv, b):
            if isinstance(lu_piv, tuple) and len(lu_piv) == 3 and lu_piv[0] in ("cpu", "gpu", "mgpu"):
                if native and lu_piv[0] != "mgpu":
                    if b.ndim == 1:
                        return lu_solve_native(lu_piv, b)
                    return lu_solve_native(lu_piv, b.reshape(b.shape[0], -1)).reshape(b.shape)
                if b.ndim == 1:
                    return lu_solve_dispatch(lu_piv, b)
                return lu_solve_dispatch(lu_piv, b.reshape(b.shape[0], -1)).reshape(b.shape)
            if b.ndim == 1:
                return lu_solve(lu_piv, b, check_finite=False)
            return lu_solve(lu_piv, b.reshape(b.shape[0], -1), check_finite=False).reshape(b.shape)

        def _ls_sigma(b):
            # Wrapper that transparently applies the v1.2.0 Schur-complement
            # reduction when active. b has shape (nfaces,) or (nfaces, npol).
            if not self._schur_active:
                return _ls(self.Sigma_lu, b)
            b_full = np.asarray(b) if not isinstance(b, np.ndarray) else b
            b_eff = self._schur_reduce_rhs(b_full)
            sig_core = _ls(self.Sigma_lu, b_eff)
            return self._schur_recover(sig_core, b_full)

        # Ensure phi, a have proper shapes
        if not isinstance(phi, np.ndarray) or phi.ndim == 0 or (isinstance(phi, np.ndarray) and phi.size == 1 and phi == 0):
            phi = xp.zeros(nfaces, dtype=complex)
        if not isinstance(a, np.ndarray) or a.ndim == 0 or (isinstance(a, np.ndarray) and a.size == 1 and a == 0):
            a = xp.zeros((nfaces, 3), dtype=complex)
        if not isinstance(alpha, np.ndarray):
            alpha = xp.zeros((nfaces, 3), dtype=complex)
        if not isinstance(De, np.ndarray):
            De = xp.zeros(nfaces, dtype=complex)

        # Promote any host arrays to GPU when native mode is active.
        if native:
            phi = _to_xp(phi)
            a = _to_xp(a)
            alpha = _to_xp(alpha)
            De = _to_xp(De)

        # Determine number of polarizations from array with most dimensions
        npol = 1
        if hasattr(a, 'ndim') and a.ndim == 3:
            npol = a.shape[2]
        elif hasattr(alpha, 'ndim') and alpha.ndim == 3:
            npol = alpha.shape[2]
        elif hasattr(phi, 'ndim') and phi.ndim == 2:
            npol = phi.shape[1]
        elif hasattr(De, 'ndim') and De.ndim == 2:
            npol = De.shape[1]

        # Squeeze arrays if npol == 1 (single polarization case)
        if npol == 1:
            if hasattr(a, 'ndim') and a.ndim == 3:
                a = a[:, :, 0]
            if hasattr(alpha, 'ndim') and alpha.ndim == 3:
                alpha = alpha[:, :, 0]
            if hasattr(phi, 'ndim') and phi.ndim == 2:
                phi = phi[:, 0]
            if hasattr(De, 'ndim') and De.ndim == 2:
                De = De[:, 0]

        # Allocate output arrays
        if npol == 1:
            sig1_all = xp.zeros(nfaces, dtype=complex)
            sig2_all = xp.zeros(nfaces, dtype=complex)
            h1_all = xp.zeros((nfaces, 3), dtype=complex)
            h2_all = xp.zeros((nfaces, 3), dtype=complex)

            # Modify alpha and De [Eqs. before (19)]
            # MATLAB: L1_phi = matmul(L1, phi)
            if np.isscalar(L1):
                L1_phi = L1 * phi
                L1_a = L1 * a
            else:
                L1_phi = L1 @ phi
                L1_a = L1 @ a

            # alpha = alpha - matmul(Sigma1, a) + 1i*k*outer(nvec, L1*phi)
            alpha_mod = alpha - (Sigma1 @ a) + 1j * k * (nvec * L1_phi[:, xp.newaxis])
            # De = De - matmul(Sigma1, matmul(L1, phi)) + 1i*k*inner(nvec, L1*a)
            if np.isscalar(L1):
                De_mod = De - Sigma1 @ (L1 * phi) + 1j * k * xp.sum(nvec * L1_a, axis=1)
            else:
                De_mod = De - Sigma1 @ L1 @ phi + 1j * k * xp.sum(nvec * L1_a, axis=1)

            # Eq. (19): surface charge
            L_diff = L1 - L2
            if np.isscalar(L_diff):
                inner_term = xp.sum(nvec * (L_diff * _ls(Delta_lu, alpha_mod)), axis=1)
            else:
                inner_term = xp.sum(nvec * (L_diff @ _ls(Delta_lu, alpha_mod)), axis=1)

            sig2 = _ls_sigma(De_mod + 1j * k * inner_term)

            # Eq. (20): surface current
            if np.isscalar(L_diff):
                outer_term = nvec * (L_diff * sig2)[:, xp.newaxis]
            else:
                outer_term = nvec * (L_diff @ sig2)[:, xp.newaxis]
            h2 = _ls(Delta_lu, 1j * k * outer_term + alpha_mod)

            # Surface charges and currents [from Eqs. (10-11)]
            sig1_all = _ls(G1_lu, sig2 + phi)
            h1_all = _ls(G1_lu, h2 + a)
            sig2_all = _ls(G2_lu, sig2)
            h2_all = _ls(G2_lu, h2)

        else:
            # Multiple polarizations - vectorized over npol axis (M4 Tier 2)
            # phi: (n, npol), a: (n, 3, npol), alpha: (n, 3, npol), De: (n, npol)
            # Broadcast nvec (n, 3) -> (n, 3, 1) when needed.
            if phi.ndim == 1:
                phi = xp.broadcast_to(phi[:, xp.newaxis], (nfaces, npol)).copy()
            if a.ndim == 2:
                a = xp.broadcast_to(a[:, :, xp.newaxis], (nfaces, 3, npol)).copy()
            if alpha.ndim == 2:
                alpha = xp.broadcast_to(alpha[:, :, xp.newaxis], (nfaces, 3, npol)).copy()
            if De.ndim == 1:
                De = xp.broadcast_to(De[:, xp.newaxis], (nfaces, npol)).copy()

            # L1 @ phi (n, npol); L1 @ a flattens over (3, npol)
            if np.isscalar(L1):
                L1_phi = L1 * phi
                L1_a = L1 * a
            else:
                L1_phi = L1 @ phi
                L1_a = (L1 @ a.reshape(nfaces, -1)).reshape(nfaces, 3, npol)

            # alpha_mod = alpha - Sigma1 @ a + ik * nvec ⊗ (L1*phi)
            Sigma1_a = (Sigma1 @ a.reshape(nfaces, -1)).reshape(nfaces, 3, npol)
            alpha_mod = alpha - Sigma1_a + 1j * k * (nvec[:, :, xp.newaxis] * L1_phi[:, xp.newaxis, :])

            # De_mod = De - Sigma1 @ (L1 phi) + ik * <nvec, L1 a>
            if np.isscalar(L1):
                Sigma1_L1_phi = Sigma1 @ (L1 * phi)
            else:
                Sigma1_L1_phi = Sigma1 @ (L1 @ phi)
            De_mod = De - Sigma1_L1_phi + 1j * k * xp.einsum('ij,ijk->ik', nvec, L1_a)

            L_diff = L1 - L2

            # Eq. (19): surface charge sig2
            # alpha_mod_solved: (n, 3, npol)
            am_solved = _ls(Delta_lu, alpha_mod)
            if np.isscalar(L_diff):
                Ld_am = L_diff * am_solved
            else:
                Ld_am = (L_diff @ am_solved.reshape(nfaces, -1)).reshape(nfaces, 3, npol)
            inner_term = xp.einsum('ij,ijk->ik', nvec, Ld_am)
            sig2 = _ls_sigma(De_mod + 1j * k * inner_term)

            # Eq. (20): surface current h2
            if np.isscalar(L_diff):
                Ld_sig2 = L_diff * sig2
            else:
                Ld_sig2 = L_diff @ sig2
            outer_term = nvec[:, :, xp.newaxis] * Ld_sig2[:, xp.newaxis, :]
            h2 = _ls(Delta_lu, 1j * k * outer_term + alpha_mod)

            # Surface charges and currents [from Eqs. (10-11)]
            sig1_all = _ls(G1_lu, sig2 + phi)
            h1_all = _ls(G1_lu, h2 + a)
            sig2_all = _ls(G2_lu, sig2)
            h2_all = _ls(G2_lu, h2)

        # MATLAB: sig = compstruct(obj.p, exc.enei, 'sig1', sig1, 'sig2', sig2, 'h1', h1, 'h2', h2)
        # v1.7 Phase 1.4 fix: host-materialize cupy results before returning
        # so user-facing code that calls ``np.asarray(sig.sig1)`` etc. does
        # not trip on cupy's implicit-conversion guard.  Downstream
        # excitation runners (PlaneWave/Dipole/EELS) already host-promote
        # internally, so this is a defensive belt-and-braces guarantee.
        if is_cupy_array(sig1_all):
            sig1_all = to_host(sig1_all)
        if is_cupy_array(sig2_all):
            sig2_all = to_host(sig2_all)
        if is_cupy_array(h1_all):
            h1_all = to_host(h1_all)
        if is_cupy_array(h2_all):
            h2_all = to_host(h2_all)

        from ..greenfun import CompStruct
        sig = CompStruct(self.p, enei, sig1=sig1_all, sig2=sig2_all, h1=h1_all, h2=h2_all)

        # MATLAB: [sig, obj] = mldivide(obj, exc)
        return sig, self

    def potential(self, sig, inout=2):
        """
        Compute potentials and surface derivatives inside/outside of particle.

        MATLAB: bemret/potential.m -> compgreenret/potential.m

        Delegates to CompGreenRet.potential() which properly evaluates
        Green functions using region-based indexing.

        Parameters
        ----------
        sig : CompStruct
            Solution containing sig1, sig2, h1, h2, enei, p
        inout : int
            1 for inside, 2 for outside (default)

        Returns
        -------
        pot : CompStruct
            Potentials: phi1/phi2, phi1p/phi2p, a1/a2, a1p/a2p
        """
        enei = sig.enei
        self.init(enei)

        # Delegate to CompGreenRet.potential() which properly evaluates
        # Green functions using region-based indexing
        return self.g.potential(sig, inout)

    def field(self, sig, inout=2):
        """
        Compute electric and magnetic fields inside/outside of particle.

        MATLAB: bemret/field.m -> compgreenret/field.m

        Delegates to CompGreenRet.field() which uses Cartesian derivative
        Green functions (Gp, H1p, H2p) for proper field computation.

        Parameters
        ----------
        sig : CompStruct
            Solution containing sig1, sig2, h1, h2, enei, p
        inout : int
            1 for inside, 2 for outside (default)

        Returns
        -------
        field : CompStruct
            Fields: e (electric), h (magnetic), enei, p

        Notes
        -----
        MATLAB formula:
            e = i*k*(G1*h1 + G2*h2) - H1p*sig1 - H2p*sig2
            h = cross(H1p, h1) + cross(H2p, h2)

        where H1p, H2p are Cartesian derivatives of Green function.
        """
        enei = sig.enei
        self.init(enei)

        # Delegate to CompGreenRet.field() which properly uses
        # Cartesian derivative Green functions (H1p, H2p)
        return self.g.field(sig, inout)

    def clear(self):
        """
        Clear Green functions and auxiliary matrices.

        MATLAB: bemret/clear.m

        Returns
        -------
        self : BEMRet
            Returns self for chaining

        Examples
        --------
        >>> bem = bem.clear()
        """
        # MATLAB: [obj.G1i, obj.G2i, obj.L1, obj.L2, obj.Sigma1, obj.Deltai, obj.Sigmai] = deal([])
        # Release distributed multi-GPU LU buffers when present
        # (mgpu handles carry a ``_distmat_keepalive`` DistributedMatrix
        # whose per-GPU tiles must be ``free()``'d explicitly — close()
        # on the handle alone only releases IPIV/workspace/cusolverMg
        # objects).
        for _attr in ('G1_lu', 'G2_lu', 'Delta_lu', 'Sigma_lu'):
            old = getattr(self, _attr, None)
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
            setattr(self, _attr, None)
        self.L1 = None
        self.L2 = None
        self.Sigma1 = None
        # Force the next init() call to rebuild the matrices.  Without
        # this, ``init()``'s "skip if same enei" guard would silently
        # leave the solver in a half-cleared state where L1/Sigma1 are
        # None but ``self.enei`` is still set, causing later solve()
        # calls to crash with cryptic ``NoneType`` errors.
        self.enei = None
        return self

    def __call__(self, enei):
        """
        Computes resolvent matrices for later use in mldivide.

        MATLAB: bemret/subsref.m case '()'

        Parameters
        ----------
        enei : float
            Light wavelength in vacuum

        Returns
        -------
        self : BEMRet
            Returns self for chaining

        Examples
        --------
        >>> bem = bem(600.0)
        """
        return self.init(enei)

    def __repr__(self):
        status = "λ={:.1f}nm".format(self.enei) if self.enei is not None else "not initialized"
        return "BEMRet(p: {} faces, {})".format(self.p.nfaces, status)

    def __str__(self):
        status = "Initialized at λ={:.2f} nm".format(self.enei) if self.enei is not None else "Not initialized"
        mat_info = "  Sigma_lu matrix: {}".format(self.Sigma_lu[0].shape) if self.Sigma_lu is not None else "  Sigma_lu matrix: Not computed"

        return (
            "BEM Solver (Retarded/Full Maxwell):\n"
            "  Particle: {} faces\n"
            "  Status: {}\n"
            "{}\n"
            "  Wavenumber k: {:.6f}".format(
                self.p.nfaces, status, mat_info, self.k) if self.k is not None else "  Wavenumber k: Not computed"
        )
