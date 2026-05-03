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

        self.enei = enei
        self.nvec = self.p.nvec
        self.k = 2 * np.pi / enei

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

        G11 = _to_dev(self.g.eval(0, 0, 'G', enei))
        G21 = _to_dev(self.g.eval(1, 0, 'G', enei))
        G22 = _to_dev(self.g.eval(1, 1, 'G', enei))
        G12 = _to_dev(self.g.eval(0, 1, 'G', enei))

        G1 = G11 - G21 if G21 is not None else G11
        G2 = G22 - G12 if G12 is not None else G22

        H11 = _to_dev(self.g.eval(0, 0, 'H1', enei))
        H21 = _to_dev(self.g.eval(1, 0, 'H1', enei))
        H22 = _to_dev(self.g.eval(1, 1, 'H2', enei))
        H12 = _to_dev(self.g.eval(0, 1, 'H2', enei))

        H1_mat = H11 - H21 if H21 is not None else H11
        H2_mat = H22 - H12 if H12 is not None else H22

        # Optional user-supplied refun (coverlayer.refine).  refun is a host
        # numpy callable; round-trip through host and re-upload — the
        # refinement touches at most a handful of pair elements so the
        # transfer cost is negligible compared with the N^3 GEMMs that
        # follow.  Applied BEFORE LU factor so factored matrices reflect
        # the refined operators.
        if self._refun is not None:
            G1_h = cp.asnumpy(G1)
            H1_h = cp.asnumpy(H1_mat)
            G2_h = cp.asnumpy(G2)
            H2_h = cp.asnumpy(H2_mat)
            G1_h, H1_h = self._refun(self.g, G1_h, H1_h)
            G2_h, H2_h = self._refun(self.g, G2_h, H2_h)
            G1 = cp.asarray(G1_h)
            H1_mat = cp.asarray(H1_h)
            G2 = cp.asarray(G2_h)
            H2_mat = cp.asarray(H2_h)

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

        n = G1_dev.shape[0]
        I_dev = cp.eye(n)

        # L matrices
        # ACA wrappers proxy the underlying CompGreenRet via .g
        _gobj = self.g.g if hasattr(self.g, 'g') and hasattr(self.g.g, 'con') else self.g
        native = _bem_native_gpu()
        if np.all(_gobj.con[0][1] == 0) or scalar_eps:
            self.L1 = self.eps1  # host scalar
            self.L2 = self.eps2
            L1_is_scalar = True
            # Avoid computing the full G1i / G2i inverses when L is scalar:
            # Sigma1 = H1 @ G1^-1 can be obtained directly via lu_solve on
            # the right (solve G1^T x^T = H1^T).  This trims two N^3 GEMMs
            # per wavelength.
            G1i_dev = None
            G2i_dev = None
        else:
            G1i_dev = _cp_lu_solve((lu1, piv1), I_dev)
            G2i_dev = _cp_lu_solve((lu2, piv2), I_dev)
            eps1_dev = cp.asarray(self.eps1)
            eps2_dev = cp.asarray(self.eps2)
            L1_dev_full = G1_dev @ eps1_dev @ G1i_dev
            L2_dev_full = G2_dev @ eps2_dev @ G2i_dev
            if native:
                # Phase 3: keep L1/L2 on device.
                self.L1 = L1_dev_full
                self.L2 = L2_dev_full
            else:
                self.L1 = cp.asnumpy(L1_dev_full)
                self.L2 = cp.asnumpy(L2_dev_full)
            L1_is_scalar = False

        # Sigma matrices: Sigma_i = H_i @ G_i^-1.
        # Use right-hand-side lu_solve when possible to skip the full
        # inverse construction.
        if G1i_dev is not None:
            Sigma1_dev = H1_mat @ G1i_dev
            Sigma2_dev = H2_mat @ G2i_dev
        else:
            # Sigma1 @ G1 = H1  =>  G1^T @ Sigma1^T = H1^T
            Sigma1_dev = _cp_lu_solve((lu1, piv1), H1_mat.T, trans=1).T
            Sigma2_dev = _cp_lu_solve((lu2, piv2), H2_mat.T, trans=1).T
        # Phase 3: keep Sigma1 on device when GPU_NATIVE is enabled so the
        # downstream solve() can do Sigma1 @ a entirely on the GPU.
        if native:
            self.Sigma1 = Sigma1_dev
        else:
            self.Sigma1 = cp.asnumpy(Sigma1_dev)

        # Delta = Sigma1 - Sigma2  (still on device)
        Delta_dev = Sigma1_dev - Sigma2_dev
        Dc = Delta_dev.copy()
        lu_d, piv_d = _cp_lu_factor(Dc, overwrite_a=True)
        self.Delta_lu = ('gpu', lu_d, piv_d)
        Deltai_dev = _cp_lu_solve((lu_d, piv_d), I_dev)

        # nvec*nvec^T on device.
        nvec_dev = cp.asarray(self.nvec)
        nvec_outer_dev = nvec_dev @ nvec_dev.T

        if L1_is_scalar:
            Sigma_dev = (Sigma1_dev * self.L1) - (Sigma2_dev * self.L2)
            L_scalar = self.L1 - self.L2
            Sigma_dev = Sigma_dev + (self.k ** 2) * L_scalar * (
                Deltai_dev * nvec_outer_dev) * L_scalar
        else:
            L1_dev = cp.asarray(self.L1)
            L2_dev = cp.asarray(self.L2)
            L_dev = L1_dev - L2_dev
            Sigma_dev = (
                Sigma1_dev @ L1_dev - Sigma2_dev @ L2_dev +
                (self.k ** 2) * ((L_dev @ Deltai_dev) * nvec_outer_dev) @ L_dev
            )

        # Final Sigma LU factor on device.
        Sc = Sigma_dev.copy()
        lu_s, piv_s = _cp_lu_factor(Sc, overwrite_a=True)
        self.Sigma_lu = ('gpu', lu_s, piv_s)
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
        native = _bem_native_gpu()
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
        self.G1_lu = None
        self.G2_lu = None
        self.L1 = None
        self.L2 = None
        self.Sigma1 = None
        self.Delta_lu = None
        self.Sigma_lu = None
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
