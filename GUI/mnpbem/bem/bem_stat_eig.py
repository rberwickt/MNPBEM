"""
BEM solver for quasistatic approximation with eigenmode expansion.

MATLAB: BEM/@bemstateig/
Given an external excitation, BEMStatEig computes the surface charges
such that the boundary conditions of Maxwell's equations in the
quasistatic approximation (using eigenmode expansion) are fulfilled.

Reference:
    Garcia de Abajo and Howie, PRB 65, 115418 (2002)
    Hohenester et al., PRL 103, 106801 (2009)
"""

import os
import numpy as np
from typing import Optional, Tuple, Any

from ..greenfun import CompGreenStat, CompStruct
from ..utils.gpu import (
    matmul_dispatch, solve_dispatch, to_host, is_cupy_array,
    eig_dispatch,
)
from .plasmonmode import plasmonmode


# v1.7.4: ``MNPBEM_GPU_EIG=1`` opt-in routes BEMStatEig's quasistatic
# eigenmode decomposition through ``eig_dispatch`` (single-GPU
# ``cupy.linalg.eig`` for n >= GPU_THRESHOLD).  When disabled (default)
# the path matches v1.7.3 exactly — ``plasmonmode()`` runs on the host
# via scipy.linalg.eig / scipy.sparse.eigs.  This keeps the regression
# surface zero while exposing the GPU win for production sweeps.
#
# v1.7.3 Phase 2: BEMStatEig's quasistatic eigenmode pipeline is dominated
# by scipy.sparse.linalg.eigs / scipy.linalg.eig (CPU only — no cuSolverMg
# eigh equivalent for non-Hermitian F).  The surrounding GEMMs / dense
# solves (ur @ inv(resolvent) @ ul) still go through matmul_dispatch /
# solve_dispatch and pick up MNPBEM_VRAM_SHARE_* automatically.  The
# remaining wins are pool drains after plasmonmode() and the per-
# wavelength resolvent solve, to keep the cupy pool from accumulating
# across long sweeps.
USE_GPU_EIG: bool = os.environ.get("MNPBEM_GPU_EIG", "0").strip() == "1"
try:
    import cupy as _cp_eig  # type: ignore
    _CUPY_OK_EIG = True
except Exception:
    _cp_eig = None  # type: ignore
    _CUPY_OK_EIG = False


def _gpu_eig_available() -> bool:
    """Return True iff cupy.linalg.eig (geev) is callable on this build.

    cupy 14 + CUDA 12.x ship without ``geev`` in cuSOLVER, so calling
    ``cupy.linalg.eig`` raises ``RuntimeError("geev is not available")``.
    Detecting that up-front avoids paying the import + tiny-matrix
    overhead inside ``_plasmonmode_gpu`` only to fall back to LAPACK.
    """
    if not _CUPY_OK_EIG:
        return False
    try:
        tiny = _cp_eig.asarray(np.eye(4, dtype=np.complex128))
        _cp_eig.linalg.eig(tiny)
        return True
    except Exception:
        return False


_GPU_EIG_AVAIL: Optional[bool] = None


def _gpu_eig_available_cached() -> bool:
    global _GPU_EIG_AVAIL
    if _GPU_EIG_AVAIL is None:
        _GPU_EIG_AVAIL = _gpu_eig_available()
    return _GPU_EIG_AVAIL


def _plasmonmode_gpu(p, F, nev, options):
    """GPU-accelerated plasmonmode replacement (MNPBEM_GPU_EIG=1 path).

    Path selection
    --------------
    - If cupy.linalg.eig (geev) is unavailable in the current cupy build
      (e.g. cupy 14 + CUDA 12.x), the function returns the *same* result
      as ``plasmonmode()`` by calling it directly — no slowdown, no
      regression.
    - When geev becomes available, the right-eigenpair is computed on
      GPU (``cupy.linalg.eig``), the left-eigenpair is computed on host
      (``scipy.linalg.eig(F.T)``), eigenvalues are paired by nearest
      neighbour, and the result is bi-orthogonalised.

    The wrapper intentionally never picks a strategy that is slower than
    ``plasmonmode()``: for n >= 2000 the host code path uses
    ``scipy.sparse.linalg.eigs`` (ARPACK, much faster than dense LAPACK
    when ``nev << n``), and our GPU path can only beat that once cupy
    ships a partial-eig backend.  Until then, falling through to
    ``plasmonmode()`` is the correct call.

    Parameters
    ----------
    p : ComParticle (unused; F encodes the geometry).
    F : (n, n) ndarray — surface derivative of the quasistatic Green
        function.  May be cupy or numpy; brought to host.
    nev : int — requested number of eigenmodes.
    options : dict — passed through if we fall back to plasmonmode().

    Returns
    -------
    ene, ur, ul : same shape/semantics as plasmonmode().
    """
    if is_cupy_array(F):
        F = to_host(F)
    n = F.shape[0]
    nev_actual = min(nev, n - 1) if n > 1 else 1

    # Fast path: if GPU eig is not usable, just call the CPU
    # plasmonmode.  This matches v1.7.3 behaviour exactly so a user who
    # flips MNPBEM_GPU_EIG=1 on a system without working geev never sees
    # a perf regression.
    if not _gpu_eig_available_cached():
        return plasmonmode(p, nev=nev, **options)

    # GPU path: right eigenpair on device, left eigenpair on host.
    out_r = eig_dispatch(F, k=None, left=False, right=True)
    ene_all = out_r[0]
    vr_all = out_r[1]

    from scipy.linalg import eig as _scipy_eig
    ene_l, vl_all = _scipy_eig(F.T, left=False, right=True,
            check_finite=False)

    # Match left eigenvalues to right eigenvalues by nearest-neighbour.
    # eigenvalues of F and F.T are mathematically identical; numerical
    # ordering may differ across solvers, so we re-pair explicitly.
    idx_sort = np.argsort(ene_all.real)[:nev_actual]
    ene_diag = ene_all[idx_sort]
    ur = vr_all[:, idx_sort]

    ene_l_idx = []
    used = set()
    for w in ene_diag:
        d = np.abs(ene_l - w)
        order = np.argsort(d)
        for j in order:
            ji = int(j)
            if ji not in used:
                ene_l_idx.append(ji)
                used.add(ji)
                break
    ul = vl_all[:, np.asarray(ene_l_idx, dtype=int)].conj().T

    # Sort by ascending real part (final ordering)
    sort_idx = np.argsort(ene_diag.real)
    ene = ene_diag[sort_idx]
    ur = ur[:, sort_idx]
    ul = ul[sort_idx, :]

    # Bi-orthogonalisation: ul = (ul @ ur)^{-1} @ ul
    overlap = ul @ ur
    ul = np.linalg.solve(overlap, ul)
    return ene, ur, ul


def _gpu_pool_cleanup_eig(apply_limit: bool = False) -> None:
    """Synchronise CUDA stream then drain cupy default + pinned pools."""
    if not _CUPY_OK_EIG:
        return
    try:
        mempool = _cp_eig.get_default_memory_pool()
        pinned = _cp_eig.get_default_pinned_memory_pool()
        if apply_limit:
            try:
                pool_limit_gb = float(os.environ.get(
                        'MNPBEM_GPU_POOL_LIMIT_GB', '0'))
            except (TypeError, ValueError):
                pool_limit_gb = 0.0
            if pool_limit_gb > 0:
                mempool.set_limit(size = int(pool_limit_gb * (1024 ** 3)))
        _cp_eig.cuda.runtime.deviceSynchronize()
        mempool.free_all_blocks()
        pinned.free_all_blocks()
    except Exception:
        pass


class BEMStatEig(object):
    """BEM solver for quasistatic approximation and eigenmode expansion.

    Given an external excitation, BEMStatEig computes the surface
    charges such that the boundary conditions of Maxwell's equations in
    the quasistatic approximation (using eigenmode expansion) are fulfilled.

    MATLAB: @bemstateig

    Parameters
    ----------
    p : ComParticle
        Composite particle (see comparticle)
    nev : int
        Number of eigenmodes to compute.  Defaults to 20.
    enei : float, optional
        Light wavelength in vacuum for pre-initialization

    Properties
    ----------
    name : str
        'bemsolver' (constant)
    needs : dict
        {'sim': 'stat', 'nev': True} (constant)
    p : ComParticle
        Composite particle
    nev : int
        Number of eigenmodes
    ene : ndarray, shape (nev,)
        Plasmon eigenenergies
    ur : ndarray, shape (n, nev)
        Right eigenvectors (surface charge patterns)
    ul : ndarray, shape (nev, n)
        Left eigenvectors
    unit : ndarray, shape (nev^2, np)
        Unit matrices for eigenmode expansion
    enei : float or None
        Light wavelength in vacuum
    mat : ndarray or None
        Resolvent matrix
    g : CompGreenStat
        Green function

    Methods
    -------
    __init__(p, nev=20, enei=None, **options)
        Initialize quasistatic BEM solver with eigenmode expansion
    solve(exc)
        Solve BEM equations for given excitation
    __truediv__(exc)
        Surface charge for given excitation (operator \\)
    __mul__(sig)
        Induced potential for given surface charge (operator *)
    field(sig, inout=2)
        Electric field inside/outside of particle surface
    potential(sig, inout=2)
        Potentials and surface derivatives inside/outside of particle
    __call__(enei)
        Computes resolvent matrix for later use in solve

    Examples
    --------
    >>> from mnpbem import EpsConst, EpsTable, trisphere, ComParticle
    >>> from mnpbem.bem import BEMStatEig
    >>>
    >>> # Create gold sphere
    >>> eps_tab = [EpsConst(1.0), EpsTable('gold.dat')]
    >>> sphere = trisphere(144, 10.0)
    >>> p = ComParticle(eps_tab, [sphere], [[2, 1]])
    >>>
    >>> # Create BEM solver
    >>> bem = BEMStatEig(p, nev=20)
    >>>
    >>> # Solve for excitation
    >>> sig, bem = bem.solve(exc)
    """

    name = 'bemsolver'
    needs = {'sim': 'stat', 'nev': True}

    def __init__(self,
            p,
            nev = 20,
            enei = None,
            **options):
        """Initialize quasistatic BEM solver with eigenmode expansion.

        MATLAB: bemstateig.m

        Parameters
        ----------
        p : ComParticle
            Compound of particles (see comparticle)
        nev : int
            Number of eigenmodes to compute.  Defaults to 20.
        enei : float, optional
            Light wavelength in vacuum
        **options : dict
            Additional options passed to CompGreenStat
        """
        self.p = p
        self.nev = nev
        self.enei = None  # type: Optional[float]

        # resolvent matrix
        self.mat = None  # type: Optional[np.ndarray]

        # Green function
        self.g = CompGreenStat(p, p, **options)

        # surface derivative of Green function
        F = self.g.F  # (n, n)
        # v1.7.3 Phase 2: F may live on cupy when MNPBEM_GPU=1.  Bring it to
        # host so subsequent host-only ops (np.diag, eigs) work.  Also release
        # the GPU view so its N^2 buffer returns to the pool before
        # plasmonmode() runs its own dense eigensolve.
        if is_cupy_array(F):
            F = to_host(F)
        if _CUPY_OK_EIG:
            _gpu_pool_cleanup_eig()

        # eigenmode expansion using plasmonmode (or GPU dispatch when opted-in)
        if USE_GPU_EIG and _CUPY_OK_EIG:
            ene, ur, ul = _plasmonmode_gpu(p, F, nev, options)
        else:
            ene, ur, ul = plasmonmode(p, nev = nev, **options)
        # v1.7.3 Phase 2: plasmonmode internally builds (and discards) a dense
        # F + may route through cupy for the eigendecomposition staging.
        # Drain the pool so the constructor leaves the device in a clean state.
        if _CUPY_OK_EIG:
            _gpu_pool_cleanup_eig()

        # actual number of eigenmodes (may be less than requested)
        self.nev = len(ene)

        self.ene = np.diag(ene)  # (nev, nev) diagonal matrix
        self.ur = ur  # (n, nev)
        self.ul = ul  # (nev, n)

        # unit matrices for eigenmode expansion
        # MATLAB: unit(:, ip) = reshape(ul(:, ind) * ur(ind, :), nev^2, 1)
        self.unit = np.zeros((self.nev ** 2, p.np), dtype = complex)
        for ip in range(p.np):
            ind = p.index_func(ip + 1)
            chunk = self.ul[:, ind] @ self.ur[ind, :]  # (nev, nev)
            self.unit[:, ip] = chunk.ravel()

        if enei is not None:
            self._init_matrices(enei)

    def _init_matrices(self, enei):
        """Initialize resolvent matrix for BEM solver.

        MATLAB: @bemstateig/subsref.m case '()'

        Parameters
        ----------
        enei : float
            Light wavelength in vacuum

        Returns
        -------
        self : BEMStatEig
            Returns self for chaining
        """
        if self.enei is not None and np.isclose(self.enei, enei):
            return self

        # v1.7.3 Phase 2: drop the previous wavelength's resolvent ``mat``
        # before the new GEMM allocates its N×nev intermediate, so the cupy
        # pool can recycle the prior N×N buffer.  Pattern mirrors BEMStat.
        self.mat = None
        _gpu_pool_cleanup_eig(apply_limit = True)

        # dielectric functions per boundary pair
        eps_vals = [eps_func(enei)[0] for eps_func in self.p.eps]

        eps1_arr = np.array([eps_vals[int(self.p.inout[j, 0]) - 1]
                             for j in range(self.p.inout.shape[0])])
        eps2_arr = np.array([eps_vals[int(self.p.inout[j, 1]) - 1]
                             for j in range(self.p.inout.shape[0])])

        # Lambda [Garcia de Abajo, Eq. (23)]
        Lambda = 2 * np.pi * (eps1_arr + eps2_arr) / (eps1_arr - eps2_arr)

        # BEM resolvent matrix from eigenmodes
        # unit @ Lambda gives (nev^2,) vector, reshaped to (nev, nev)
        unit_lambda = self.unit @ Lambda[:]  # (nev^2,)
        unit_lambda_mat = unit_lambda.reshape(self.nev, self.nev)
        resolvent = unit_lambda_mat + self.ene  # (nev, nev)

        # mat = -ur @ inv(resolvent) @ ul
        # resolvent is (nev, nev) — small, so solve is CPU-bound; the leading
        # ur @ (...) GEMM dominates at large mesh and benefits from GPU.
        # solve_dispatch + matmul_dispatch already honour MNPBEM_VRAM_SHARE_*
        # via their env-var auto-wiring; no explicit kwargs needed here.
        inv_ul = solve_dispatch(resolvent, self.ul)
        self.mat = -matmul_dispatch(self.ur, inv_ul)
        # v1.7.3 Phase 2: drop the inv_ul intermediate (~ nev × N) so cupy
        # reclaims its buffer before the next wavelength enters.
        del inv_ul

        self.enei = enei
        _gpu_pool_cleanup_eig()
        return self

    def solve(self, exc):
        """Solve BEM equations for given excitation.

        MATLAB: @bemstateig/solve.m

        Parameters
        ----------
        exc : CompStruct
            compstruct with field 'phip' for external excitation

        Returns
        -------
        sig : CompStruct
            compstruct with field for surface charge
        obj : BEMStatEig
            Updated solver
        """
        return self.__truediv__(exc)

    def __truediv__(self, exc):
        """Surface charge for given excitation.

        MATLAB: @bemstateig/mldivide.m

        Parameters
        ----------
        exc : CompStruct
            compstruct with field 'phip' for external excitation

        Returns
        -------
        sig : CompStruct
            compstruct with field for surface charge
        obj : BEMStatEig
            Updated solver
        """
        self._init_matrices(exc.enei)

        sig_result = _matmul(self.mat, exc.phip)
        sig = CompStruct(self.p, exc.enei, sig = sig_result)

        return sig, self

    def __mul__(self, sig):
        """Induced potential for given surface charge.

        MATLAB: @bemstateig/mtimes.m

        Parameters
        ----------
        sig : CompStruct
            compstruct with fields for surface charge

        Returns
        -------
        phi : CompStruct
            compstruct with fields for induced potential
        """
        pot1 = self.potential(sig, 1)
        pot2 = self.potential(sig, 2)

        phi = CompStruct(self.p, sig.enei,
                phi1 = pot1.phi1, phi1p = pot1.phi1p,
                phi2 = pot2.phi2, phi2p = pot2.phi2p)
        return phi

    def potential(self, sig, inout = 2):
        """Potentials and surface derivatives inside/outside of particle.

        MATLAB: @bemstateig/potential.m

        Parameters
        ----------
        sig : CompStruct
            compstruct with surface charges
        inout : int, optional
            Potential inside (inout=1) or outside (inout=2, default)

        Returns
        -------
        pot : CompStruct
            compstruct object with potentials
        """
        return self.g.potential(sig, inout)

    def field(self, sig, inout = 2):
        """Electric field inside/outside of particle surface.

        MATLAB: @bemstateig/field.m

        Parameters
        ----------
        sig : CompStruct
            COMPSTRUCT object with surface charges
        inout : int, optional
            Electric field inside (inout=1) or outside (inout=2, default)

        Returns
        -------
        field : CompStruct
            COMPSTRUCT object with electric field
        """
        return self.g.field(sig, inout)

    def __call__(self, enei):
        """Computes resolvent matrix for later use in solve.

        MATLAB: @bemstateig/subsref.m case '()'

        Parameters
        ----------
        enei : float
            Light wavelength in vacuum

        Returns
        -------
        self : BEMStatEig
            Returns self for chaining
        """
        return self._init_matrices(enei)

    def clear(self):
        """Clear auxiliary resolvent matrix.

        v1.7 A3: added for API parity with BEMStat / BEMStatLayer /
        BEMStatIter so calling code can drop the cached dense ``mat``
        and force a rebuild at the next solve.  Also resets ``enei``
        so the cache gate does not skip the rebuild.

        Returns
        -------
        self : BEMStatEig
            Returns self for chaining.
        """
        self.mat = None
        self.enei = None
        # v1.7.3 Phase 2: explicit clear() drains the cupy pool so the device
        # buffer of the released ``mat`` returns immediately, not on next
        # rebuild.  Mirrors BEMStat.clear pattern.
        if _CUPY_OK_EIG:
            _gpu_pool_cleanup_eig()
        return self

    def __repr__(self):
        """String representation."""
        status = 'enei={}'.format(self.enei) if self.enei is not None else 'not initialized'
        return 'BEMStatEig(p={}, nev={}, {})'.format(self.p, self.nev, status)


def _matmul(a, x):
    """Generalized matrix multiplication for tensors.

    MATLAB: Misc/matmul.m

    Handles scalar, 1D, 2D, and higher-dimensional inputs.
    For a 2D matrix a and a multi-dimensional x, the multiplication
    is performed along the first axis of x.
    """
    if np.isscalar(a) or (isinstance(a, np.ndarray) and a.size == 1):
        if a == 0:
            return 0
        return a * x
    if np.isscalar(x) or (isinstance(x, np.ndarray) and x.size == 1):
        if x == 0:
            return 0
        return a * x

    siza = a.shape
    sizx = x.shape if hasattr(x, 'shape') else (len(x),)

    if len(siza) == 3:
        # a is (n1, 3, n2), x is (n2,) or (n2, ...)
        n1, _, n2 = siza
        if len(sizx) == 1:
            return np.tensordot(a, x, axes = ([2], [0]))
        else:
            a_flat = a.reshape(n1 * 3, n2)
            x_flat = x.reshape(n2, -1)
            y_flat = a_flat @ x_flat
            new_shape = (n1, 3) + sizx[1:]
            return y_flat.reshape(new_shape)
    else:
        # Standard 2D matrix multiplication
        if len(sizx) == 1:
            return a @ x
        else:
            x_flat = x.reshape(sizx[0], -1)
            y_flat = a @ x_flat
            return y_flat.reshape((siza[0],) + sizx[1:])
