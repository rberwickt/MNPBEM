import os
import numpy as np
from scipy.sparse.linalg import eigs
from typing import Optional, List, Tuple, Any, Union

from ..greenfun import CompStruct
from ..greenfun.compgreen_stat_mirror import CompGreenStatMirror
from ..geometry.comparticle_mirror import CompStructMirror
from ..utils.gpu import (
    matmul_dispatch, solve_dispatch, to_host, is_cupy_array,
    eig_dispatch,
)
from .bem_stat_mirror import _mirror_stat_eval_host


# v1.7.4: ``MNPBEM_GPU_EIG=1`` opt-in routes the per-symmetry eigen-
# decomposition through ``eig_dispatch`` (single-GPU ``cupy.linalg.eig``
# for n>=GPU_THRESHOLD).  When disabled (default), the path matches
# v1.7.3 (scipy.sparse.linalg.eigs).  cuSolverMg has no non-Hermitian
# multi-GPU eig, so the Hermitian-only Multi-GPU path does not apply.
#
# v1.7.3 Phase 2: BEMStatEigMirror's eigs loop is CPU-only (scipy.sparse
# eigs has no GPU eigh equivalent for non-Hermitian operators), but the
# per-symmetry resolvent solve + ur @ inv(...) GEMM pipeline still picks
# up MNPBEM_VRAM_SHARE_* via solve_dispatch / matmul_dispatch's env-var
# auto-wiring.  Pool drains after the eigs loop and per-symmetry block
# keep the cupy pool from accumulating across wavelength sweeps.
USE_GPU_EIG_MIR: bool = os.environ.get("MNPBEM_GPU_EIG", "0").strip() == "1"


_GPU_EIG_AVAIL_MIR: Optional[bool] = None


def _gpu_eig_available_cached_mir() -> bool:
    """Return True iff cupy.linalg.eig (geev) is callable on this build.

    Cached one-time probe; same pattern as bem_stat_eig._gpu_eig_available
    but local to the mirror module so neither calls into the other.
    """
    global _GPU_EIG_AVAIL_MIR
    if _GPU_EIG_AVAIL_MIR is not None:
        return _GPU_EIG_AVAIL_MIR
    if not _CUPY_OK_EIGMIR:
        _GPU_EIG_AVAIL_MIR = False
        return False
    try:
        tiny = _cp_eigmir.asarray(np.eye(4, dtype=np.complex128))
        _cp_eigmir.linalg.eig(tiny)
        _GPU_EIG_AVAIL_MIR = True
    except Exception:
        _GPU_EIG_AVAIL_MIR = False
    return _GPU_EIG_AVAIL_MIR
try:
    import cupy as _cp_eigmir  # type: ignore
    _CUPY_OK_EIGMIR = True
except Exception:
    _cp_eigmir = None  # type: ignore
    _CUPY_OK_EIGMIR = False


def _gpu_pool_cleanup_eigmir(apply_limit: bool = False) -> None:
    """Synchronise CUDA stream then drain cupy default + pinned pools."""
    if not _CUPY_OK_EIGMIR:
        return
    try:
        mempool = _cp_eigmir.get_default_memory_pool()
        pinned = _cp_eigmir.get_default_pinned_memory_pool()
        if apply_limit:
            try:
                pool_limit_gb = float(os.environ.get(
                        'MNPBEM_GPU_POOL_LIMIT_GB', '0'))
            except (TypeError, ValueError):
                pool_limit_gb = 0.0
            if pool_limit_gb > 0:
                mempool.set_limit(size = int(pool_limit_gb * (1024 ** 3)))
        _cp_eigmir.cuda.runtime.deviceSynchronize()
        mempool.free_all_blocks()
        pinned.free_all_blocks()
    except Exception:
        pass


class BEMStatEigMirror(object):
    """BEM solver for quasistatic approximation and eigenmode expansion
    using mirror symmetry.

    Given an external excitation, BEMStatEigMirror computes the surface
    charges such that the boundary conditions of Maxwell's equations in
    the quasistatic approximation (using eigenmode expansion) are fulfilled.

    MATLAB: @bemstateigmirror

    Parameters
    ----------
    p : ComParticleMirror
        Composite particle with mirror symmetry
    nev : int
        Number of eigenmodes
    enei : float, optional
        Light wavelength in vacuum for pre-initialization
    """

    name = 'bemsolver'
    needs = {'sim': 'stat', 'nev': True, 'sym': True}

    def __init__(self,
            p: Any,
            nev: int = 20,
            enei: Optional[float] = None,
            **options: Any) -> None:
        self.p = p
        self.nev = nev
        self.enei = None  # type: Optional[float]

        # eigenmodes (one set per symmetry value)
        self.ur = []  # type: List[np.ndarray]
        self.ul = []  # type: List[np.ndarray]
        self.ene = []  # type: List[np.ndarray]
        self.unit = []  # type: List[np.ndarray]

        # resolvent matrices
        self.mat = None  # type: Optional[List]

        # Green function
        self.g = CompGreenStatMirror(p, p, **options)

        # surface derivative of Green function (list, one per symmetry value).
        # Host-promoting wrapper to keep MNPBEM_GPU=1 from producing zero lists.
        F_list = _mirror_stat_eval_host(self.g, 'F')

        # Mirror half-particle face index ranges.  The mirror solver lives on
        # the contracted half mesh (size = nfaces_half); ``p.np`` and
        # ``p.index_func`` are inherited from the *full* particle and therefore
        # return indices > nfaces_half, which is out of range for the
        # eigenvector slicing below.  Build the half-particle ranges directly
        # from ``self.p.p`` (mirror half particle list).
        half_indices = []
        offset = 0
        for part in p.p:
            half_indices.append(np.arange(offset, offset + part.nfaces, dtype = int))
            offset += part.nfaces
        n_half_particles = len(half_indices)

        # eigenmode expansion
        for i in range(len(F_list)):
            F_i = F_list[i]
            # Host promotion guard (mirror solver sometimes ships cupy F_i).
            if is_cupy_array(F_i):
                F_i = to_host(F_i)

            # left and right eigenvectors.
            # v1.7.4: with MNPBEM_GPU_EIG=1 AND a working cupy.linalg.eig
            # (geev) in the current cupy build, the per-symmetry right
            # eigenpair is computed on GPU and the left eigenpair is
            # paired on host via scipy.linalg.eig(F.T).
            #
            # cupy 14 + CUDA 12.x do not yet ship geev — calling
            # cupy.linalg.eig raises "geev is not available".  Detect
            # this once and cache the result so each symmetry block
            # falls back to the v1.7.3 scipy.sparse.linalg.eigs path
            # without paying detection overhead per iteration.
            n_i = F_i.shape[0]
            use_gpu_eig_here = (
                USE_GPU_EIG_MIR
                and n_i >= 1500
                and self.nev < n_i - 1
                and _gpu_eig_available_cached_mir())
            if use_gpu_eig_here:
                # Right eigenpair via eig_dispatch (GPU full + slice).
                _w_r, ur_i = eig_dispatch(F_i, k=int(self.nev),
                        left=False, right=True, which='SR')
                # Left eigenpair via host LAPACK on F.T (full spectrum).
                from scipy.linalg import eig as _scipy_eig
                ene_l_all, vl_all = _scipy_eig(F_i.T, left=False, right=True,
                        check_finite=False)
                ene_l_idx: List[int] = []
                used = set()  # type: set
                for w in _w_r:
                    d = np.abs(ene_l_all - w)
                    order = np.argsort(d)
                    for j in order:
                        ji = int(j)
                        if ji not in used:
                            ene_l_idx.append(ji)
                            used.add(ji)
                            break
                ul_i = vl_all[:, np.asarray(ene_l_idx, dtype=int)].T
                ene_i = np.diag(_w_r)
            else:
                # eigs returns (eigenvalues, eigenvectors) where eigenvectors is (n, k)
                _, ul_i = eigs(F_i.T, k = self.nev, which = 'SR', maxiter = 1000)
                ul_i = ul_i.T  # (nev, n)
                ene_i, ur_i = eigs(F_i, k = self.nev, which = 'SR', maxiter = 1000)
                # ur_i is (n, nev), ene_i is (nev,)
                ene_i = np.diag(ene_i)

            # make eigenvectors orthogonal
            overlap = ul_i @ ur_i  # (nev, nev)
            ul_i = np.linalg.solve(overlap, ul_i)

            # unit matrices (one column per half-particle)
            unit_i = np.zeros((self.nev ** 2, n_half_particles), dtype = complex)
            for ip in range(n_half_particles):
                ind = half_indices[ip]
                chunk = ul_i[:, ind] @ ur_i[ind, :]  # (nev, nev)
                unit_i[:, ip] = chunk.ravel()

            self.ur.append(ur_i)
            self.ul.append(ul_i)
            self.ene.append(ene_i)
            self.unit.append(unit_i)
            # v1.7.3 Phase 2: per-symmetry pool drain — each eigs block can
            # leave Arnoldi scratch in the pool when sparse eigs routes
            # through cupy.  Drain inside the loop so n_sym blocks don't
            # accumulate.
            if _CUPY_OK_EIGMIR:
                _gpu_pool_cleanup_eigmir()

        # v1.7.3 Phase 2: final eigs-loop pool drain.
        if _CUPY_OK_EIGMIR:
            _gpu_pool_cleanup_eigmir()

        if enei is not None:
            self._init_matrices(enei)

    def _init_matrices(self, enei: float) -> 'BEMStatEigMirror':
        """Initialize matrices for BEM solver.

        MATLAB: @bemstateigmirror/subsref.m case '()'
        """
        if self.enei is not None and np.isclose(self.enei, enei):
            return self

        # v1.7.3 Phase 2: drop the previous wavelength's resolvent matrices
        # before new device allocations.  Pattern mirrors BEMStatMirror /
        # BEMStat (v1.7.2).
        self.mat = None
        _gpu_pool_cleanup_eigmir(apply_limit = True)

        # dielectric functions
        eps_vals = [eps_func(enei)[0] for eps_func in self.p.eps]

        # inside and outside dielectric function
        eps1_arr = np.array([eps_vals[int(self.p.inout[j, 0]) - 1]
                            for j in range(self.p.inout.shape[0])])
        eps2_arr = np.array([eps_vals[int(self.p.inout[j, 1]) - 1]
                            for j in range(self.p.inout.shape[0])])

        # Lambda [Garcia de Abajo, Eq. (23)]
        Lambda = 2 * np.pi * (eps1_arr + eps2_arr) / (eps1_arr - eps2_arr)

        self.mat = []
        for i in range(len(self.ur)):
            # BEM resolvent matrix from eigenmodes
            unit_lambda = self.unit[i] @ Lambda[:]  # (nev^2,)
            unit_lambda_mat = unit_lambda.reshape(self.nev, self.nev)
            resolvent = unit_lambda_mat + self.ene[i]
            # resolvent is (nev, nev) — small; the ur @ (...) GEMM dominates
            # for large mesh and benefits from GPU dispatch.
            # solve_dispatch + matmul_dispatch already honour
            # MNPBEM_VRAM_SHARE_* via their env-var auto-wiring.
            inv_ul = solve_dispatch(resolvent, self.ul[i])
            self.mat.append(-matmul_dispatch(self.ur[i], inv_ul))
            # v1.7.3 Phase 2: drop per-block scratch so the next symmetry
            # iteration sees a clean pool.
            del inv_ul, resolvent, unit_lambda, unit_lambda_mat
            if _CUPY_OK_EIGMIR:
                _gpu_pool_cleanup_eigmir()

        self.enei = enei
        _gpu_pool_cleanup_eigmir()
        return self

    def solve(self, exc: CompStructMirror) -> Tuple[CompStructMirror, 'BEMStatEigMirror']:
        """Surface charge for given excitation.

        MATLAB: @bemstateigmirror/mldivide.m

        Parameters
        ----------
        exc : CompStructMirror
            External excitation with field 'phip'

        Returns
        -------
        sig : CompStructMirror
            Surface charge
        obj : BEMStatEigMirror
            Updated solver
        """
        self._init_matrices(exc.enei)

        sig = CompStructMirror(self.p, exc.enei, exc.fun)

        for i in range(len(exc.val)):
            ind = self.p.symindex(exc.val[i].symval[-1, :])

            sig_val = _matmul(self.mat[ind], exc.val[i].phip)

            val = CompStruct(self.p, exc.enei, sig = sig_val)
            val.symval = exc.val[i].symval
            sig.val.append(val)

        return sig, self

    def __truediv__(self, exc: CompStructMirror) -> Tuple[CompStructMirror, 'BEMStatEigMirror']:
        return self.solve(exc)

    def __mul__(self, sig: CompStructMirror) -> CompStructMirror:
        """Induced potential for given surface charge.

        MATLAB: @bemstateigmirror/mtimes.m
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

        MATLAB: @bemstateigmirror/potential.m
        """
        return self.g.potential(sig, inout)

    def field(self,
            sig: CompStructMirror,
            inout: int = 2) -> CompStructMirror:
        """Electric field inside/outside of particle surface.

        MATLAB: @bemstateigmirror/field.m
        """
        return self.g.field(sig, inout)

    def __call__(self, enei: float) -> 'BEMStatEigMirror':
        return self._init_matrices(enei)

    def clear(self) -> 'BEMStatEigMirror':
        """Clear cached resolvent matrices and force rebuild on next solve.

        v1.7.3 Phase 2: API parity with BEMStat / BEMStatLayer / BEMStatMirror
        / BEMStatIter.  Drops the per-symmetry ``mat`` list and resets the
        cache gate so a subsequent solve() at the same wavelength does not
        skip rebuild.
        """
        self.mat = None
        self.enei = None
        if _CUPY_OK_EIGMIR:
            _gpu_pool_cleanup_eigmir()
        return self

    def __repr__(self) -> str:
        status = 'enei={}'.format(self.enei) if self.enei is not None else 'not initialized'
        return 'BEMStatEigMirror(p={}, nev={}, {})'.format(self.p, self.nev, status)


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
        return a @ x
    return a @ x
