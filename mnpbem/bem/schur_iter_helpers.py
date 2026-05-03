import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, gmres


# Schur complement reduction over the iterative (GMRES) BEM solvers
# ------------------------------------------------------------------
# The dense v1.2.0 path (mnpbem/bem/schur_helpers.py) eliminates the
# EpsNonlocal cover-layer "shell" faces by inverting G_ss directly.
# That is impossible when G is held as an HMatrix (no cheap inverse
# is available). Instead we expose a LinearOperator that applies
#
#     A_eff(x_c) = A_cc(x_c) - A_cs * A_ss^{-1} * A_sc(x_c)
#
# implicitly: each call only needs full matvecs A * v_full and a
# linear solve against A_ss.  A_ss^{-1} can be evaluated via dense
# LU on the (small) shell block, an inner GMRES, or any user-supplied
# callable.  The reduced size of the GMRES Krylov space is
# (#core_faces) * components instead of (#all_faces) * components,
# which is the savings the user is paying for.
#
# Conventions
# -----------
# - shell_face_indices / core_face_indices are face-level integer
#   arrays produced by ``schur_helpers.detect_shell_core_partition``.
# - ``components`` repeats the partition across ``components`` blocks
#   of size ``nfaces``: BEMStatIter uses components=1, BEMRetIter
#   uses components=8 (phi, a_x, a_y, a_z, phip, ap_x, ap_y, ap_z).
#   The 8-component packed vector layout is column-major
#   (``order='F'``), so the per-face shell/core indices repeat with
#   stride ``nfaces`` between the eight component blocks.


def _lift_indices(
        face_indices: np.ndarray,
        nfaces: int,
        components: int) -> np.ndarray:

    # Lift face-level shell/core indices into a packed-vector index
    # array of size ``len(face_indices) * components`` matching the
    # column-major (order='F') 8N layout used by BEMRetIter.
    face_indices = np.asarray(face_indices, dtype = np.int64)
    if components == 1:
        return face_indices.copy()

    out = np.empty(face_indices.size * components, dtype = np.int64)
    for k in range(components):
        out[k * face_indices.size:(k + 1) * face_indices.size] = face_indices + k * nfaces
    return out


class SchurIterOperator(LinearOperator):

    # MNPBEM v1.5.0: Schur reduced GMRES operator on top of a
    # block-structured BEM matvec.  Combines with HMatrix-backed
    # Green functions because the full matvec ``A_full(x_full)``
    # is the only thing we ever ask of the upstream solver.
    #
    # v1.6.0 (B-Schur): added ``eps_form`` to communicate whether the
    # upstream ``_afun`` uses operator-form eps (β v1.5.1 fix).  When
    # ``eps_form='operator'``, the dense A_ss probe is bypassed in favor
    # of an inner GMRES solver because the operator-form full matvec
    # produces an A_ss whose dense probe is ill-conditioned for nonlocal
    # cover-layer geometries.  See ``/tmp/b_schur_derivation.md``.

    def __init__(self,
            A_full_matvec: Callable[[np.ndarray], np.ndarray],
            shell_face_indices: np.ndarray,
            core_face_indices: np.ndarray,
            nfaces: int,
            components: int = 1,
            dtype: Any = complex,
            g_ss_solver: str = 'auto',
            inner_tol: float = 1e-8,
            inner_maxit: int = 200,
            g_ss_dense: Optional[np.ndarray] = None,
            user_g_ss_solver: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            eps_form: str = 'pointwise',
            eps_diag: Optional[Dict[str, Any]] = None) -> None:

        # ``g_ss_solver`` selects the strategy used to apply A_ss^{-1}:
        #     'lu_dense'    -- assemble A_ss as a dense block (probe with unit
        #                      vectors) and factorize once.  Cheap when
        #                      ``len(shell_face_indices) * components`` is
        #                      small, accurate to machine precision.
        #     'gmres'       -- inner GMRES against A_ss using the same matvec
        #                      (no preconditioner).  Cheap memory but slow
        #                      when many outer iterations are required.
        #     'callable'    -- user supplied ``user_g_ss_solver`` (e.g. a
        #                      preconditioner from BEMRetIter._mfun).
        #     'auto'        -- v1.6.0 ``eps_form='operator'`` always picks
        #                      'gmres' (probe ill-conditioned).  Otherwise
        #                      pick 'lu_dense' if the shell block is small
        #                      (< 500 faces * components), else 'gmres'.
        # ``eps_form`` (v1.6.0):
        #     'pointwise'   -- legacy v1.5.0; upstream ``_afun`` applies eps
        #                      after the Green matvec (uniform-eps fast path).
        #     'operator'    -- β v1.5.1 ``_afun``: eps multiplied into σ
        #                      *before* the Green matvec.  For nonlocal
        #                      cover-layer geometries the dense probe of
        #                      A_ss inherits a near-singular eps-weighted
        #                      mixing — switch to inner GMRES instead.
        # ``eps_diag``: optional block-decomposed per-face eps for diagnostics
        #     and future analytical preconditioning.  Format:
        #         {'shell': array_of_eps_for_shell_faces or scalar,
        #          'core' : array_of_eps_for_core_faces  or scalar}
        assert g_ss_solver in {'auto', 'lu_dense', 'gmres', 'callable'}, \
                '[error] g_ss_solver must be one of <auto|lu_dense|gmres|callable>, got <{}>'.format(g_ss_solver)
        assert eps_form in {'pointwise', 'operator'}, \
                '[error] eps_form must be one of <pointwise|operator>, got <{}>'.format(eps_form)

        self._A_full_matvec = A_full_matvec
        self._shell_face_idx = np.asarray(shell_face_indices, dtype = np.int64)
        self._core_face_idx = np.asarray(core_face_indices, dtype = np.int64)
        self.nfaces = int(nfaces)
        self.components = int(components)

        # Lifted index arrays operating on the full packed vector.
        self._shell_idx = _lift_indices(self._shell_face_idx, self.nfaces, self.components)
        self._core_idx = _lift_indices(self._core_face_idx, self.nfaces, self.components)
        self._N = self.nfaces * self.components
        self._n_shell = self._shell_idx.size
        self._n_core = self._core_idx.size

        self._dtype = np.dtype(dtype)
        self._eps_form = eps_form
        self._eps_diag = eps_diag

        # LinearOperator interface (acts on N_core sized vectors).
        super(SchurIterOperator, self).__init__(self._dtype, (self._n_core, self._n_core))

        # Resolve g_ss_solver strategy.
        if g_ss_solver == 'auto':
            # v1.6.0 (B-Schur): for operator-form eps the inner GMRES path
            # becomes prohibitively expensive (each outer GMRES iter triggers
            # a full inner Krylov sweep, doubling the per-matvec cost from
            # the β v1.5.1 eps-apply + extra G/H matvec).  Dense LU probe
            # is correct for any A_full and gives O(n²) per outer iter.
            # Bump the lu_dense / gmres cutoff up to 4096 (≈128 MB at
            # complex128; 8N=4096 covers 60-face nonlocal core-shell up
            # to ~512 core faces) so operator-form callers get the cheap
            # path.  Above the cutoff we fall back to inner GMRES because
            # the dense probe + LU factor would exceed memory.
            if eps_form == 'operator':
                g_ss_solver = 'lu_dense' if self._n_shell <= 4096 else 'gmres'
            else:
                g_ss_solver = 'lu_dense' if self._n_shell < 500 else 'gmres'
        self._g_ss_solver_kind = g_ss_solver

        self._inner_tol = float(inner_tol)
        self._inner_maxit = int(inner_maxit)
        self._user_g_ss_solver = user_g_ss_solver

        self._A_ss_lu = None
        self._A_ss_piv = None
        if g_ss_solver == 'lu_dense':
            if g_ss_dense is None:
                g_ss_dense = self._probe_dense_block(self._shell_idx, self._shell_idx)
            self._A_ss_dense = np.asarray(g_ss_dense)
            self._A_ss_lu, self._A_ss_piv = lu_factor(self._A_ss_dense, check_finite = False)
        elif g_ss_solver == 'callable':
            assert user_g_ss_solver is not None, \
                    '[error] g_ss_solver=<callable> requires <user_g_ss_solver>'

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _probe_dense_block(self,
            row_idx: np.ndarray,
            col_idx: np.ndarray) -> np.ndarray:

        # Assemble A[row_idx, col_idx] by probing the full matvec with
        # unit vectors at each column index.  This costs len(col_idx)
        # full matvecs and is only invoked when ``g_ss_solver='lu_dense'``
        # (cheap shell blocks, < 500 entries).
        m = row_idx.size
        n = col_idx.size
        block = np.empty((m, n), dtype = self._dtype)
        e = np.zeros(self._N, dtype = self._dtype)
        for j in range(n):
            e[col_idx[j]] = 1.0
            y = self._A_full_matvec(e)
            block[:, j] = y[row_idx]
            e[col_idx[j]] = 0.0
        return block

    def _solve_g_ss(self,
            rhs: np.ndarray) -> np.ndarray:

        # Apply A_ss^{-1} to a shell-sized vector ``rhs``.
        if self._g_ss_solver_kind == 'lu_dense':
            return lu_solve((self._A_ss_lu, self._A_ss_piv), rhs, check_finite = False)

        if self._g_ss_solver_kind == 'callable':
            return self._user_g_ss_solver(rhs)

        # 'gmres': inner Krylov solve against A_ss.  We expose A_ss as a
        # LinearOperator using the full matvec restricted to shell rows
        # of a shell-only inflated vector.
        def _mv(x_shell: np.ndarray) -> np.ndarray:
            x_full = np.zeros(self._N, dtype = self._dtype)
            x_full[self._shell_idx] = x_shell
            y_full = self._A_full_matvec(x_full)
            return y_full[self._shell_idx]

        op = LinearOperator((self._n_shell, self._n_shell), matvec = _mv, dtype = self._dtype)
        x, _ = gmres(op, rhs, rtol = self._inner_tol, maxiter = self._inner_maxit,
                restart = min(self._n_shell, 50))
        return x

    # ------------------------------------------------------------------
    # LinearOperator interface
    # ------------------------------------------------------------------
    def _matvec(self,
            x_core: np.ndarray) -> np.ndarray:

        # Compute A_eff(x_core) = A_cc x_core - A_cs A_ss^{-1} A_sc x_core.
        x_core = x_core.ravel()

        # Step 1: inflate x_core -> x_full (shell block zero), apply A_full,
        # extract shell-row contribution = A_sc x_core.
        x_full = np.zeros(self._N, dtype = self._dtype)
        x_full[self._core_idx] = x_core
        y1_full = self._A_full_matvec(x_full)
        a_sc_x = y1_full[self._shell_idx]
        a_cc_x = y1_full[self._core_idx]

        # Step 2: apply A_ss^{-1}.
        z_shell = self._solve_g_ss(a_sc_x)

        # Step 3: inflate z_shell -> full (core block zero), apply A_full,
        # extract core rows = A_cs z_shell.
        z_full = np.zeros(self._N, dtype = self._dtype)
        z_full[self._shell_idx] = z_shell
        y2_full = self._A_full_matvec(z_full)
        a_cs_z = y2_full[self._core_idx]

        return a_cc_x - a_cs_z

    def reduce_rhs(self,
            b_full: np.ndarray) -> np.ndarray:

        # b_eff = b_c - A_cs * A_ss^{-1} * b_s
        b_full = np.asarray(b_full).ravel()
        b_s = b_full[self._shell_idx]
        b_c = b_full[self._core_idx]

        z_shell = self._solve_g_ss(b_s)

        z_full = np.zeros(self._N, dtype = self._dtype)
        z_full[self._shell_idx] = z_shell
        corr = self._A_full_matvec(z_full)[self._core_idx]

        return b_c - corr

    def recover_full(self,
            sigma_core: np.ndarray,
            b_full: np.ndarray) -> np.ndarray:

        # Reconstruct sigma_full from (sigma_core, b_full):
        #     sigma_shell = A_ss^{-1} * (b_s - A_sc * sigma_core)
        sigma_core = np.asarray(sigma_core).ravel()
        b_full = np.asarray(b_full).ravel()
        b_s = b_full[self._shell_idx]

        x_full = np.zeros(self._N, dtype = self._dtype)
        x_full[self._core_idx] = sigma_core
        a_sc_sig = self._A_full_matvec(x_full)[self._shell_idx]

        sigma_shell = self._solve_g_ss(b_s - a_sc_sig)

        out = np.empty(self._N, dtype = np.result_type(sigma_core.dtype, sigma_shell.dtype))
        out[self._core_idx] = sigma_core
        out[self._shell_idx] = sigma_shell
        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def info(self) -> Dict[str, Any]:

        return {
            'nfaces': self.nfaces,
            'components': self.components,
            'n_shell_faces': int(self._shell_face_idx.size),
            'n_core_faces': int(self._core_face_idx.size),
            'n_full_dof': int(self._N),
            'n_core_dof': int(self._n_core),
            'n_shell_dof': int(self._n_shell),
            'g_ss_solver': self._g_ss_solver_kind,
            'eps_form': self._eps_form,
            'has_eps_diag': self._eps_diag is not None,
        }


def detect_iter_partition(particle: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:

    # Re-export the v1.2.0 partition detector unchanged so callers
    # need only one import.
    from .schur_helpers import detect_shell_core_partition
    return detect_shell_core_partition(particle)


def schur_iter_memory_estimate(
        nfaces_total: int,
        nfaces_shell: int,
        components: int = 1) -> Dict[str, Any]:

    # Memory estimate for the iterative Schur path.  Compared to the
    # dense v1.2.0 estimator this accounts for the per-face component
    # multiplier (8 for BEMRetIter, 1 for BEMStatIter) and reports
    # the GMRES Krylov-vector savings (sub-space size scales with
    # the reduced DOF count).
    nfaces_core = nfaces_total - nfaces_shell

    full_dof = nfaces_total * components
    core_dof = nfaces_core * components
    shell_dof = nfaces_shell * components

    bytes_per_entry = 16  # complex128

    full_krylov_bytes = full_dof * bytes_per_entry
    core_krylov_bytes = core_dof * bytes_per_entry
    g_ss_dense_bytes = shell_dof * shell_dof * bytes_per_entry

    return {
        'nfaces_total': nfaces_total,
        'nfaces_shell': nfaces_shell,
        'nfaces_core': nfaces_core,
        'components': components,
        'full_dof': full_dof,
        'core_dof': core_dof,
        'shell_dof': shell_dof,
        'krylov_vector_full_bytes': full_krylov_bytes,
        'krylov_vector_core_bytes': core_krylov_bytes,
        'g_ss_dense_bytes': g_ss_dense_bytes,
        'krylov_reduction_ratio': core_krylov_bytes / full_krylov_bytes if full_krylov_bytes else 0.0,
    }
