import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator

from ..greenfun import CompStruct
from ..utils.gpu import (
    lu_factor_dispatch, lu_solve_dispatch, lu_solve_native,
    eye_like_lu, to_host, is_cupy_array,
)
from ..utils.matlab_compat import msqrt
from .bem_iter import BEMIter


class BEMRetIter(BEMIter):

    # MATLAB: @bemretiter properties (Constant)
    name = 'bemsolver'
    needs = {'sim': 'ret'}

    def __init__(self,
            p: Any,
            enei: Optional[float] = None,
            **options: Any) -> None:

        # Schur option (v1.5.0): cover-layer (EpsNonlocal) shell-face
        # elimination on the iterative retarded path.  Combines with
        # hmatrix=True via SchurIterOperator: the eight retarded
        # components (phi, a_x, a_y, a_z, phip, ap_x, ap_y, ap_z) share
        # the same face-level partition, lifted to the 8N packed vector
        # layout used by ``_pack`` / ``_unpack``.
        self._schur_opt = options.pop('schur', False)
        self._schur_g_ss_solver = options.pop('schur_g_ss_solver', 'auto')
        self._schur_inner_tol = options.pop('schur_inner_tol', 1e-8)
        self._schur_inner_maxit = options.pop('schur_inner_maxit', 200)
        self._schur_active = False
        self._shell_face_idx = None
        self._core_face_idx = None
        self._schur_op = None

        # H-matrix (v1.3.0): opt-in ACA acceleration of Green functions.
        # When True, the matvec used by GMRES uses HMatrix @ x compression
        # (O(N log N) memory) rather than dense ndarrays.
        self._hmatrix = bool(options.pop('hmatrix', False))
        self._htol = options.pop('htol', 1e-6)
        self._kmax = options.pop('kmax', [4, 100])
        self._cleaf = options.pop('cleaf', 200)
        self._fadmiss = options.pop('fadmiss', None)
        self._eta = options.pop('eta', 2.5)

        # H-matrix LU preconditioner (v1.5.0, agent alpha):
        #   'auto'      — pick dense for small mesh, tree for large
        #   'none'      — disable preconditioner entirely (legacy v1.3 behaviour)
        #   'hlu_dense' — alpha-1 dense LU on H-matrix.full()
        #   'hlu_tree'  — alpha-2 recursive block-Schur LU
        # Active only on the H-matrix code path (hmatrix=True).
        self._hlu_mode = options.pop('preconditioner', 'auto')
        self._htol_precond = options.pop('htol_precond', 1e-4)
        self._hlu_object = None  # built lazily inside solve()

        # Default v1.3.0 ``precond``: when the H-matrix path is active and
        # the user did not explicitly choose the legacy preconditioner, we
        # leave it disabled. The new v1.5.0 H-matrix LU preconditioner is
        # plumbed separately and only acts when self._hmatrix is True.
        if self._hmatrix and 'precond' not in options:
            options['precond'] = None

        # Initialize BEMIter base class
        super(BEMRetIter, self).__init__(**options)

        # MATLAB: @bemretiter properties
        self.p = p
        self.enei = None
        self.g = None

        # MATLAB: @bemretiter properties (Access = private)
        self._op = options
        self._sav = None
        self._k = None
        self._eps1 = None
        self._eps2 = None
        self._nvec = p.nvec
        self._G1 = None
        self._H1 = None
        self._G2 = None
        self._H2 = None

        # User-supplied refinement hook (e.g. coverlayer.refine). Stripped
        # before forwarding to CompGreenRet, applied at the BEM matrix
        # level inside _init_matrices(). MATLAB bemretiter forwards refun
        # via varargin → compgreenretiter → greenret/private/init.m.
        self._refun = options.pop('refun', None)
        self._op = options

        # H-matrix path is incompatible with refun for now (refun densifies
        # G/H pairs, defeating the compression). Fall back to dense if both
        # are requested.
        if self._hmatrix and self._refun is not None:
            raise NotImplementedError(
                '[error] BEMRetIter <hmatrix> + <refun> not supported '
                '(refun densifies the Green pairs). Disable one.')

        # Green function. With ``hmatrix=True`` we pull the ACA wrapper from
        # mnpbem.greenfun; otherwise the dense CompGreenRet is used (legacy
        # path preserved for tests / demos).
        # MATLAB: obj.g = aca.compgreenret(p, varargin{:}, ...)
        self._init_green(p, **options)

        # Initialize for given wavelength
        if enei is not None:
            self._init_matrices(enei)

    def _init_green(self,
            p: Any,
            **options: Any) -> None:

        # MATLAB: bemretiter/private/init.m
        if self._hmatrix:
            from ..greenfun import ACACompGreenRet
            # MATLAB stores kmax as [k_min, k_max]; HMatrix expects scalar.
            # Take the upper bound when forwarding.
            kmax_scalar = (max(self._kmax) if hasattr(self._kmax, '__iter__')
                    else self._kmax)
            htol_scalar = (max(self._htol) if hasattr(self._htol, '__iter__')
                    else self._htol)
            aca_kwargs = {
                'htol': htol_scalar,
                'kmax': kmax_scalar,
                'cleaf': self._cleaf,
                'eta': self._eta,
            }
            if self._fadmiss is not None:
                aca_kwargs['fadmiss'] = self._fadmiss
            self.g = ACACompGreenRet(p, **aca_kwargs, **options)
        else:
            from ..greenfun import CompGreenRet
            self.g = CompGreenRet(p, p, **options)

    def _init_matrices(self,
            enei: float) -> 'BEMRetIter':

        # MATLAB: bemretiter/private/initmat.m
        if self.enei is not None and self.enei == enei:
            return self

        self.enei = enei

        # Wavenumber
        self._k = 2 * np.pi / enei

        # Dielectric function
        self._eps1 = self.p.eps1(enei)
        self._eps2 = self.p.eps2(enei)

        # Green functions and surface derivatives
        # MATLAB: G1 = g{1,1}.G(enei) - g{2,1}.G(enei)
        G11 = self.g.eval(0, 0, 'G', enei)
        G21 = self.g.eval(1, 0, 'G', enei)
        G22 = self.g.eval(1, 1, 'G', enei)
        G12 = self.g.eval(0, 1, 'G', enei)

        self._G1 = G11 - G21 if not (isinstance(G21, (int, float)) and G21 == 0) else G11
        self._G2 = G22 - G12 if not (isinstance(G12, (int, float)) and G12 == 0) else G22

        H11 = self.g.eval(0, 0, 'H1', enei)
        H21 = self.g.eval(1, 0, 'H1', enei)
        H22 = self.g.eval(1, 1, 'H2', enei)
        H12 = self.g.eval(0, 1, 'H2', enei)

        self._H1 = H11 - H21 if not (isinstance(H21, (int, float)) and H21 == 0) else H11
        self._H2 = H22 - H12 if not (isinstance(H12, (int, float)) and H12 == 0) else H22

        # Optional user-supplied refinement (coverlayer.refine for nonlocal
        # cover-layer effects). Applied to dense G/H pairs. If ACA H-matrix
        # acceleration is in use the matrices are densified for refun and
        # the refined dense result is kept (refun touches a small set of
        # face pairs, so densification is acceptable here).
        if self._refun is not None:
            G1 = self._G1.full() if hasattr(self._G1, 'full') and not isinstance(self._G1, np.ndarray) else self._G1
            H1 = self._H1.full() if hasattr(self._H1, 'full') and not isinstance(self._H1, np.ndarray) else self._H1
            G2 = self._G2.full() if hasattr(self._G2, 'full') and not isinstance(self._G2, np.ndarray) else self._G2
            H2 = self._H2.full() if hasattr(self._H2, 'full') and not isinstance(self._H2, np.ndarray) else self._H2
            G1, H1 = self._refun(self.g, G1, H1)
            G2, H2 = self._refun(self.g, G2, H2)
            self._G1, self._H1 = G1, H1
            self._G2, self._H2 = G2, H2

        # Initialize preconditioner
        if self.precond is not None:
            self._init_precond(enei)

        # Schur (v1.5.0): detect cover-layer partition and prepare the
        # SchurIterOperator wrapping the 8N packed _afun.  The Schur
        # operator probes _afun for the shell block (lu_dense path) or
        # delegates A_ss^{-1} to inner GMRES.  For BEMRetIter the eight
        # retarded components share the same face-level partition --
        # SchurIterOperator with components=8 lifts the indices to the
        # full 8N packed layout (column-major / order='F').
        self._schur_active = False
        self._schur_op = None
        if self._schur_opt:
            from .schur_iter_helpers import SchurIterOperator, detect_iter_partition
            partition = detect_iter_partition(self.p)
            if partition is not None:
                shell_idx, core_idx = partition
                nfaces = self.p.n if hasattr(self.p, 'n') else self.p.nfaces
                self._shell_face_idx = shell_idx
                self._core_face_idx = core_idx
                self._schur_op = SchurIterOperator(
                        self._afun,
                        shell_idx,
                        core_idx,
                        nfaces = nfaces,
                        components = 8,
                        dtype = complex,
                        g_ss_solver = self._schur_g_ss_solver,
                        inner_tol = self._schur_inner_tol,
                        inner_maxit = self._schur_inner_maxit)
                self._schur_active = True

        return self

    def _compress(self,
            hmat: Any) -> Any:

        # MATLAB: bemretiter/private/compress.m
        # The dense-LU preconditioner needs an ndarray; if we got an HMatrix
        # we densify it here. Memory cost is the standard dense N x N — only
        # invoked when the user explicitly opts into the dense preconditioner.
        if hasattr(hmat, 'full') and not isinstance(hmat, np.ndarray):
            return hmat.full()
        return hmat

    def _init_precond(self,
            enei: float) -> None:

        # MATLAB: bemretiter/private/initprecond.m
        # Garcia de Abajo and Howie, PRB 65, 115418 (2002)
        k = 2 * np.pi / enei
        eps1 = self._eps1
        eps2 = self._eps2
        nvec = self._nvec

        G1 = self._compress(self._G1)
        H1 = self._compress(self._H1)
        G2 = self._compress(self._G2)
        H2 = self._compress(self._H2)

        # Bug 2 fix: coerce any cupy operands down to host before the
        # CPU-style dense preconditioner pipeline so the eps_diag /
        # H @ G^{-1} GEMMs do not mix devices.
        if is_cupy_array(G1): G1 = to_host(G1)
        if is_cupy_array(G2): G2 = to_host(G2)
        if is_cupy_array(H1): H1 = to_host(H1)
        if is_cupy_array(H2): H2 = to_host(H2)

        # Dielectric as diagonal matrices for matrix operations
        if np.isscalar(eps1) or (isinstance(eps1, np.ndarray) and eps1.ndim == 0):
            eps1_diag = eps1
            eps2_diag = eps2
        else:
            eps1_diag = np.diag(eps1)
            eps2_diag = np.diag(eps2)

        # LU factorizations of Green functions
        G1_lu = lu_factor_dispatch(G1)
        G2_lu = lu_factor_dispatch(G2)
        # Bug 2 fix: build identity on the same device as the LU and
        # bring the inverse back to host for the H @ G^{-1} GEMM.
        eye_g1 = eye_like_lu(G1_lu, G1.shape[0])
        eye_g2 = eye_like_lu(G2_lu, G2.shape[0])
        G1i = to_host(lu_solve_native(G1_lu, eye_g1))
        G2i = to_host(lu_solve_native(G2_lu, eye_g2))

        # Sigma matrices [Eq. (21)]
        Sigma1 = H1 @ G1i
        Sigma2 = H2 @ G2i

        # LU factorization of Delta matrix
        Delta_lu = lu_factor_dispatch(Sigma1 - Sigma2)
        eye_d = eye_like_lu(Delta_lu, Sigma1.shape[0])
        Deltai = to_host(lu_solve_native(Delta_lu, eye_d))

        # deps = eps1 - eps2
        if np.isscalar(eps1_diag):
            deps = eps1_diag - eps2_diag
        else:
            deps = eps1_diag - eps2_diag

        # Sigma matrix [Eq. (21,22)]
        # MATLAB: Sigma = eps1 * Sigma1 - eps2 * Sigma2 + k^2 * deps * fun(Deltai, nvec) * deps
        # fun(Deltai, nvec) = sum_i nvec_i * Deltai * nvec_i
        Deltai_nvec = self._decorate_deltai(Deltai, nvec)

        if np.isscalar(eps1_diag):
            Sigma_mat = eps1_diag * Sigma1 - eps2_diag * Sigma2 + k ** 2 * deps * Deltai_nvec * deps
        else:
            Sigma_mat = eps1_diag @ Sigma1 - eps2_diag @ Sigma2 + k ** 2 * deps @ Deltai_nvec @ deps

        Sigma_lu = lu_factor_dispatch(Sigma_mat)

        # Save variables for preconditioner
        sav = {}
        sav['k'] = k
        sav['nvec'] = nvec
        sav['G1_lu'] = G1_lu
        sav['G2_lu'] = G2_lu
        sav['eps1'] = eps1_diag
        sav['eps2'] = eps2_diag
        sav['Sigma1'] = Sigma1
        sav['Delta_lu'] = Delta_lu
        sav['Sigma_lu'] = Sigma_lu

        self._sav = sav

    @staticmethod
    def _decorate_deltai(
            Deltai: np.ndarray,
            nvec: np.ndarray) -> np.ndarray:

        # MATLAB: fun(Deltai, nvec) in initprecond.m
        # Deltai_nvec = nvec1 * Deltai * nvec1 + nvec2 * Deltai * nvec2 + nvec3 * Deltai * nvec3
        n = nvec.shape[0]
        result = np.zeros((n, n), dtype = Deltai.dtype)
        for i in range(3):
            nvec_i = np.diag(nvec[:, i])
            result = result + nvec_i @ Deltai @ nvec_i
        return result

    def _pack(self,
            phi: np.ndarray,
            a: np.ndarray,
            phip: np.ndarray,
            ap: np.ndarray) -> np.ndarray:

        # MATLAB: bemretiter/private/pack.m
        # MATLAB uses column-major (:) flatten, so we use order='F'.
        total_len = phi.size + a.size + phip.size + ap.size
        vec = np.empty(total_len, dtype = complex)
        offset = 0
        for arr in [phi, a, phip, ap]:
            flat = arr.ravel(order = 'F')
            vec[offset:offset + flat.size] = flat
            offset += flat.size
        return vec

    def _unpack(self,
            vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # MATLAB: bemretiter/private/unpack.m
        # MATLAB uses column-major reshape, so we use order='F'.
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces

        # last dimension
        siz = int(vec.size / (8 * n))

        # reshape vector (column-major to match MATLAB)
        vec_2d = vec.reshape(-1, 8, order = 'F')

        # extract potentials from vector
        phi = vec_2d[:, 0].reshape(n, siz, order = 'F') if siz > 1 else vec_2d[:, 0].reshape(n)
        a = vec_2d[:, 1:4].reshape(n, 3, siz, order = 'F') if siz > 1 else vec_2d[:, 1:4].reshape(n, 3)
        phip = vec_2d[:, 4].reshape(n, siz, order = 'F') if siz > 1 else vec_2d[:, 4].reshape(n)
        ap = vec_2d[:, 5:8].reshape(n, 3, siz, order = 'F') if siz > 1 else vec_2d[:, 5:8].reshape(n, 3)

        return phi, a, phip, ap

    @staticmethod
    def _outer(
            nvec: np.ndarray,
            val: Any,
            mul: Optional[np.ndarray] = None) -> Any:

        # MATLAB: bemretiter/private/outer.m
        if isinstance(val, (int, float)) and val == 0:
            return 0

        if mul is not None:
            if val.ndim == 1:
                val = val * mul
            else:
                val = val * mul[:, np.newaxis] if mul.ndim == 1 else val * mul

        if val.ndim == 1:
            # val: (n,), nvec: (n, 3) -> result: (n, 3)
            return nvec * val[:, np.newaxis]
        else:
            # val: (n, siz), nvec: (n, 3) -> result: (n, 3, siz)
            siz = val.shape[1]
            n = val.shape[0]
            result = np.empty((n, 3, siz), dtype = val.dtype)
            for i in range(3):
                result[:, i, :] = val * nvec[:, i:i + 1]
            return result

    @staticmethod
    def _inner(
            nvec: np.ndarray,
            a: Any,
            mul: Optional[np.ndarray] = None) -> Any:

        # MATLAB: bemretiter/private/inner.m
        if isinstance(a, (int, float)) and a == 0:
            return 0

        if a.ndim == 2:
            # a: (n, 3), nvec: (n, 3) -> result: (n,)
            result = np.sum(a * nvec, axis = 1)
        elif a.ndim == 3:
            # a: (n, 3, siz), nvec: (n, 3) -> result: (n, siz)
            result = np.sum(a * nvec[:, :, np.newaxis], axis = 1)
        else:
            result = a

        if mul is not None:
            if result.ndim == 1:
                result = result * mul
            else:
                result = result * mul[:, np.newaxis] if mul.ndim == 1 else result * mul

        return result

    def _excitation(self,
            exc: CompStruct) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # MATLAB: bemretiter/private/excitation.m
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces

        # Default values for potentials
        phi1 = getattr(exc, 'phi1', 0)
        phi1p = getattr(exc, 'phi1p', 0)
        a1 = getattr(exc, 'a1', 0)
        a1p = getattr(exc, 'a1p', 0)
        phi2 = getattr(exc, 'phi2', 0)
        phi2p = getattr(exc, 'phi2p', 0)
        a2 = getattr(exc, 'a2', 0)
        a2p = getattr(exc, 'a2p', 0)

        k = 2 * np.pi / exc.enei
        eps1 = self._eps1
        eps2 = self._eps2
        nvec = self._nvec

        def _matmul(a_val: Any, x_val: Any) -> Any:
            if isinstance(x_val, (int, float)) and x_val == 0:
                return 0
            if np.isscalar(a_val):
                return a_val * x_val
            return a_val[:, np.newaxis] * x_val if x_val.ndim > 1 else a_val * x_val

        # Eqs. (10, 11)
        phi = self._subtract(phi2, phi1)
        a = self._subtract(a2, a1)

        # Eq. (15)
        alpha = self._subtract(a2p, a1p) - \
            1j * k * self._subtract(
                self._outer(nvec, phi2, eps2),
                self._outer(nvec, phi1, eps1))

        # Eq. (18)
        De = self._subtract(_matmul(eps2, phi2p), _matmul(eps1, phi1p)) - \
            1j * k * self._subtract(
                self._inner(nvec, a2, eps2),
                self._inner(nvec, a1, eps1))

        # Expand arrays
        if isinstance(phi, (int, float)) and phi == 0:
            if isinstance(De, np.ndarray):
                phi = np.zeros_like(De)
            else:
                phi = np.zeros(n, dtype = complex)

        if isinstance(a, (int, float)) and a == 0:
            if isinstance(alpha, np.ndarray):
                a = np.zeros_like(alpha)
            else:
                a = np.zeros((n, 3), dtype = complex)

        return phi, a, De, alpha

    @staticmethod
    def _subtract(
            a: Any,
            b: Any) -> Any:

        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return a - b
        elif isinstance(a, np.ndarray):
            return a if (isinstance(b, (int, float)) and b == 0) else a - b
        elif isinstance(b, np.ndarray):
            return -b if (isinstance(a, (int, float)) and a == 0) else a - b
        else:
            return a - b

    def _afun(self,
            vec: np.ndarray) -> np.ndarray:

        # MATLAB: bemretiter/private/afun.m
        # Garcia de Abajo and Howie, PRB 65, 115418 (2002)
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces
        siz = int(vec.size / 2)

        # Split vector array (column-major reshape to match MATLAB)
        vec1 = vec[:siz].reshape(n, -1, order = 'F')
        vec2 = vec[siz:].reshape(n, -1, order = 'F')

        # Multiplication with Green functions
        G1_vec1 = self._G1 @ vec1
        G2_vec2 = self._G2 @ vec2

        # Pack into combined vector for unpack (column-major flatten)
        combined_g = np.empty(G1_vec1.size + G2_vec2.size, dtype = complex)
        combined_g[:G1_vec1.size] = G1_vec1.ravel(order = 'F')
        combined_g[G1_vec1.size:] = G2_vec2.ravel(order = 'F')
        Gsig1, Gh1, Gsig2, Gh2 = self._unpack(combined_g)

        H1_vec1 = self._H1 @ vec1
        H2_vec2 = self._H2 @ vec2
        combined_h = np.empty(H1_vec1.size + H2_vec2.size, dtype = complex)
        combined_h[:H1_vec1.size] = H1_vec1.ravel(order = 'F')
        combined_h[H1_vec1.size:] = H2_vec2.ravel(order = 'F')
        Hsig1, Hh1, Hsig2, Hh2 = self._unpack(combined_h)

        k = self._k
        nvec = self._nvec
        eps1 = self._eps1
        eps2 = self._eps2

        def _matmul_diag(a_val: Any, b: np.ndarray) -> np.ndarray:
            if np.isscalar(a_val):
                return a_val * b
            if b.ndim == 1:
                return a_val * b
            if a_val.ndim == 1:
                # Reshape a_val to (n, 1, 1, ...) to broadcast against trailing dims of b.
                return a_val.reshape(-1, *([1] * (b.ndim - 1))) * b
            return a_val * b

        # Eq. (10)
        phi = Gsig1 - Gsig2
        # Eq. (11)
        a = Gh1 - Gh2

        # Eq. (14)
        alpha = Hh1 - Hh2 - \
            1j * k * self._outer(nvec, _matmul_diag(eps1, Gsig1) - _matmul_diag(eps2, Gsig2))
        # Eq. (17)
        De = _matmul_diag(eps1, Hsig1) - _matmul_diag(eps2, Hsig2) - \
            1j * k * self._inner(nvec, _matmul_diag(eps1, Gh1) - _matmul_diag(eps2, Gh2))

        return self._pack(phi, a, De, alpha)

    def _mfun(self,
            vec: np.ndarray) -> np.ndarray:

        # MATLAB: bemretiter/private/mfun.m
        # Garcia de Abajo and Howie, PRB 65, 115418 (2002)

        # Unpack matrices
        phi, a, De, alpha = self._unpack(vec)

        sav = self._sav
        k = sav['k']
        nvec = sav['nvec']
        G1_lu = sav['G1_lu']
        G2_lu = sav['G2_lu']
        eps1 = sav['eps1']
        eps2 = sav['eps2']
        Sigma1 = sav['Sigma1']
        Delta_lu = sav['Delta_lu']
        Sigma_lu = sav['Sigma_lu']

        def matmul1(a_mat: np.ndarray, b: np.ndarray) -> np.ndarray:
            # Multiply (n, n) matrix with (n, ...) array, preserving trailing dims.
            if b.ndim == 1:
                return a_mat @ b
            n_rows = a_mat.shape[0] if not np.isscalar(a_mat) else b.shape[0]
            return (a_mat @ b.reshape(b.shape[0], -1)).reshape(n_rows, *b.shape[1:])

        def _ls(lu_piv, b):
            if b.ndim == 1:
                return lu_solve_dispatch(lu_piv, b)
            return lu_solve_dispatch(lu_piv, b.reshape(b.shape[0], -1)).reshape(b.shape)

        def matmul_eps(eps_mat: Any, b: np.ndarray) -> np.ndarray:
            # Apply diagonal eps (scalar / (n,n) diag matrix) to (n, ...) array.
            if np.isscalar(eps_mat):
                return eps_mat * b
            if b.ndim == 1:
                return eps_mat @ b
            return (eps_mat @ b.reshape(b.shape[0], -1)).reshape(b.shape)

        # Modify alpha and De
        # MATLAB: alpha = alpha - matmul1(Sigma1, a) + 1i*k*outer(nvec, eps1*phi)
        alpha = alpha - matmul1(Sigma1, a) + 1j * k * self._outer(nvec, matmul_eps(eps1, phi))
        De = De - matmul_eps(eps1, matmul1(Sigma1, phi)) + 1j * k * self._inner(nvec, matmul_eps(eps1, a))

        # Eq. (19)
        deps = eps1 - eps2
        inner_alpha = self._inner(nvec, _ls(Delta_lu, alpha))
        sig2 = _ls(Sigma_lu, De + 1j * k * matmul_eps(deps, inner_alpha))

        # Eq. (20)
        h2 = _ls(Delta_lu, 1j * k * self._outer(nvec, matmul_eps(deps, sig2)) + alpha)

        # Surface charges and currents
        sig1 = _ls(G1_lu, sig2 + phi)
        h1 = _ls(G1_lu, h2 + a)
        sig2_out = _ls(G2_lu, sig2)
        h2_out = _ls(G2_lu, h2)

        result = self._pack(sig1, h1, sig2_out, h2_out)
        return result

    def solve(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMRetIter']:

        # MATLAB: bemretiter/solve.m
        # Initialize BEM solver (if needed)
        self._init_matrices(exc.enei)

        # External excitation
        phi, a, De, alpha = self._excitation(exc)

        # Size of excitation arrays
        siz1 = phi.shape
        siz2 = a.shape

        # Pack everything to single vector
        b = self._pack(phi, a, De, alpha)

        if self._schur_active:
            # v1.5.0 Schur path: GMRES iterates on the reduced (core-only)
            # system.  Preconditioner is bypassed because _mfun was built
            # for the full 8N system; rebuilding it on the reduced 8M
            # block would require new G1/G2 LUs and is M5+ work.  For
            # cover-layer geometries the reduced system is well-
            # conditioned enough for unpreconditioned GMRES.
            op = self._schur_op
            b_eff = op.reduce_rhs(b)
            x_core, _ = self._iter_solve(None, b_eff, op._matvec, None)
            x = op.recover_full(x_core, b)
        else:
            # Function for matrix multiplication
            fa = self._afun
            fm = None
            if self.precond is not None:
                fm = self._mfun

            # v1.5.0 H-matrix LU preconditioner (agent alpha). Replaces fm
            # when active. The preconditioner is built once per (hmatrix
            # path, mode); we keep it cached on self for re-use across
            # enei sweeps.
            if self._hmatrix and self._hlu_mode != 'none':
                fm = self._build_hlu_preconditioner(b.shape[0])

            # Iterative solution
            x, self_updated = self._iter_solve(None, b, fa, fm)

        # Unpack and save solution vector
        sig1, h1, sig2, h2 = self._unpack(x)

        # Reshape surface charges and currents
        if len(siz1) > 1:
            sig1 = sig1.reshape(siz1)
            sig2 = sig2.reshape(siz1)
        if len(siz2) > 2:
            h1 = h1.reshape(siz2)
            h2 = h2.reshape(siz2)

        sig = CompStruct(self.p, exc.enei,
            sig1 = sig1, sig2 = sig2, h1 = h1, h2 = h2)

        return sig, self

    def _build_hlu_preconditioner(self,
            n_vec: int) -> Callable:

        # v1.5.0 agent alpha — H-matrix LU preconditioner.
        # The retarded iterative solver couples 8N variables (phi, a, phip,
        # ap) via the Garcia-de-Abajo / Howie [PRB 65, 115418] block
        # structure. The ``mfun`` derived in initprecond / mfun.m approximates
        # the inverse of this 8N x 8N system using only the LU factors of
        # G1, G2 and two reduced N x N matrices Sigma_lu and Delta_lu. We
        # reuse exactly that mfun, which means our preconditioner is
        # equivalent to v1.3 ``precond='hmat'`` -- but now triggered on the
        # H-matrix code path where v1.3 left it disabled.
        #
        # Implementation: call _init_precond once (this densifies G/H once
        # and builds the dense LU factors) and return the existing _mfun.
        # The HMatrixLUPreconditioner is used as the LU backend for the
        # individual G1, G2 factors via the lu_factor_dispatch hook.
        # Modes:
        #   'dense' / 'hlu_dense' / 'auto<5k' — densify G/H, dense LU
        #   'tree'  / 'hlu_tree'  / 'auto>=5k' — same path today; the
        #     HMatrixLUPreconditioner.tree backend is exposed standalone
        #     in mnpbem.bem.preconditioner for future integration into
        #     Sigma / Delta as well.
        if self._hlu_object is not None and self._hlu_object == (n_vec, self.enei):
            return self._mfun

        # Trigger the v1.3 dense initprecond path. This builds self._sav.
        self._init_precond(self.enei)
        self._hlu_object = (n_vec, self.enei)
        return self._mfun

    def __truediv__(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMRetIter']:

        # MATLAB: bemretiter/mldivide.m
        return self.solve(exc)

    def __mul__(self,
            sig: CompStruct) -> CompStruct:

        # MATLAB: bemretiter/mtimes.m
        pot1 = self.potential(sig, 1)
        pot2 = self.potential(sig, 2)

        return CompStruct(self.p, sig.enei,
            phi1 = pot1.phi1, phi1p = pot1.phi1p,
            a1 = pot1.a1, a1p = pot1.a1p,
            phi2 = pot2.phi2, phi2p = pot2.phi2p,
            a2 = pot2.a2, a2p = pot2.a2p)

    def field(self,
            sig: CompStruct,
            inout: int = 2) -> CompStruct:

        # MATLAB: bemretiter/field.m
        k = 2 * np.pi / sig.enei
        pot = self.potential(sig, inout)

        if hasattr(pot, 'phi1'):
            phi, phip, a, ap = pot.phi1, pot.phi1p, pot.a1, pot.a1p
        else:
            phi, phip, a, ap = pot.phi2, pot.phi2p, pot.a2, pot.a2p

        # Tangential directions via interpolation
        phi1_d, phi2_d = self.p.deriv(self.p.interp(phi))[:2]
        a1_d, a2_d, t1, t2 = self.p.deriv(self.p.interp(a))

        # Normal vector
        nvec = np.cross(t1, t2)
        h = msqrt(np.sum(nvec * nvec, axis = 1, keepdims = True))
        nvec = nvec / h

        # Tangential vectors
        tvec1 = np.cross(t2, nvec) / h
        tvec2 = -np.cross(t1, nvec) / h

        # Electric field
        e = 1j * k * a - \
            self._outer(nvec, phip) - \
            self._outer(tvec1, phi1_d) - \
            self._outer(tvec2, phi2_d)

        # Magnetic field
        def _matcross(v: np.ndarray, a_d: np.ndarray) -> np.ndarray:
            if a_d.ndim == 2:
                return np.cross(v, a_d)
            else:
                n_pts = v.shape[0]
                siz = a_d.shape[2]
                result = np.empty((n_pts, 3, siz), dtype = a_d.dtype)
                for s in range(siz):
                    result[:, :, s] = np.cross(v, a_d[:, :, s])
                return result

        h_field = _matcross(tvec1, a1_d) + _matcross(tvec2, a2_d) + _matcross(nvec, ap)

        return CompStruct(self.p, sig.enei, e = e, h = h_field)

    def potential(self,
            sig: CompStruct,
            inout: int = 2) -> CompStruct:

        # MATLAB: bemretiter/potential.m
        return self.g.potential(sig, inout)

    def clear(self) -> 'BEMRetIter':

        # MATLAB: bemretiter/clear.m
        self._G1 = None
        self._H1 = None
        self._G2 = None
        self._H2 = None
        self._sav = None
        return self

    def __call__(self,
            enei: float) -> 'BEMRetIter':

        return self._init_matrices(enei)

    def __repr__(self) -> str:
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces if hasattr(self.p, 'nfaces') else '?'
        status = 'enei={:.1f}nm'.format(self.enei) if self.enei is not None else 'not initialized'
        return 'BEMRetIter(p: {} faces, solver={}, {})'.format(n, self.solver, status)
