import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from ..utils.gpu import lu_factor_dispatch, lu_solve_dispatch, matmul_dispatch, to_host, is_cupy_array

from ..greenfun import CompGreenRetLayer, CompStruct


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
            self._G1_lu = lu_factor_dispatch(G1)
            G1i = lu_solve_dispatch(self._G1_lu, np.eye(G1.shape[0]))

            self._G2p_lu = lu_factor_dispatch(G2['p'])
            G2pi = lu_solve_dispatch(self._G2p_lu, np.eye(G2['p'].shape[0]))

            # Sigma matrices [Eq.(21)]
            Sigma1 = matmul_dispatch(H1, G1i)
            Sigma1e = matmul_dispatch(H1e, G1i)
            Sigma2p = matmul_dispatch(H2['p'], G2pi)

            # Auxiliary dielectric function matrices
            L1 = matmul_dispatch(G1e, G1i)
            L2p = matmul_dispatch(G2e['p'], G2pi)

            # Gamma matrix
            self._Gamma_lu = lu_factor_dispatch(Sigma1 - Sigma2p)
            Gamma = lu_solve_dispatch(self._Gamma_lu, np.eye(Sigma1.shape[0]))

            # Gammapar = ik*(L1-L2p)*Gamma .* (npar*npar')
            # Element-wise multiply with outer product of parallel normals
            npar_outer = npar @ npar.T  # (n, n)
            Gammapar = 1j * k * matmul_dispatch(L1 - L2p, Gamma) * npar_outer

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

            # Assemble 2x2 block matrix (2n x 2n) and LU factorize
            m_full = np.empty((2 * n, 2 * n), dtype = complex)
            m_full[:n, :n] = m11
            m_full[:n, n:] = m12
            m_full[n:, :n] = m21
            m_full[n:, n:] = m22
            self.m_full = None
            self.m_lu = lu_factor_dispatch(m_full)

        # Store all needed matrices
        self.G1i = G1i
        self.G2pi = G2pi
        self.G2 = G2
        self.G2e = G2e
        self.L1 = L1
        self.L2p = L2p
        self.Sigma1 = Sigma1
        self.Sigma1e = Sigma1e
        self.Gamma = Gamma

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
            if isinstance(m_lu, tuple) and len(m_lu) == 3 and m_lu[0] in ("cpu", "gpu"):
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
        self.enei = None
        return self

    def __call__(self,
            enei: float) -> 'BEMRetLayer':

        return self.init(enei)

    def __repr__(self) -> str:
        status = 'enei={:.1f}nm'.format(self.enei) if self.enei is not None else 'not initialized'
        n = self.p.nfaces if hasattr(self.p, 'nfaces') else self.p.n if hasattr(self.p, 'n') else '?'
        return 'BEMRetLayer(p: {} faces, {})'.format(n, status)
