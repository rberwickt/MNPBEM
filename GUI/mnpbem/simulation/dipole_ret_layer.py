import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from ..greenfun import CompStruct
from ..greenfun.greentab_layer import GreenTabLayer


class DipoleRetLayer(object):

    name = 'dipole'
    needs = {'sim': 'ret'}

    def __init__(self,
            pt: Any,
            layer: Any,
            dip: Optional[np.ndarray] = None,
            full: bool = False,
            medium: int = 1,
            pinfty: Optional[Any] = None,
            **options: Any) -> None:

        self.pt = pt
        self.layer = layer
        self.varargin = options

        self._init(dip, full, medium, pinfty, **options)

    def _init(self,
            dip: Optional[np.ndarray] = None,
            full: bool = False,
            medium: int = 1,
            pinfty: Optional[Any] = None,
            **options: Any) -> None:

        if dip is None:
            dip = np.eye(3)
            full = False

        dip = np.asarray(dip, dtype = float)

        if full:
            if dip.ndim == 2:
                dip = dip.reshape(dip.shape + (1,))
            self.dip = dip
        else:
            if dip.ndim == 1:
                dip = dip.reshape(1, -1)

            dip_reshaped = dip.T.reshape(1, dip.shape[1], dip.shape[0])
            self.dip = np.tile(dip_reshaped, (self.pt.n, 1, 1))

        self._medium = medium

        # Tabulated Green function for reflected contribution
        tab = options.get('tab', None)
        greentab = options.get('greentab', None)
        if greentab is not None:
            # Use pre-computed GreenTabLayer directly
            if hasattr(greentab, 'r') and greentab.r is not None:
                self.tab = greentab
            elif hasattr(greentab, 'tab'):
                self.tab = greentab.tab
            else:
                self.tab = GreenTabLayer(self.layer)
        elif tab is not None:
            self.tab = GreenTabLayer(self.layer, tab=tab)
        else:
            self.tab = GreenTabLayer(self.layer)

        # Spectrum for radiative decay rate
        from ..spectrum import SpectrumRetLayer
        self.spec = SpectrumRetLayer(pinfty, self.layer, medium=medium)
        self._pinfty = pinfty

    def field(self,
            p: Any,
            enei: float,
            key: Optional[str] = None) -> CompStruct:

        # MATLAB: @dipoleretlayer/field.m
        # E = ik0*A - grad(V), H = curl(A)

        pt = self.pt
        pos1 = p.pos if hasattr(p, 'pos') else p.pc.pos
        pos2 = pt.pos
        ndip = self.dip.shape[2]
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        exc = CompStruct(p, enei)
        exc.e = np.zeros((n1, 3, n2, ndip), dtype=complex)
        exc.h = np.zeros((n1, 3, n2, ndip), dtype=complex)

        # Direct contribution (skip if 'refl' mode)
        eps_vals = []
        k_vals = []
        for eps_func in p.eps:
            eps, k = eps_func(enei)
            eps_vals.append(eps)
            k_vals.append(k)

        eps_med = eps_vals[self._medium - 1]
        k_med = k_vals[self._medium - 1]

        if key != 'refl':
            e_direct, h_direct = self._dipolefield(pos1, pos2, self.dip, eps_med, k_med)
            exc.e += e_direct
            exc.h += h_direct

        # Reflected contribution using Green function derivatives
        k0 = 2 * np.pi / enei

        # Dielectric at dipole positions
        eps2 = np.array([eps_med] * n2, dtype=complex)

        G, F = self._greenderiv(enei, pos1, pos2)
        dip = self.dip                                      # (n2, 3, ndip)
        dip2 = dip / eps2[:, np.newaxis, np.newaxis]        # reduced dipole

        def fun(g, d):
            """g: (n1,n2), d: (n2,ndip) -> (n1,n2,ndip)"""
            if isinstance(g, (int, float)):
                return 0 if g == 0 else g * d[np.newaxis, :, :]
            return g[:, :, np.newaxis] * d[np.newaxis, :, :]

        # Vector potential A
        a1 = -1j * k0 * fun(G.get('p', 0), dip[:, 0, :])
        a2 = -1j * k0 * fun(G.get('p', 0), dip[:, 1, :])
        a3 = (-1j * k0 * fun(G.get('hh', 0), dip[:, 2, :])
              + fun(F[(1, 2)].get('hs', 0), dip2[:, 0, :])
              + fun(F[(1, 3)].get('hs', 0), dip2[:, 1, :])
              + fun(F[(1, 4)].get('hs', 0), dip2[:, 2, :]))

        # E += ik0 * A
        exc.e[:, 0, :, :] += 1j * k0 * a1
        exc.e[:, 1, :, :] += 1j * k0 * a2
        exc.e[:, 2, :, :] += 1j * k0 * a3

        # Derivatives of A for H-field: curl(A)
        # dA3/dy - dA2/dz
        a23 = (-1j * k0 * fun(F[(3, 1)].get('hh', 0), dip[:, 2, :])
              + fun(F[(3, 2)].get('hs', 0), dip2[:, 0, :])
              + fun(F[(3, 3)].get('hs', 0), dip2[:, 1, :])
              + fun(F[(3, 4)].get('hs', 0), dip2[:, 2, :]))
        a32 = -1j * k0 * fun(F[(4, 1)].get('p', 0), dip[:, 1, :])

        # dA1/dz - dA3/dx
        a31 = -1j * k0 * fun(F[(4, 1)].get('p', 0), dip[:, 0, :])
        a13 = (-1j * k0 * fun(F[(4, 1)].get('hh', 0), dip[:, 2, :])
              + fun(F[(4, 2)].get('hs', 0), dip2[:, 0, :])
              + fun(F[(4, 3)].get('hs', 0), dip2[:, 1, :])
              + fun(F[(4, 4)].get('hs', 0), dip2[:, 2, :]))

        # dA2/dx - dA1/dy
        a12 = -1j * k0 * fun(F[(2, 1)].get('p', 0), dip[:, 1, :])
        a21 = -1j * k0 * fun(F[(3, 1)].get('p', 0), dip[:, 0, :])

        # grad(V): scalar potential gradients
        phi1 = (fun(F[(2, 2)].get('ss', 0), dip2[:, 0, :])
              + fun(F[(2, 3)].get('ss', 0), dip2[:, 1, :])
              + fun(F[(2, 4)].get('ss', 0), dip2[:, 2, :])
              - 1j * k0 * fun(F[(2, 1)].get('sh', 0), dip[:, 2, :]))
        phi2 = (fun(F[(3, 2)].get('ss', 0), dip2[:, 0, :])
              + fun(F[(3, 3)].get('ss', 0), dip2[:, 1, :])
              + fun(F[(3, 4)].get('ss', 0), dip2[:, 2, :])
              - 1j * k0 * fun(F[(3, 1)].get('sh', 0), dip[:, 2, :]))
        phi3 = (fun(F[(4, 2)].get('ss', 0), dip2[:, 0, :])
              + fun(F[(4, 3)].get('ss', 0), dip2[:, 1, :])
              + fun(F[(4, 4)].get('ss', 0), dip2[:, 2, :])
              - 1j * k0 * fun(F[(1, 4)].get('sh', 0), dip[:, 2, :]))

        # E -= grad(V)
        exc.e[:, 0, :, :] -= phi1
        exc.e[:, 1, :, :] -= phi2
        exc.e[:, 2, :, :] -= phi3

        # H = curl(A) (sign convention matches MATLAB)
        exc.h[:, 0, :, :] -= (a23 - a32)
        exc.h[:, 1, :, :] -= (a31 - a13)
        exc.h[:, 2, :, :] -= (a12 - a21)

        return exc

    def _greenderiv(self,
            enei: float,
            pos1: np.ndarray,
            pos2: np.ndarray) -> Tuple[Dict, Dict]:
        """Green function and derivatives via finite differences.

        MATLAB: @dipoleretlayer/private/greenderiv.m

        Returns
        -------
        G_dict : dict
            Reflected Green function components, each (n1, n2).
        F : dict
            F[(i,j)][name] = (n1, n2) array of 2nd derivatives.
            Indices: 1=value, 2=x, 3=y, 4=z.
        """
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        # Handle self-interaction: perturb pos2 to avoid singular limit
        if n1 == n2 and np.allclose(pos1, pos2):
            pos2 = pos2.copy()
            pos2[:, 0] += self.layer.rmin

        # Lateral distances
        dx = pos1[:, 0:1] - pos2[:, 0:1].T  # (n1, n2)
        dy = pos1[:, 1:2] - pos2[:, 1:2].T  # (n1, n2)
        r = np.sqrt(dx ** 2 + dy ** 2)

        # z-components
        z1 = np.repeat(pos1[:, 2:3], n2, axis=1)  # (n1, n2)
        z2 = np.tile(pos2[:, 2:3].T, (n1, 1))     # (n1, n2)

        rmin = self.layer.rmin
        eta = 1e-6

        # Enforce minimum distance
        r = np.maximum(r, rmin)
        # Unit vectors
        xhat = dx / r
        yhat = dy / r

        # Round z-values
        z1_r, z2_r = self.layer.round_z(z1.ravel(), z2.ravel())
        r_flat = np.maximum(r.ravel(), rmin)

        # Baseline: G, Fr, Fz at (r, z1, z2)
        # MATLAB @dipoleretlayer/private/greenderiv.m uses interp(obj.tab, ...) on
        # tabulated Green functions; use eval_components so tabulation is used when
        # available (matches MATLAB behavior).
        G0, Fr0, Fz0 = self.tab.eval_components(enei, r_flat, z1_r, z2_r)

        # Perturbed in r: (r+eta, z1, z2)
        _, Fr_r, Fz_r = self.tab.eval_components(enei, r_flat + eta, z1_r, z2_r)

        # Perturbed in z2: (r, z1, z2+eta)
        G_z, Fr_z, Fz_z = self.tab.eval_components(enei, r_flat, z1_r, z2_r + eta)

        names = list(G0.keys())
        shape = (n1, n2)

        # Reshape Green function
        G_dict = {}
        for name in names:
            G_dict[name] = G0[name].reshape(shape)

        # Build derivative tensor F[(i,j)][name]
        F = {}
        for key in [(1, 2), (1, 3), (2, 1), (3, 1), (4, 1), (1, 4),
                     (2, 2), (2, 3), (3, 2), (3, 3),
                     (2, 4), (3, 4), (4, 2), (4, 3), (4, 4)]:
            F[key] = {}

        for name in names:
            Fr_val = Fr0[name].reshape(shape)
            Fz_val = Fz0[name].reshape(shape)

            # Finite difference derivatives
            Frr = (Fr_r[name].reshape(shape) - Fr_val) / eta
            Fr1 = (Fz_r[name].reshape(shape) - Fz_val) / eta
            F2 = (G_z[name].reshape(shape) - G_dict[name]) / eta
            Fr2 = (Fr_z[name].reshape(shape) - Fr_val) / eta
            F12 = (Fz_z[name].reshape(shape) - Fz_val) / eta

            # 1st derivatives (Cartesian)
            F[(1, 2)][name] = -Fr_val * xhat       # dG/dx'
            F[(1, 3)][name] = -Fr_val * yhat       # dG/dy'
            F[(2, 1)][name] = Fr_val * xhat        # dG/dx
            F[(3, 1)][name] = Fr_val * yhat        # dG/dy
            F[(4, 1)][name] = Fz_val               # dG/dz1
            F[(1, 4)][name] = F2                   # dG/dz2

            # 2nd derivatives
            F[(2, 2)][name] = -Fr_val * yhat ** 2 / r - Frr * xhat ** 2
            F[(2, 3)][name] = -Fr_val * xhat * yhat / r - Frr * xhat * yhat
            F[(3, 3)][name] = -Fr_val * xhat ** 2 / r - Frr * yhat ** 2
            F[(3, 2)][name] = F[(2, 3)][name]      # symmetric

            # Mixed z derivatives
            F[(2, 4)][name] = Fr2 * xhat            # d2G/dx dz2
            F[(3, 4)][name] = Fr2 * yhat            # d2G/dy dz2
            F[(4, 2)][name] = -Fr1 * xhat           # d2G/dz1 dx'
            F[(4, 3)][name] = -Fr1 * yhat           # d2G/dz1 dy'
            F[(4, 4)][name] = F12                   # d2G/dz1 dz2

        return G_dict, F

    def _dipolefield(self,
            pos1: np.ndarray,
            pos2: np.ndarray,
            dip: np.ndarray,
            eps: complex,
            k: complex) -> Tuple[np.ndarray, np.ndarray]:

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        x = pos1[:, 0:1] - pos2[:, 0].T
        y = pos1[:, 1:2] - pos2[:, 1].T
        z = pos1[:, 2:3] - pos2[:, 2].T
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        r = np.maximum(r, np.finfo(float).eps)
        x, y, z = x / r, y / r, z / r

        G = np.exp(1j * k * r) / r

        ndip = dip.shape[2]

        e = np.zeros((n1, 3, n2, ndip), dtype = complex)
        h = np.zeros((n1, 3, n2, ndip), dtype = complex)

        for i in range(ndip):
            dx = np.tile(dip[:, 0, i], (n1, 1))
            dy = np.tile(dip[:, 1, i], (n1, 1))
            dz = np.tile(dip[:, 2, i], (n1, 1))

            inner = x * dx + y * dy + z * dz

            fac = k ** 2 * G * (1 - 1 / (1j * k * r)) / np.sqrt(eps)
            h[:, 0, :, i] = fac * (y * dz - z * dy)
            h[:, 1, :, i] = fac * (z * dx - x * dz)
            h[:, 2, :, i] = fac * (x * dy - y * dx)

            fac1 = k ** 2 * G / eps
            fac2 = G * (1 / r ** 2 - 1j * k / r) / eps
            e[:, 0, :, i] = fac1 * (dx - inner * x) + fac2 * (3 * inner * x - dx)
            e[:, 1, :, i] = fac1 * (dy - inner * y) + fac2 * (3 * inner * y - dy)
            e[:, 2, :, i] = fac1 * (dz - inner * z) + fac2 * (3 * inner * z - dz)

        return e, h

    def potential(self,
            p: Any,
            enei: float) -> CompStruct:
        """Potential of dipole excitation for use in BEM.

        MATLAB: @dipoleretlayer/potential.m
        """
        pt = self.pt
        pos1 = p.pos if hasattr(p, 'pos') else p.pc.pos
        nvec = p.nvec if hasattr(p, 'nvec') else p.pc.nvec
        n1 = pos1.shape[0]
        n2 = pt.pos.shape[0]
        ndip = self.dip.shape[2]

        # Direct contribution
        eps_vals = []
        k_vals = []
        for eps_func in p.eps:
            eps, k = eps_func(enei)
            eps_vals.append(eps)
            k_vals.append(k)

        eps_med = eps_vals[self._medium - 1]
        k_med = k_vals[self._medium - 1]

        exc = CompStruct(p, enei)
        # Initialize with zeros
        exc.phi1 = np.zeros((n1, n2, ndip), dtype = complex)
        exc.phi1p = np.zeros((n1, n2, ndip), dtype = complex)
        exc.phi2 = np.zeros((n1, n2, ndip), dtype = complex)
        exc.phi2p = np.zeros((n1, n2, ndip), dtype = complex)
        exc.a1 = np.zeros((n1, 3, n2, ndip), dtype = complex)
        exc.a1p = np.zeros((n1, 3, n2, ndip), dtype = complex)
        exc.a2 = np.zeros((n1, 3, n2, ndip), dtype = complex)
        exc.a2p = np.zeros((n1, 3, n2, ndip), dtype = complex)

        # Direct potentials assigned to outside boundary (dipole in medium 1)
        phi_d, phip_d, a_d, ap_d = self._pot(
            pos1, pt.pos, nvec, self.dip, eps_med, k_med)
        exc.phi2 = phi_d
        exc.phi2p = phip_d
        exc.a2 = a_d
        exc.a2p = ap_d

        # Reflected contribution using Green function derivatives
        k0 = 2 * np.pi / enei

        eps2_arr = np.array([eps_med] * n2, dtype=complex)
        G, F = self._greenderiv(enei, pos1, pt.pos)
        dip = self.dip
        dip2 = dip / eps2_arr[:, np.newaxis, np.newaxis]

        def fun(g, d):
            """g: (n1,n2), d: (n2,ndip) -> (n1,n2,ndip)"""
            if isinstance(g, (int, float)):
                return 0 if g == 0 else g * d[np.newaxis, :, :]
            return g[:, :, np.newaxis] * d[np.newaxis, :, :]

        # Vector potential: a
        a1 = -1j * k0 * fun(G.get('p', 0), dip[:, 0, :])
        a2 = -1j * k0 * fun(G.get('p', 0), dip[:, 1, :])
        a3 = (-1j * k0 * fun(G.get('hh', 0), dip[:, 2, :])
              + fun(F[(1, 2)].get('hs', 0), dip2[:, 0, :])
              + fun(F[(1, 3)].get('hs', 0), dip2[:, 1, :])
              + fun(F[(1, 4)].get('hs', 0), dip2[:, 2, :]))

        # Scalar potential: phi
        phi_r = (fun(F[(1, 2)].get('ss', 0), dip2[:, 0, :])
               + fun(F[(1, 3)].get('ss', 0), dip2[:, 1, :])
               + fun(F[(1, 4)].get('ss', 0), dip2[:, 2, :])
               - 1j * k0 * fun(G.get('sh', 0), dip[:, 2, :]))

        # Add reflected to a2, phi2
        exc.a2[:, 0, :, :] += a1
        exc.a2[:, 1, :, :] += a2
        exc.a2[:, 2, :, :] += a3
        exc.phi2 += phi_r

        # Surface derivatives: nvec dot grad
        def deriv(comp_name, col_idx):
            """Normal derivative: nvec . [F{2,j}, F{3,j}, F{4,j}]"""
            f2 = F[(2, col_idx)].get(comp_name, None)
            f3 = F[(3, col_idx)].get(comp_name, None)
            f4 = F[(4, col_idx)].get(comp_name, None)
            if f2 is None and f3 is None and f4 is None:
                return 0
            result = np.zeros((n1, n2, 1), dtype=complex)
            if f2 is not None:
                result += nvec[:, 0:1, np.newaxis] * f2[:, :, np.newaxis]
            if f3 is not None:
                result += nvec[:, 1:2, np.newaxis] * f3[:, :, np.newaxis]
            if f4 is not None:
                result += nvec[:, 2:3, np.newaxis] * f4[:, :, np.newaxis]
            return result

        def fun3(g, d):
            """g: (n1,n2,1) or 0, d: (n2,ndip) -> (n1,n2,ndip)"""
            if isinstance(g, (int, float)) and g == 0:
                return 0
            return g * d[np.newaxis, :, :]

        # Surface derivative of vector potential: ap
        # MATLAB potential.m L75 uses dip(:, 2, :) (column 2 = y) for the deriv('hh', 1)
        # term — this is inconsistent with a3 (which uses column 3 = z) and appears to
        # be a MATLAB typo, but we replicate it exactly to match MATLAB reference output.
        a1p = -1j * k0 * fun3(deriv('p', 1), dip[:, 0, :])
        a2p = -1j * k0 * fun3(deriv('p', 1), dip[:, 1, :])
        a3p = (-1j * k0 * fun3(deriv('hh', 1), dip[:, 1, :])
              + fun3(deriv('hs', 2), dip2[:, 0, :])
              + fun3(deriv('hs', 3), dip2[:, 1, :])
              + fun3(deriv('hs', 4), dip2[:, 2, :]))

        # Surface derivative of scalar potential: phip
        phip_r = (fun3(deriv('ss', 2), dip2[:, 0, :])
                + fun3(deriv('ss', 3), dip2[:, 1, :])
                + fun3(deriv('ss', 4), dip2[:, 2, :])
                - 1j * k0 * fun3(deriv('sh', 1), dip[:, 2, :]))

        exc.a2p[:, 0, :, :] += a1p
        exc.a2p[:, 1, :, :] += a2p
        exc.a2p[:, 2, :, :] += a3p
        exc.phi2p += phip_r

        # Reshape from (nfaces, n_pts, ndip) to (nfaces, n_pts*ndip) for BEM solver
        n = n1
        npol = n2 * ndip
        exc.phi1 = exc.phi1.reshape(n, npol)
        exc.phi1p = exc.phi1p.reshape(n, npol)
        exc.phi2 = exc.phi2.reshape(n, npol)
        exc.phi2p = exc.phi2p.reshape(n, npol)
        exc.a1 = exc.a1.reshape(n, 3, npol)
        exc.a1p = exc.a1p.reshape(n, 3, npol)
        exc.a2 = exc.a2.reshape(n, 3, npol)
        exc.a2p = exc.a2p.reshape(n, 3, npol)

        return exc

    def _pot(self,
            pos1: np.ndarray,
            pos2: np.ndarray,
            nvec: np.ndarray,
            dip: np.ndarray,
            eps: complex,
            k: complex) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        k0 = k / np.sqrt(eps)

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        x = pos1[:, 0:1] - pos2[:, 0].T
        y = pos1[:, 1:2] - pos2[:, 1].T
        z = pos1[:, 2:3] - pos2[:, 2].T
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        r = np.maximum(r, np.finfo(float).eps)
        x, y, z = x / r, y / r, z / r

        G = np.exp(1j * k * r) / r
        F = (1j * k - 1 / r) * G

        nx = np.tile(nvec[:, 0:1], (1, n2))
        ny = np.tile(nvec[:, 1:2], (1, n2))
        nz = np.tile(nvec[:, 2:3], (1, n2))
        en = nx * x + ny * y + nz * z

        ndip = dip.shape[2]
        phi = np.zeros((n1, n2, ndip), dtype = complex)
        phip = np.zeros((n1, n2, ndip), dtype = complex)
        a = np.zeros((n1, 3, n2, ndip), dtype = complex)
        ap = np.zeros((n1, 3, n2, ndip), dtype = complex)

        for i in range(ndip):
            dx = np.tile(dip[:, 0, i], (n1, 1))
            dy = np.tile(dip[:, 1, i], (n1, 1))
            dz = np.tile(dip[:, 2, i], (n1, 1))

            ep = x * dx + y * dy + z * dz
            np_dot = nx * dx + ny * dy + nz * dz

            phi[:, :, i] = -ep * F / eps
            phip[:, :, i] = (
                (np_dot - 3 * en * ep) / r ** 2 * (1 - 1j * k * r) * G / eps
                + k ** 2 * ep * en * G / eps)

            a[:, 0, :, i] = -1j * k0 * dx * G
            a[:, 1, :, i] = -1j * k0 * dy * G
            a[:, 2, :, i] = -1j * k0 * dz * G

            ap[:, 0, :, i] = -1j * k0 * dx * en * F
            ap[:, 1, :, i] = -1j * k0 * dy * en * F
            ap[:, 2, :, i] = -1j * k0 * dz * en * F

        return phi, phip, a, ap

    def decayrate(self,
            sig: CompStruct) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        p, enei = sig.p, sig.enei

        npt = self.pt.n
        ndip = self.dip.shape[2]

        from ..greenfun import CompGreenRetLayer
        # Convert greentab -> greentab_obj for CompGreenRetLayer
        g_opts = dict(self.varargin)
        if 'greentab' in g_opts:
            gt_val = g_opts.pop('greentab')
            if hasattr(gt_val, 'tab'):
                g_opts['greentab_obj'] = gt_val.tab
            elif hasattr(gt_val, 'r'):
                g_opts['greentab_obj'] = gt_val
        g = CompGreenRetLayer(self.pt, sig.p, self.layer, **g_opts)

        # MATLAB: e = field(g, sig) + field(obj, obj.dip.pt, sig.enei, 'refl')
        # A5 fix: materialize cupy sig members on host before invoking Green
        # function so numpy matmul does not raise on a cupy operand.
        for _name in ('sig1', 'sig2', 'h1', 'h2'):
            if hasattr(sig, _name):
                _val = getattr(sig, _name)
                if hasattr(_val, 'get') and not isinstance(_val, np.ndarray):
                    setattr(sig, _name, _val.get())
        field_struct = g.field(sig)
        _e_raw = field_struct.e
        e_field = (_e_raw.get() if (hasattr(_e_raw, 'get')
            and not isinstance(_e_raw, np.ndarray)) else np.asarray(_e_raw))  # (npt, 3, npt*ndip)
        # Reshape to (npt, 3, npt, ndip) to match reflected field shape
        if e_field.ndim == 3:
            e_field = e_field.reshape(npt, 3, npt, ndip)
        refl_struct = self.field(self.pt, enei, key = 'refl')
        e = e_field + refl_struct.e

        k0 = 2 * np.pi / sig.enei
        gamma = 4 / 3 * k0 ** 3
        tot = np.zeros((npt, ndip))
        rad0 = np.zeros((npt, ndip))

        # MATLAB: sca = scattering(obj.spec.farfield(sig) + farfield(obj, obj.spec, sig.enei))
        from .retarded_utils import scattering as sca_func
        ff_spec = self.spec.farfield(sig)
        ff_dip = self.farfield(self.spec, enei)

        # Ensure matching shapes
        e_spec = ff_spec.e
        h_spec = ff_spec.h
        e_dip = ff_dip.e
        h_dip = ff_dip.h

        # Reshape to common dimensionality
        npol = npt * ndip
        if e_spec.ndim == 2:
            e_spec = e_spec.reshape(e_spec.shape[0], 3, 1)
        if h_spec.ndim == 2:
            h_spec = h_spec.reshape(h_spec.shape[0], 3, 1)
        if e_dip.ndim == 4:
            e_dip = e_dip.reshape(e_dip.shape[0], 3, npol)
        if h_dip.ndim == 4:
            h_dip = h_dip.reshape(h_dip.shape[0], 3, npol)
        if e_spec.ndim == 3 and e_dip.ndim == 3 and e_spec.shape[2] != e_dip.shape[2]:
            if e_spec.shape[2] == 1:
                e_spec = np.tile(e_spec, (1, 1, npol))
                h_spec = np.tile(h_spec, (1, 1, npol))

        ff_e = e_spec + e_dip
        ff_h = h_spec + h_dip
        ff_combined = CompStruct(ff_spec.p, enei, e = ff_e, h = ff_h)
        sca = sca_func(ff_combined)
        if np.isscalar(sca):
            sca = np.array([sca])
        rad = np.reshape(sca, (npt, ndip)) / (2 * np.pi * k0)

        # Reshape e from (npt, 3, npol) to (npt, 3, npt, ndip) if needed
        if e.ndim == 3:
            e = e.reshape(npt, 3, npt, ndip)
        elif e.ndim == 2:
            e = e.reshape(npt, 3, npt, ndip)

        for ipos in range(npt):
            for idip in range(ndip):
                dip = self.dip[ipos, :, idip]
                nb = np.sqrt(self.pt.eps1(sig.enei)[ipos])

                # MATLAB @dipoleretlayer/decayrate.m:52-57 —
                # tot = 1 + imag(e * dip') / (0.5*nb*gamma)   (no rad term!)
                # rad = rad / (0.5*nb*gamma)
                # MATLAB stores complex if nb is complex (absorbing medium).
                e_i = e[ipos, :, ipos, idip]
                tot[ipos, idip] = 1 + np.imag(e_i @ dip) / (0.5 * nb * gamma)
                rad[ipos, idip] = rad[ipos, idip] / (0.5 * nb * gamma)
                rad0[ipos, idip] = nb * gamma

        return tot, rad, rad0

    def decayrate0(self,
            enei: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # MATLAB: dipoleretlayer/decayrate0.m
        # Total and radiative decay rate for oscillating dipole above
        # layer structure (w/o nanoparticle) in units of the free-space decay rate

        # TRUE if all dielectric functions real
        ir = all(np.isreal(eps_func(enei)[0]) for eps_func in self.layer.eps)

        # Reflected electric field
        e = self.field(self.pt, enei, key = 'refl').e

        k0 = 2 * np.pi / enei
        # Wigner-Weisskopf decay rate in free space
        gamma = 4 / 3 * k0 ** 3

        npt = self.pt.n
        ndip = self.dip.shape[2]
        tot = np.zeros((npt, ndip))
        rad0 = np.zeros((npt, ndip))

        # Scattering cross section from far-field
        from .retarded_utils import scattering as sca_func
        ff = self.farfield(self.spec, enei)
        sca = sca_func(ff)
        rad = sca.reshape((npt, ndip)) / (2 * np.pi * k0)

        for ipos in range(npt):
            for idip in range(ndip):
                dip = self.dip[ipos, :, idip]

                nb = np.sqrt(self.pt.eps1(enei)[ipos])
                if np.imag(nb) != 0:
                    import warnings
                    warnings.warn('Dipole embedded in medium with complex dielectric function')

                # Radiative decay rate in units of free-space decay rate
                rad[ipos, idip] = rad[ipos, idip] / (0.5 * nb * gamma)
                # Free-space decay rate
                rad0[ipos, idip] = np.real(nb * gamma)

                # Total decay rate
                if ir:
                    tot[ipos, idip] = rad[ipos, idip]
                else:
                    tot[ipos, idip] = 1 + np.imag(
                        np.squeeze(e[ipos, :, ipos, idip]) @ dip) / (0.5 * nb * gamma)

        return tot, rad, rad0

    def farfield(self,
            spec: Any,
            enei: float) -> CompStruct:
        """Far-field of dipole above layer structure.

        MATLAB: @dipoleretlayer/farfield.m
        Uses layer.reflection() + finite differences for proper treatment
        of surface charge/current components (p, ss, hh, hs, sh).
        """
        from ..spectrum.spectrum_ret import SpectrumRet

        nvec_dir = spec.pinfty.nvec if hasattr(spec.pinfty, 'nvec') else spec.nvec
        layer = self.layer
        pt = self.pt
        pos_dip = pt.pos.copy()
        dip_moments = self.dip.copy()

        # Reduced dipole: dip / eps_embedding
        eps_at_dip = pt.eps1(enei)
        dip2 = dip_moments / eps_at_dip[:, np.newaxis, np.newaxis]

        k0 = 2 * np.pi / enei
        k_vals = np.array([eps_func(enei)[1] for eps_func in layer.eps])
        medium = [layer.ind[0], layer.ind[-1]]

        ind1 = nvec_dir[:, 2] >= 0  # upper hemisphere
        ind2 = nvec_dir[:, 2] < 0   # lower hemisphere

        ndir = nvec_dir.shape[0]
        npt = dip_moments.shape[0]
        ndip = dip_moments.shape[2]
        npol = npt * ndip

        # Finite-difference displaced positions
        # MATLAB: pos = cat(3, pos, pos+eta*ex, pos+eta*ey, pos+eta*ez)
        eta = 1e-6
        pos_base = pos_dip  # (npt, 3)
        pos_all = np.zeros((npt * 4, 3))
        pos_all[:npt] = pos_base
        pos_all[npt:2*npt] = pos_base + np.array([[eta, 0, 0]])
        pos_all[2*npt:3*npt] = pos_base + np.array([[0, eta, 0]])
        pos_all[3*npt:4*npt] = pos_base + np.array([[0, 0, eta]])

        # Position structures for reflection
        z2 = pos_all[:, 2]
        ind2_layer, _ = layer.indlayer(z2)
        pos_up = {'z1': np.atleast_1d(layer.z[0]), 'ind1': np.atleast_1d(1),
                  'z2': z2, 'ind2': ind2_layer}
        pos_down = {'z1': np.atleast_1d(layer.z[-1]),
                    'ind1': np.atleast_1d(layer.n + 1),
                    'z2': z2, 'ind2': ind2_layer}

        # Which dipoles are connected to the layer
        # MATLAB: ind = any(bsxfun(@eq, pt.expand(pt.inout(:,end)), layer.ind), 2)
        ind_layer = np.ones(npt, dtype=bool)

        # Allocate reflected vector potential
        a = np.zeros((ndir, 3, npol), dtype=complex)

        # ---- Upper hemisphere (positive propagation) ----
        if np.imag(k_vals[0]) == 0 and np.any(ind1):
            for idir in np.where(ind1)[0]:
                kpar = np.real(k_vals[0]) * np.sqrt(1 - nvec_dir[idir, 2] ** 2)

                r, _ = layer.reflection(enei, kpar, pos_up)

                # Distance for phase: pos_xy . dir_xy + z_layer * dir_z
                xy_pos = pos_all[np.tile(ind_layer, 4), :2]
                dist = xy_pos @ nvec_dir[idir, :2] + layer.z[0] * nvec_dir[idir, 2]
                phase = np.exp(-1j * k_vals[0] * dist)  # (npt*4,)

                # Apply phase and take finite differences
                r_phased = {}
                for name in r:
                    rv = np.asarray(r[name]).ravel()
                    rv_p = rv * phase  # (npt*4,)
                    rv_4 = rv_p.reshape(4, npt).T  # (npt, 4)
                    # Finite difference: columns 1,2,3 = derivative wrt x,y,z
                    rv_4[:, 1:] = (rv_4[:, 1:] - rv_4[:, 0:1]) / eta
                    r_phased[name] = rv_4

                # Vector potential: a(idir, :, :) = -ik0*r.p*dip + r.hs*dip2 terms
                rp = r_phased.get('p', np.zeros((npt, 4)))
                rhh = r_phased.get('hh', np.zeros((npt, 4)))
                rhs = r_phased.get('hs', np.zeros((npt, 4)))

                for idip in range(ndip):
                    ipol = np.arange(npt) * ndip + idip
                    d = dip_moments[ind_layer, :, idip]  # (npt, 3)
                    d2 = dip2[ind_layer, :, idip]  # (npt, 3)

                    a[idir, 0, ipol] = -1j * k0 * (rp[:, 0] * d[:, 0])
                    a[idir, 1, ipol] = -1j * k0 * (rp[:, 0] * d[:, 1])
                    a[idir, 2, ipol] = (-1j * k0 * (rhh[:, 0] * d[:, 2])
                                        + rhs[:, 1] * d2[:, 0]
                                        + rhs[:, 2] * d2[:, 1]
                                        + rhs[:, 3] * d2[:, 2])

        # ---- Lower hemisphere (negative propagation) ----
        if np.imag(k_vals[-1]) == 0 and np.any(ind2):
            for idir in np.where(ind2)[0]:
                kpar = np.real(k_vals[-1]) * np.sqrt(1 - nvec_dir[idir, 2] ** 2)

                r, _ = layer.reflection(enei, kpar, pos_down)

                xy_pos = pos_all[np.tile(ind_layer, 4), :2]
                dist = xy_pos @ nvec_dir[idir, :2] + layer.z[-1] * nvec_dir[idir, 2]
                phase = np.exp(-1j * k_vals[-1] * dist)

                r_phased = {}
                for name in r:
                    rv = np.asarray(r[name]).ravel()
                    rv_p = rv * phase
                    rv_4 = rv_p.reshape(4, npt).T
                    rv_4[:, 1:] = (rv_4[:, 1:] - rv_4[:, 0:1]) / eta
                    r_phased[name] = rv_4

                rp = r_phased.get('p', np.zeros((npt, 4)))
                rhh = r_phased.get('hh', np.zeros((npt, 4)))
                rhs = r_phased.get('hs', np.zeros((npt, 4)))

                for idip in range(ndip):
                    ipol = np.arange(npt) * ndip + idip
                    d = dip_moments[ind_layer, :, idip]
                    d2 = dip2[ind_layer, :, idip]

                    a[idir, 0, ipol] = -1j * k0 * (rp[:, 0] * d[:, 0])
                    a[idir, 1, ipol] = -1j * k0 * (rp[:, 0] * d[:, 1])
                    a[idir, 2, ipol] = (-1j * k0 * (rhh[:, 0] * d[:, 2])
                                        + rhs[:, 1] * d2[:, 0]
                                        + rhs[:, 2] * d2[:, 1]
                                        + rhs[:, 3] * d2[:, 2])

        # Electric field from vector potential
        e = 1j * k0 * a  # (ndir, 3, npol)

        # ---- Direct (free-space) dipole fields ----
        # MATLAB: farfield(dip, spectrumret(pinfty, 'medium', medium(1)), enei)
        #       + farfield(dip, spectrumret(pinfty, 'medium', medium(2)), enei)
        # MATLAB MNPBEM bug-compat: when spec.pinfty is a struct (i.e. supplied
        # by the user as a plain nvec wrapper), MATLAB's spectrumret(...) hits
        # the isstruct branch and falls back to trisphere(256, 2). Subsequent
        # indexing e(ind1, :, :) += field1.e(ind1, :, :) then uses linear
        # indices, picking the FIRST ndir entries from the 256-direction sphere.
        # When spec.pinfty is the SpectrumRetLayer-internal default (a real
        # particle from select+vertcat), MATLAB takes the else-branch and uses
        # the supplied pinfty directly. We mirror this by checking the flag
        # SpectrumRetLayer set during init.
        from .dipole_ret import DipoleRet
        dip_free = DipoleRet(pt, dip=None)
        dip_free.dip = dip_moments

        use_fallback = bool(getattr(spec, '_user_pinfty', True))
        if use_fallback:
            spec1 = SpectrumRet(pinfty=None, medium=medium[0])
            spec2 = SpectrumRet(pinfty=None, medium=medium[1])
        else:
            spec1 = SpectrumRet(spec.pinfty, medium=medium[0])
            spec2 = SpectrumRet(spec.pinfty, medium=medium[1])
        ff1 = dip_free.farfield(spec1, enei)
        ff2 = dip_free.farfield(spec2, enei)

        # Reshape free fields to (n_default, 3, npol) then take first ndir rows
        n_default = ff1.e.shape[0]
        e1_full = ff1.e.reshape(n_default, 3, npol) + ff2.e.reshape(n_default, 3, npol)
        e1 = e1_full[:ndir, :, :].copy()

        # Determine which medium the dipoles are in
        inout_vals = pt.inout  # medium indices for each dipole group
        # For single dipole point in upper medium: set lower medium contribution to 0
        e_up = e1.copy()
        e_down = e1.copy()
        # Simplified: for dipoles in medium[0], keep e_up; for medium[1], keep e_down
        # For a single dipole in air above glass, medium[0]=air, dipole is in air
        # So e_up = full free field, e_down = 0 for this dipole
        for ig in range(len(pt.p)):
            io = int(inout_vals[ig])
            ipol_start = sum(pt.p[k].n for k in range(ig)) * ndip
            ipol_end = ipol_start + pt.p[ig].n * ndip
            if io != medium[0]:
                e_up[:, :, ipol_start:ipol_end] = 0
            if io != medium[1]:
                e_down[:, :, ipol_start:ipol_end] = 0

        # Add direct fields to reflected
        e[ind1, :, :] += e_up[ind1, :, :]
        e[ind2, :, :] += e_down[ind2, :, :]

        # Make electric field transversal: e = e - dir * (dir . e)
        dir_dot_e = np.einsum('ij,ijk->ik', nvec_dir, e)  # (ndir, npol)
        e -= nvec_dir[:, :, np.newaxis] * dir_dot_e[:, np.newaxis, :]

        # Magnetic field: h = (k/k0) * cross(dir, e) per hemisphere
        h = np.zeros_like(e)
        dir_3d = nvec_dir[:, :, np.newaxis]
        dir_3d = np.broadcast_to(dir_3d, e.shape)
        if np.any(ind1):
            h[ind1] = k_vals[0] / k0 * np.cross(dir_3d[ind1], e[ind1], axis=1)
        if np.any(ind2):
            h[ind2] = k_vals[-1] / k0 * np.cross(dir_3d[ind2], e[ind2], axis=1)

        # Reshape to (ndir, 3, npt, ndip)
        e = e.reshape(ndir, 3, npt, ndip)
        h = h.reshape(ndir, 3, npt, ndip)

        field = CompStruct(spec.pinfty, enei, e=e, h=h)
        return field

    def __call__(self,
            p: Any,
            enei: float) -> CompStruct:

        return self.potential(p, enei)

    def __repr__(self) -> str:
        return 'DipoleRetLayer(npt={}, ndip={})'.format(
            self.pt.n, self.dip.shape[2])
