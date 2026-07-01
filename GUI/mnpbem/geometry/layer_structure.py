"""
Dielectric layer structure for stratified media.

MATLAB: Particles/@layerstructure/

Implements dielectric layer structures for BEM simulations with
substrates and multilayer systems. Provides Fresnel coefficients,
reflected Green functions, and BEM equation solvers for layer systems.

Reference:
    M. Paulus et al., PRE 62, 5797 (2000)
    Waxenegger et al., Comp. Phys. Commun. 193, 138 (2015)
"""

import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy.special import jv as besselj
from scipy.special import hankel1
from scipy.integrate import solve_ivp, quad_vec

from ..utils.matlab_compat import (
    mcos, msin, msqrt, mlinspace, mlog10, mtanh, matan,
    m_exp_c, m_sqrt_c,
)
from ..utils.matlab_ode45 import matlab_ode45
from ..utils.gpu import layer_gpu_active, get_layer_xp


class LayerStructure(object):
    """
    Dielectric layer structure for stratified media.

    MATLAB: @layerstructure

    The outer surface normals of the layers must point upwards.
    Geometry:

                  eps[0]
        --------------------------  z[0]
                  eps[1]
        --------------------------  z[1]
                   ...
        --------------------------  z[-1]
                  eps[-1]

    Properties
    ----------
    eps : list
        Dielectric functions for each layer
    z : ndarray
        z-positions of layer interfaces
    ind : ndarray
        Index to table of dielectric functions
    ztol : float
        Tolerance for detecting points in layer (default: 2e-2)
    rmin : float
        Minimum radial distance for Green function (default: 1e-2)
    zmin : float
        Minimum distance to layer for Green function (default: 1e-2)
    semi : float
        Imaginary part of semiellipse for complex integration (default: 0.1)
    ratio : float
        z:r ratio which determines integration path (default: 2)

    Methods
    -------
    fresnel(enei, kpar, pos)
        Fresnel reflection/transmission coefficients for potentials
    efresnel(pol, dir, enei)
        Reflected and transmitted electric fields for plane wave
    green(enei, r, z1, z2)
        Reflected Green function via complex integration
    reflection(enei, kpar, pos)
        Reflection coefficients for surface charges and currents
    bemsolve(enei, kpar)
        Solve BEM equations for layer structure
    indlayer(z)
        Find layer index for given z-values
    mindist(z)
        Minimal distance of z-values to layer boundaries
    round_z(*z_args)
        Round z-values to achieve minimal distance to layers
    tabspace(...)
        Generate grids for tabulated Green functions

    Examples
    --------
    >>> from mnpbem import EpsConst
    >>> from mnpbem.geometry import LayerStructure
    >>>
    >>> # Single interface (substrate)
    >>> eps_tab = [EpsConst(1.0), EpsConst(2.25)]
    >>> layer = LayerStructure(eps_tab, [1, 2], [0.0])
    """

    def __init__(self,
            epstab: list,
            ind: Union[list, np.ndarray],
            z: Union[list, np.ndarray],
            **options: Any) -> None:

        ind = np.asarray(ind, dtype = int)
        z = np.asarray(z, dtype = float)

        # eps is indexed from epstab using ind (MATLAB 1-indexed -> Python 0-indexed)
        self.eps = [epstab[i - 1] for i in ind]
        self.ind = ind
        self.z = z

        # Default options for complex integration
        self.ztol = options.get('ztol', 2e-2)
        self.rmin = options.get('rmin', 1e-2)
        self.zmin = options.get('zmin', 1e-2)
        self.semi = options.get('semi', 0.1)
        self.ratio = options.get('ratio', 2.0)

        # ODE integration options
        self.atol = options.get('atol', 1e-6)
        self.rtol = options.get('rtol', 1e-3)
        self.initial_step = options.get('initial_step', 1e-3)

        # Wave 33: Sommerfeld integration backend selection.
        # use_ode=True selects scipy solve_ivp with RK45 to mirror MATLAB ode45
        # (AbsTol=1e-6, RelTol=1e-3, InitialStep=1e-3). Default False keeps the
        # high-accuracy GL panels backend used by 69 perf demos. Demos requiring
        # MATLAB-faithful Sommerfeld inaccuracy (demospecret13/dipret8/dipret9)
        # can opt in via LayerStructure(..., use_ode=True).
        self.use_ode = options.get('use_ode', False)
        # Wave 48: bit-exact MATLAB ode45 step controller. When set, the ODE
        # backend uses our matlab_ode45 reimplementation (1:1 with MATLAB's
        # ode45.m PI-style controller and error norm) instead of scipy
        # solve_ivp. Only active when use_ode is also True.
        self.use_matlab_ode45 = options.get('use_matlab_ode45', False)

        # Sommerfeld GL panel/order configuration (Wave 26)
        # Defaults match Wave 20 A revert (suitable for most demos).
        # demospecret13 (near-interface, sub-nm gap) needs higher resolution
        # and can override these via LayerStructure(..., gl_semi_panels=...).
        self.gl_semi_panels = options.get('gl_semi_panels', 4)
        self.gl_semi_order = options.get('gl_semi_order', 40)
        self.gl_real_panels = options.get('gl_real_panels', 10)
        self.gl_real_order = options.get('gl_real_order', 40)
        self.gl_imag_panels = options.get('gl_imag_panels', 10)
        self.gl_imag_order = options.get('gl_imag_order', 40)

        # Wave 51: opt-in MATLAB Engine API path for Green function
        # evaluation. When enabled, LayerStructure.green() is delegated to a
        # MATLAB session via mnpbem_layer_green_helper.m, yielding bit-
        # identical results vs MATLAB. Default False (no engine dependency).
        self.use_matlab_engine = options.get('use_matlab_engine', False)
        self._matlab_engine = None
        self._matlab_eps_specs_cmd = None
        if self.use_matlab_engine:
            self._init_matlab_engine(epstab)

    def _init_matlab_engine(self, epstab: list) -> None:
        # Wave 51: spin up MATLAB Engine and pre-build a workspace command
        # that recreates the epstab cell array used to construct
        # layerstructure.  Each entry must map to a Python EpsConst /
        # EpsTable so we can mirror it to MATLAB's epsconst / epstable.
        import matlab.engine as _matlab_engine_mod

        eng = _matlab_engine_mod.start_matlab()
        eng.addpath(eng.genpath('/home/yoojk20/workspace/MNPBEM'), nargout=0)

        specs = []
        for eps in epstab:
            cls_name = type(eps).__name__
            if cls_name == 'EpsConst':
                specs.append("{{'const',{:.17g}}}".format(float(eps.eps.real)))
            elif cls_name == 'EpsTable':
                specs.append("{{'table','{}'}}".format(eps.filename))
            else:
                raise ValueError(
                    "use_matlab_engine: unsupported eps class {}".format(cls_name))

        self._matlab_engine = eng
        self._matlab_eps_specs_cmd = "eps_specs = { " + ", ".join(specs) + " };"

    @property
    def n(self) -> int:
        return len(self.z)

    def indlayer(self,
            z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # MATLAB: layers are ordered with decreasing z-values
        # [~, ind] = histc(-z, [-inf, -obj.z, inf])
        z = np.asarray(z, dtype = float)
        shape = z.shape
        z_flat = z.ravel()

        # Bin edges: [-inf, -z[0], -z[1], ..., inf] for -z
        edges = np.empty(len(self.z) + 2, dtype = float)
        edges[0] = -np.inf
        edges[1:-1] = -self.z
        edges[-1] = np.inf

        ind = np.digitize(-z_flat, edges)
        # digitize returns 1-based bin indices, we keep them 1-based for MATLAB compat
        ind = ind.reshape(shape)

        # Is point located in layer?
        zmin_vals, _ = self.mindist(z)
        in_layer = np.abs(zmin_vals) < self.ztol

        return ind, in_layer

    def mindist(self,
            z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = np.asarray(z, dtype = float)
        shape = z.shape
        z_flat = z.ravel()

        # Compute distance to each interface
        # z_flat: (n,), self.z: (m,) -> diff: (n, m)
        diff = z_flat[:, np.newaxis] - self.z[np.newaxis, :]

        # Find minimum absolute distance
        abs_diff = np.abs(diff)
        ind = np.argmin(abs_diff, axis = 1)
        zmin = abs_diff[np.arange(len(z_flat)), ind]

        # MATLAB uses 1-based indexing for ind
        ind = ind + 1

        return zmin.reshape(shape), ind.reshape(shape)

    def round_z(self, *z_args: np.ndarray) -> Tuple[np.ndarray, ...]:
        # MATLAB: round.m
        results = []

        for z in z_args:
            z = np.asarray(z, dtype = float).copy()
            zmin_vals, ind = self.mindist(z)

            # z-value of nearest layer (1-based ind -> 0-based)
            ztab = self.z[ind - 1]

            # Shift direction
            direction = np.sign(z - ztab)

            # Shift points that are too close to layer
            mask = zmin_vals <= self.zmin
            z[mask] = ztab[mask] + direction[mask] * self.zmin

            results.append(z)

        return tuple(results)

    def _mul(self,
            a: np.ndarray,
            b: np.ndarray) -> np.ndarray:
        # MATLAB: private/mul.m
        a = np.asarray(a)
        b = np.asarray(b)

        if a.shape == b.shape:
            return a * b
        else:
            # Outer product
            a_flat = a.ravel()
            b_flat = b.ravel()
            c = np.outer(a_flat, b_flat)

            # Determine output shape
            siza = list(a.shape)
            sizb = list(b.shape)
            if siza[-1] == 1:
                siza = siza[:-1]
            if sizb[0] == 1:
                sizb = sizb[1:]
            out_shape = siza + sizb
            if len(out_shape) == 0:
                return c.ravel()[0]
            return c.reshape(out_shape)

    def reflection(self,
            enei: float,
            kpar: Union[float, complex],
            pos: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # MATLAB: reflection.m
        # Use simpler equations for substrate in case of single interface
        if len(self.z) == 1:
            return self._reflection_subs(enei, kpar, pos)

        return self._reflection_full(enei, kpar, pos)

    def _reflection_subs(self,
            enei: float,
            kpar: Union[float, complex],
            pos: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # MATLAB: reflectionsubs.m
        # Dielectric functions and wavenumbers in media
        eps_vals = np.empty(len(self.eps), dtype = complex)
        k_vals = np.empty(len(self.eps), dtype = complex)
        for i, eps_func in enumerate(self.eps):
            eps_vals[i], k_vals[i] = eps_func(enei)

        # z-component of wavevector
        kz = m_sqrt_c(k_vals ** 2 - kpar ** 2)
        kz = kz * np.sign(np.imag(kz + 1e-10j))

        # Dielectric functions and wavenumbers
        # eps1 is above, eps2 is below
        eps1 = eps_vals[0]
        k1z = kz[0]
        eps2 = eps_vals[1]
        k2z = kz[1]

        # Parallel surface current
        rr_p = (k1z - k2z) / (k2z + k1z)
        r_p = np.array([[rr_p, 1 + rr_p], [1 - rr_p, -rr_p]], dtype = complex)

        # Wavenumber of light in vacuum
        k0 = 2 * np.pi / enei

        # Auxiliary quantity
        Delta = (k2z + k1z) * (eps1 * k2z + eps2 * k1z)

        # Safe ratio: k1z/k2z and k2z/k1z (avoid divide-by-zero)
        # MATLAB: these divisions produce NaN when kz=0, but the off-diagonal
        # elements are multiplied by factors that go to 0 at the same rate,
        # so the product is finite. Use safe division to handle this.
        k1z_safe = k1z if np.abs(k1z) > 1e-30 else 1e-30 + 0j
        k2z_safe = k2z if np.abs(k2z) > 1e-30 else 1e-30 + 0j
        ratio_12 = k1z / k2z_safe  # k1z / k2z
        ratio_21 = k2z / k1z_safe  # k2z / k1z

        # Surface charge from surface charge source
        r_ss_11 = (k1z + k2z) * (2 * eps1 * k1z - eps2 * k1z - eps1 * k2z) / Delta
        r_ss_22 = (k2z + k1z) * (2 * eps2 * k2z - eps1 * k2z - eps2 * k1z) / Delta
        r_ss = np.array([[r_ss_11, ratio_12 * (r_ss_22 + 1)],
                         [ratio_21 * (r_ss_11 + 1), r_ss_22]], dtype = complex)

        # Induced surface current from surface charge source
        r_hs_11 = -2 * k0 * (eps2 - eps1) * eps1 * k1z / Delta
        r_hs_22 = -2 * k0 * (eps1 - eps2) * eps2 * k2z / Delta
        r_hs = np.array([[r_hs_11, -ratio_12 * r_hs_22],
                         [ratio_21 * r_hs_11, -r_hs_22]], dtype = complex)

        # Induced surface charge from surface current source
        r_sh_11 = -2 * k0 * (eps2 - eps1) * k1z / Delta
        r_sh_22 = -2 * k0 * (eps1 - eps2) * k2z / Delta
        r_sh = np.array([[r_sh_11, -ratio_12 * r_sh_22],
                         [ratio_21 * r_sh_11, -r_sh_22]], dtype = complex)

        # Surface current from surface current source
        r_hh_11 = (k1z - k2z) * (2 * eps1 * k1z - eps2 * k1z + eps1 * k2z) / Delta
        r_hh_22 = (k2z - k1z) * (2 * eps2 * k2z - eps1 * k2z + eps2 * k1z) / Delta
        r_hh = np.array([[r_hh_11, ratio_12 * (r_hh_22 + 1)],
                         [ratio_21 * (r_hh_11 + 1), r_hh_22]], dtype = complex)

        r = {'p': r_p, 'ss': r_ss, 'hs': r_hs, 'sh': r_sh, 'hh': r_hh}
        rz = {}

        # Green function propagation factors
        ind1 = np.atleast_1d(pos['ind1'])
        ind2 = np.atleast_1d(pos['ind2'])
        z1 = np.atleast_1d(pos['z1'])
        z2 = np.atleast_1d(pos['z2'])

        abs_z1 = np.abs(z1[:, np.newaxis] - self.z) if z1.ndim == 1 else np.abs(z1 - self.z)
        abs_z2 = np.abs(z2[:, np.newaxis] - self.z) if z2.ndim == 1 else np.abs(z2 - self.z)
        g1 = m_exp_c(1j * kz[ind1 - 1][:, np.newaxis] * abs_z1)
        g2 = m_exp_c(1j * kz[ind2 - 1][:, np.newaxis] * abs_z2)
        # Derivative of Green function wrt z-value
        sign_z1 = np.sign(z1[:, np.newaxis] - self.z) if z1.ndim == 1 else np.sign(z1 - self.z)
        g1z = g1 * sign_z1

        # Apply propagation factors
        r_out = {}
        rz_out = {}

        same_size = (ind1.shape == ind2.shape)

        for name in r.keys():
            rr = r[name]
            if same_size:
                idx = (ind1 - 1, ind2 - 1)
                rr_sel = rr[idx]
                r_out[name] = g1.ravel() * rr_sel * g2.ravel()
                rz_out[name] = g1z.ravel() * rr_sel * g2.ravel()
            else:
                r_out[name] = rr[ind1 - 1][:, ind2 - 1] * np.outer(g1.ravel(), g2.ravel())
                rz_out[name] = rr[ind1 - 1][:, ind2 - 1] * np.outer(g1z.ravel(), g2.ravel())

        return r_out, rz_out

    def _reflection_full(self,
            enei: float,
            kpar: Union[float, complex],
            pos: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # MATLAB: reflection.m (multi-layer case)
        # Wavenumbers in media
        k_vals = np.empty(len(self.eps), dtype = complex)
        for i, eps_func in enumerate(self.eps):
            _, k_vals[i] = eps_func(enei)

        # Perpendicular component of wavevector
        # MATLAB intbessel.m / reflection.m: kz = sqrt(...); kz = kz * sign(imag(kz + 1e-10i))
        # The 1e-10j is only used inside sign() to break ties, not added to kz itself.
        kz = m_sqrt_c(k_vals ** 2 - kpar ** 2)
        kz = kz * np.sign(np.imag(kz + 1e-10j))

        ind1 = np.atleast_1d(pos['ind1'])
        ind2 = np.atleast_1d(pos['ind2'])
        z1 = np.atleast_1d(pos['z1'])
        z2 = np.atleast_1d(pos['z2'])

        k1z = kz[ind1 - 1]
        k2z = kz[ind2 - 1]

        # Distance to lower interfaces
        z_lower = np.append(self.z, -1e100)
        z_upper = np.append(1e100, self.z)
        dn1 = np.abs(z1 - z_lower[ind1 - 1])
        dn2 = np.abs(z2 - z_lower[ind2 - 1])
        up1 = np.abs(z1 - z_upper[ind1 - 1])
        up2 = np.abs(z2 - z_upper[ind2 - 1])

        # Excitation matrix
        n_ind2 = len(ind2)
        siz = (2 * len(self.z) + 2, n_ind2)

        fac = 2j * np.pi / k2z

        exc = np.zeros(siz, dtype = complex)
        for j in range(n_ind2):
            exc[2 * ind2[j] - 1, j] += fac[j] * m_exp_c(1j * k2z[j] * dn2[j])
            exc[2 * ind2[j] - 2, j] += fac[j] * m_exp_c(1j * k2z[j] * up2[j])

        # Remove layers at infinity
        exc = exc[1:-1, :]

        # BEM solve
        par, perp = self.bemsolve(enei, kpar)

        same_size = (ind1.shape == ind2.shape)

        def _multiply(a, b):
            if same_size:
                return a[np.arange(len(ind1)), :][
                    np.arange(len(ind1)),
                    np.arange(a.shape[1]) if a.shape[1] == len(ind1) else 0
                ] * b if a.ndim == 2 else a * b
            else:
                return a[ind1 - 1, :] * b

        r = {}
        rz = {}

        # Parallel surface current
        y = par @ exc
        n_z = len(self.z)
        zeros_row = np.zeros((1, y.shape[1]), dtype = complex)

        h1_p = np.empty((n_z + 1, y.shape[1]), dtype = complex)
        h1_p[0, :] = 0
        h1_p[1:, :] = y[1::2, :]

        h2_p = np.empty((n_z + 1, y.shape[1]), dtype = complex)
        h2_p[:-1, :] = y[0::2, :]
        h2_p[-1, :] = 0

        r['p'] = self._layer_multiply(pos, h2_p, m_exp_c(1j * k1z * dn1), ind1) + \
                 self._layer_multiply(pos, h1_p, m_exp_c(1j * k1z * up1), ind1)
        rz['p'] = self._layer_multiply(pos, h2_p, m_exp_c(1j * k1z * dn1), ind1) - \
                  self._layer_multiply(pos, h1_p, m_exp_c(1j * k1z * up1), ind1)

        # Surface charge
        exc2 = np.zeros((2 * exc.shape[0], exc.shape[1]), dtype = complex)
        exc2[0::2, :] = exc

        y = perp @ exc2

        sig1 = np.empty((n_z + 1, y.shape[1]), dtype = complex)
        sig1[0, :] = 0
        sig1[1:, :] = y[2::4, :]

        sig2 = np.empty((n_z + 1, y.shape[1]), dtype = complex)
        sig2[:-1, :] = y[0::4, :]
        sig2[-1, :] = 0

        r['ss'] = self._layer_multiply(pos, sig2, m_exp_c(1j * k1z * dn1), ind1) + \
                  self._layer_multiply(pos, sig1, m_exp_c(1j * k1z * up1), ind1)
        rz['ss'] = self._layer_multiply(pos, sig2, m_exp_c(1j * k1z * dn1), ind1) - \
                   self._layer_multiply(pos, sig1, m_exp_c(1j * k1z * up1), ind1)

        h1_s = np.empty((n_z + 1, y.shape[1]), dtype = complex)
        h1_s[0, :] = 0
        h1_s[1:, :] = y[3::4, :]

        h2_s = np.empty((n_z + 1, y.shape[1]), dtype = complex)
        h2_s[:-1, :] = y[1::4, :]
        h2_s[-1, :] = 0

        r['hs'] = self._layer_multiply(pos, h2_s, m_exp_c(1j * k1z * dn1), ind1) + \
                  self._layer_multiply(pos, h1_s, m_exp_c(1j * k1z * up1), ind1)
        rz['hs'] = self._layer_multiply(pos, h2_s, m_exp_c(1j * k1z * dn1), ind1) - \
                   self._layer_multiply(pos, h1_s, m_exp_c(1j * k1z * up1), ind1)

        # Perpendicular surface current
        exc2 = np.zeros((2 * exc.shape[0], exc.shape[1]), dtype = complex)
        exc2[1::2, :] = exc

        y = perp @ exc2

        sig1 = np.empty((n_z + 1, y.shape[1]), dtype = complex)
        sig1[0, :] = 0
        sig1[1:, :] = y[2::4, :]

        sig2 = np.empty((n_z + 1, y.shape[1]), dtype = complex)
        sig2[:-1, :] = y[0::4, :]
        sig2[-1, :] = 0

        r['sh'] = self._layer_multiply(pos, sig2, m_exp_c(1j * k1z * dn1), ind1) + \
                  self._layer_multiply(pos, sig1, m_exp_c(1j * k1z * up1), ind1)
        rz['sh'] = self._layer_multiply(pos, sig2, m_exp_c(1j * k1z * dn1), ind1) - \
                   self._layer_multiply(pos, sig1, m_exp_c(1j * k1z * up1), ind1)

        h1_h = np.empty((n_z + 1, y.shape[1]), dtype = complex)
        h1_h[0, :] = 0
        h1_h[1:, :] = y[3::4, :]

        h2_h = np.empty((n_z + 1, y.shape[1]), dtype = complex)
        h2_h[:-1, :] = y[1::4, :]
        h2_h[-1, :] = 0

        r['hh'] = self._layer_multiply(pos, h2_h, m_exp_c(1j * k1z * dn1), ind1) + \
                  self._layer_multiply(pos, h1_h, m_exp_c(1j * k1z * up1), ind1)
        rz['hh'] = self._layer_multiply(pos, h2_h, m_exp_c(1j * k1z * dn1), ind1) - \
                   self._layer_multiply(pos, h1_h, m_exp_c(1j * k1z * up1), ind1)

        return r, rz

    def _layer_multiply(self,
            pos: Dict[str, Any],
            a: np.ndarray,
            b: np.ndarray,
            ind1: np.ndarray) -> np.ndarray:
        z1 = np.atleast_1d(pos['z1'])
        z2 = np.atleast_1d(pos['z2'])
        ind2 = np.atleast_1d(pos['ind2'])

        same_size = (z1.shape == z2.shape)

        if same_size:
            # Direct product
            return a[ind1 - 1, np.arange(a.shape[1])] * b
        else:
            return a[ind1 - 1, :] * b[:, np.newaxis]

    def fresnel(self,
            enei: float,
            kpar: Union[float, complex],
            pos: Dict[str, Any]) -> Dict[str, Any]:
        # MATLAB: fresnel.m
        # Wavenumber in media
        k_vals = np.empty(len(self.eps), dtype = complex)
        for i, eps_func in enumerate(self.eps):
            _, k_vals[i] = eps_func(enei)

        # Perpendicular component of wavevector
        # MATLAB intbessel.m / reflection.m: kz = sqrt(...); kz = kz * sign(imag(kz + 1e-10i))
        # The 1e-10j is only used inside sign() to break ties, not added to kz itself.
        kz = m_sqrt_c(k_vals ** 2 - kpar ** 2)
        kz = kz * np.sign(np.imag(kz + 1e-10j))

        # Perpendicular components
        ind1 = np.atleast_1d(pos['ind1'])
        ind2 = np.atleast_1d(pos['ind2'])
        k1z = kz[ind1 - 1]
        k2z = kz[ind2 - 1]

        # Ratio of z-components
        z1 = np.atleast_1d(pos['z1'])
        z2 = np.atleast_1d(pos['z2'])
        if z1.shape == z2.shape:
            ratio = k2z / k1z
        else:
            ratio = np.outer(1.0 / k1z, k2z)

        # Reflection and transmission coefficients
        r, _ = self.reflection(enei, kpar, pos)

        # Correct: REFLECTION uses surface charges/currents, FRESNEL uses potentials
        for name in r.keys():
            r[name] = r[name] * ratio

        return r

    def efresnel(self,
            pol: np.ndarray,
            dir: np.ndarray,
            enei: float) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        # MATLAB: efresnel.m
        pol = np.atleast_2d(pol)
        dir = np.atleast_2d(dir)

        k0 = 2 * np.pi / enei
        k_vals = np.empty(len(self.eps), dtype = complex)
        for i, eps_func in enumerate(self.eps):
            _, k_vals[i] = eps_func(enei)

        # Upper and lower layers
        z1_val = self.z[0] + 1e-10
        ind1 = 1
        z2_val = self.z[-1] - 1e-10
        ind2 = 1 + len(self.z)

        n_dir = pol.shape[0]
        ei = np.zeros_like(pol, dtype = complex)
        er = np.zeros_like(pol, dtype = complex)
        et = np.zeros_like(pol, dtype = complex)
        ki = np.zeros_like(pol, dtype = complex)
        kr = np.zeros_like(pol, dtype = complex)
        kt = np.zeros_like(pol, dtype = complex)

        for i in range(n_dir):
            if dir[i, 2] < 0:
                # Excitation through upper medium
                posr = {'r': 0, 'z1': z1_val, 'ind1': ind1, 'z2': z1_val, 'ind2': ind1}
                post = {'r': 0, 'z1': z2_val, 'ind1': ind2, 'z2': z1_val, 'ind2': ind1}
            else:
                # Excitation through lower medium
                posr = {'r': 0, 'z1': z2_val, 'ind1': ind2, 'z2': z2_val, 'ind2': ind2}
                post = {'r': 0, 'z1': z1_val, 'ind1': ind1, 'z2': z2_val, 'ind2': ind2}

            kpar_vec = k_vals[post['ind2'] - 1] * dir[i, 0:2]
            kpar_mag = m_sqrt_c(np.sum(kpar_vec ** 2))

            kzr = m_sqrt_c(k_vals[posr['ind1'] - 1] ** 2 - kpar_mag ** 2)
            kzr = kzr * np.sign(np.imag(kzr + 1e-10j))
            kzt = m_sqrt_c(k_vals[post['ind1'] - 1] ** 2 - kpar_mag ** 2)
            kzt = kzt * np.sign(np.imag(kzt + 1e-10j))

            ki[i, :] = np.array([kpar_vec[0], kpar_vec[1],
                                 np.sign(dir[i, 2]) * kzr])
            kr[i, :] = np.array([kpar_vec[0], kpar_vec[1],
                                 -np.sign(dir[i, 2]) * kzr])
            kt[i, :] = np.array([kpar_vec[0], kpar_vec[1],
                                 np.sign(dir[i, 2]) * kzt])

            # Reflection and transmission coefficients
            r = self.fresnel(enei, kpar_mag, posr)
            t = self.fresnel(enei, kpar_mag, post)

            ei[i, :] = pol[i, :]

            # Reflected and transmitted electric field
            r_p = np.atleast_1d(r['p']).ravel()[0]
            r_hh = np.atleast_1d(r['hh']).ravel()[0]
            r_sh = np.atleast_1d(r['sh']).ravel()[0]
            t_p = np.atleast_1d(t['p']).ravel()[0]
            t_hh = np.atleast_1d(t['hh']).ravel()[0]
            t_sh = np.atleast_1d(t['sh']).ravel()[0]

            er[i, 0:2] = r_p * pol[i, 0:2]
            er[i, 2] = r_hh * pol[i, 2]
            er[i, :] = er[i, :] - kr[i, :] / k0 * r_sh * pol[i, 2]

            et[i, 0:2] = t_p * pol[i, 0:2]
            et[i, 2] = t_hh * pol[i, 2]
            et[i, :] = et[i, :] - kt[i, :] / k0 * t_sh * pol[i, 2]

            # Phase factors
            er[i, :] *= m_exp_c(-1j * kr[i, 2] * posr['z2'] - 1j * kr[i, 2] * posr['z1'])
            et[i, :] *= m_exp_c(-1j * kr[i, 2] * post['z2'] - 1j * kt[i, 2] * post['z1'])

        e = {'i': ei, 'r': er, 't': et}
        k = {'i': ki, 'r': kr, 't': kt}
        return e, k

    def _matlab_green(self,
            enei: float,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        # Wave 51: delegate to MATLAB Engine for bit-identical Green output.
        import matlab as _matlab_mod

        eng = self._matlab_engine
        r_in = np.asarray(r, dtype=float)
        z1_in = np.asarray(z1, dtype=float)
        z2_in = np.asarray(z2, dtype=float)

        # Mirror Python's pos handling: green.m flattens internally and
        # returns shape(r) at the end; we replicate the squeezed shape on
        # the Python side for parity with the local code path.
        r_flat = r_in.ravel().tolist()
        z1_flat = z1_in.ravel().tolist()
        z2_flat = z2_in.ravel().tolist()

        eng.workspace['r'] = _matlab_mod.double(r_flat)
        eng.workspace['z1'] = _matlab_mod.double(z1_flat)
        eng.workspace['z2'] = _matlab_mod.double(z2_flat)
        eng.workspace['enei'] = float(enei)
        eng.workspace['layer_inds'] = _matlab_mod.double(
            [int(i) for i in self.ind])
        eng.workspace['ztab'] = _matlab_mod.double(
            [float(z) for z in np.atleast_1d(self.z)])
        eng.eval(self._matlab_eps_specs_cmd, nargout=0)
        eng.eval(
            "opts = struct('ztol',{:.17g},'rmin',{:.17g},'zmin',{:.17g},"
            "'semi',{:.17g},'ratio',{:.17g},'atol',{:.17g},'rtol',{:.17g},"
            "'initial_step',{:.17g});".format(
                self.ztol, self.rmin, self.zmin, self.semi, self.ratio,
                self.atol, self.rtol, self.initial_step),
            nargout=0)
        eng.eval(
            "[G_r, G_i, Fr_r, Fr_i, Fz_r, Fz_i, names] = "
            "mnpbem_layer_green_helper(eps_specs, layer_inds, ztab, opts, "
            "enei, r, z1, z2);",
            nargout=0)

        names = list(eng.workspace['names'])

        out_shape = r_in.shape

        def fetch(prefix: str, name: str) -> np.ndarray:
            re = np.asarray(eng.eval("{}_r.{}".format(prefix, name)),
                            dtype=float).ravel()
            im = np.asarray(eng.eval("{}_i.{}".format(prefix, name)),
                            dtype=float).ravel()
            arr = re + 1j * im
            return np.squeeze(arr.reshape(out_shape))

        G = {nm: fetch('G', nm) for nm in names}
        Fr = {nm: fetch('Fr', nm) for nm in names}
        Fz = {nm: fetch('Fz', nm) for nm in names}

        # Build pos_out parallel to local green() return value.
        r_local = np.maximum(r_in, self.rmin)
        z1_local, z2_local = self.round_z(z1_in, z2_in)
        r_exp = self._mul(r_local,
                          self._mul(np.ones_like(z1_local),
                                    np.ones_like(z2_local)))
        z1_exp = self._mul(np.ones_like(r_local),
                           self._mul(z1_local, np.ones_like(z2_local)))
        z2_exp = self._mul(np.ones_like(r_local),
                           self._mul(np.ones_like(z1_local), z2_local))
        zmin_1, _ = self.mindist(z1_exp.ravel())
        zmin_2, _ = self.mindist(z2_exp.ravel())
        zmin = (zmin_1 + zmin_2).reshape(r_exp.shape)
        pos_out = {'r': r_exp, 'z1': z1_exp, 'z2': z2_exp, 'zmin': zmin}

        return G, Fr, Fz, pos_out

    def green(self,
            enei: float,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        # MATLAB: green.m
        if self.use_matlab_engine:
            return self._matlab_green(enei, r, z1, z2)
        r = np.asarray(r, dtype = float)
        z1 = np.asarray(z1, dtype = float)
        z2 = np.asarray(z2, dtype = float)

        # Round radii and z-values
        r = np.maximum(r, self.rmin)
        z1, z2 = self.round_z(z1, z2)

        # Save positions
        pos = {'r': r, 'z1': z1, 'z2': z2}
        ind1, _ = self.indlayer(z1)
        ind2, _ = self.indlayer(z2)
        pos['ind1'] = ind1
        pos['ind2'] = ind2

        # Expand positions using mul
        r_exp = self._mul(r, self._mul(np.ones_like(z1), np.ones_like(z2)))
        z1_exp = self._mul(np.ones_like(r), self._mul(z1, np.ones_like(z2)))
        z2_exp = self._mul(np.ones_like(r), self._mul(np.ones_like(z1), z2))

        # Minimal distance to layers
        zmin_1, _ = self.mindist(z1_exp.ravel())
        zmin_2, _ = self.mindist(z2_exp.ravel())
        zmin = (zmin_1 + zmin_2).reshape(r_exp.shape)

        # Size of integrand
        n1 = r_exp.size

        # Get number of reflection coefficient names
        test_r, _ = self.reflection(enei, 0, pos)
        names = list(test_r.keys())
        n_names = len(names)

        # Solve ODEs for complex integration
        # Semi-ellipse integration
        y1 = self._integrate_semiellipse(enei, pos, n1, n_names)

        # Determine integration path
        flat_zmin = zmin.ravel()
        flat_r = r_exp.ravel()
        ind_real = np.where(flat_zmin >= flat_r / self.ratio)[0]
        ind_imag = np.where(flat_zmin < flat_r / self.ratio)[0]

        n2 = len(ind_real)
        n3 = len(ind_imag)
        y2 = np.zeros(15 * n2) if n2 > 0 else np.zeros(0)
        y3 = np.zeros(15 * n3) if n3 > 0 else np.zeros(0)

        # Integration along real axis
        if n2 > 0:
            y2 = self._integrate_real(enei, pos, ind_real, n2, n_names)

        # Integration along imaginary axis
        if n3 > 0:
            y3 = self._integrate_imag(enei, pos, ind_imag, n3, n_names)

        # Combine results
        G = {}
        Fr = {}
        Fz = {}

        for iname, name in enumerate(names):
            g = y1[(iname * 3) * n1:(iname * 3 + 1) * n1].copy()
            fr = y1[(iname * 3 + 1) * n1:(iname * 3 + 2) * n1].copy()
            fz = y1[(iname * 3 + 2) * n1:(iname * 3 + 3) * n1].copy()

            if n2 > 0:
                g[ind_real] += y2[(iname * 3) * n2:(iname * 3 + 1) * n2]
                fr[ind_real] += y2[(iname * 3 + 1) * n2:(iname * 3 + 2) * n2]
                fz[ind_real] += y2[(iname * 3 + 2) * n2:(iname * 3 + 3) * n2]

            if n3 > 0:
                g[ind_imag] += y3[(iname * 3) * n3:(iname * 3 + 1) * n3]
                fr[ind_imag] += y3[(iname * 3 + 1) * n3:(iname * 3 + 2) * n3]
                fz[ind_imag] += y3[(iname * 3 + 2) * n3:(iname * 3 + 3) * n3]

            G[name] = np.squeeze(g.reshape(r_exp.shape))
            Fr[name] = np.squeeze(fr.reshape(r_exp.shape))
            Fz[name] = np.squeeze(fz.reshape(r_exp.shape))

        pos_out = {'r': r_exp, 'z1': z1_exp, 'z2': z2_exp, 'zmin': zmin}
        return G, Fr, Fz, pos_out

    def _build_integrate_context(self,
            enei: float,
            pos: Dict[str, Any]) -> Dict[str, Any]:
        # Precompute pos/enei-dependent quantities shared across ODE RHS evals.
        eps_vals = np.empty(len(self.eps), dtype = complex)
        k_vals = np.empty(len(self.eps), dtype = complex)
        for i, eps_func in enumerate(self.eps):
            eps_vals[i], k_vals[i] = eps_func(enei)

        r_exp = self._mul(pos['r'], self._mul(np.ones_like(pos['z1']), np.ones_like(pos['z2'])))
        r_flat = r_exp.ravel()

        ind1_raw = pos['ind1']
        ind2_raw = pos['ind2']
        z1_raw = pos['z1']
        z2_raw = pos['z2']

        ind1 = np.atleast_1d(ind1_raw).ravel()
        ind2 = np.atleast_1d(ind2_raw).ravel()
        z1 = np.atleast_1d(z1_raw).ravel()
        z2 = np.atleast_1d(z2_raw).ravel()

        # kz expansion index mapping ind1 -> r_flat layout.
        if len(ind1) == len(r_flat):
            kz_expand_idx = ind1 - 1
        else:
            n_r = np.atleast_1d(pos['r']).size
            n_z2 = z2.size
            kz_expand_idx = np.tile(np.repeat(ind1 - 1, n_z2), n_r)[:len(r_flat)]

        # Shape check mirroring _reflection_subs:
        # original used z1.ndim==1 branch. Reproduce that.
        if hasattr(z1_raw, 'ndim') and np.ndim(z1_raw) == 1:
            abs_z1 = np.abs(z1[:, np.newaxis] - self.z)
            abs_z2 = np.abs(z2[:, np.newaxis] - self.z)
            sign_z1 = np.sign(z1[:, np.newaxis] - self.z)
        else:
            abs_z1 = np.abs(np.asarray(z1_raw) - self.z)
            abs_z2 = np.abs(np.asarray(z2_raw) - self.z)
            sign_z1 = np.sign(np.asarray(z1_raw) - self.z)

        same_size_refl = (np.shape(ind1_raw) == np.shape(ind2_raw))

        ctx: Dict[str, Any] = {
            'enei': enei,
            'eps_vals': eps_vals,
            'k_vals': k_vals,
            'k0': 2 * np.pi / enei,
            'r_flat': r_flat,
            'ind1': ind1,
            'ind2': ind2,
            'z1': z1,
            'z2': z2,
            'kz_expand_idx': kz_expand_idx,
            'abs_z1': abs_z1,
            'abs_z2': abs_z2,
            'sign_z1': sign_z1,
            'same_size_refl': same_size_refl,
            'is_subs': (len(self.z) == 1),
            'pos': pos,
        }
        return ctx

    def _reflection_subs_ctx(self,
            kpar: Union[float, complex],
            ctx: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
        # Fast substrate reflection path using precomputed ctx.
        # Returns (refl_dict, reflz_dict, kz_vec).
        eps_vals = ctx['eps_vals']
        k_vals = ctx['k_vals']
        k0 = ctx['k0']

        kz = m_sqrt_c(k_vals ** 2 - kpar ** 2)
        kz = kz * np.sign(np.imag(kz + 1e-10j))

        eps1 = eps_vals[0]
        eps2 = eps_vals[1]
        k1z = kz[0]
        k2z = kz[1]

        rr_p = (k1z - k2z) / (k2z + k1z)
        r_p = np.array([[rr_p, 1 + rr_p], [1 - rr_p, -rr_p]], dtype = complex)

        Delta = (k2z + k1z) * (eps1 * k2z + eps2 * k1z)

        k1z_safe = k1z if np.abs(k1z) > 1e-30 else 1e-30 + 0j
        k2z_safe = k2z if np.abs(k2z) > 1e-30 else 1e-30 + 0j
        ratio_12 = k1z / k2z_safe
        ratio_21 = k2z / k1z_safe

        r_ss_11 = (k1z + k2z) * (2 * eps1 * k1z - eps2 * k1z - eps1 * k2z) / Delta
        r_ss_22 = (k2z + k1z) * (2 * eps2 * k2z - eps1 * k2z - eps2 * k1z) / Delta
        r_ss = np.array([[r_ss_11, ratio_12 * (r_ss_22 + 1)],
                         [ratio_21 * (r_ss_11 + 1), r_ss_22]], dtype = complex)

        r_hs_11 = -2 * k0 * (eps2 - eps1) * eps1 * k1z / Delta
        r_hs_22 = -2 * k0 * (eps1 - eps2) * eps2 * k2z / Delta
        r_hs = np.array([[r_hs_11, -ratio_12 * r_hs_22],
                         [ratio_21 * r_hs_11, -r_hs_22]], dtype = complex)

        r_sh_11 = -2 * k0 * (eps2 - eps1) * k1z / Delta
        r_sh_22 = -2 * k0 * (eps1 - eps2) * k2z / Delta
        r_sh = np.array([[r_sh_11, -ratio_12 * r_sh_22],
                         [ratio_21 * r_sh_11, -r_sh_22]], dtype = complex)

        r_hh_11 = (k1z - k2z) * (2 * eps1 * k1z - eps2 * k1z + eps1 * k2z) / Delta
        r_hh_22 = (k2z - k1z) * (2 * eps2 * k2z - eps1 * k2z + eps2 * k1z) / Delta
        r_hh = np.array([[r_hh_11, ratio_12 * (r_hh_22 + 1)],
                         [ratio_21 * (r_hh_11 + 1), r_hh_22]], dtype = complex)

        r_mat = (('p', r_p), ('ss', r_ss), ('hs', r_hs), ('sh', r_sh), ('hh', r_hh))

        ind1 = ctx['ind1']
        ind2 = ctx['ind2']
        abs_z1 = ctx['abs_z1']
        abs_z2 = ctx['abs_z2']
        sign_z1 = ctx['sign_z1']

        kz_ind1 = kz[ind1 - 1]
        kz_ind2 = kz[ind2 - 1]
        # Propagation factors
        if abs_z1.ndim == 2:
            g1 = m_exp_c(1j * kz_ind1[:, np.newaxis] * abs_z1)
            g2 = m_exp_c(1j * kz_ind2[:, np.newaxis] * abs_z2)
        else:
            g1 = m_exp_c(1j * kz_ind1 * abs_z1)
            g2 = m_exp_c(1j * kz_ind2 * abs_z2)
        g1z = g1 * sign_z1

        r_out: Dict[str, np.ndarray] = {}
        rz_out: Dict[str, np.ndarray] = {}

        same_size = ctx['same_size_refl']
        g1_flat = g1.ravel()
        g2_flat = g2.ravel()
        g1z_flat = g1z.ravel()

        if same_size:
            idx = (ind1 - 1, ind2 - 1)
            for name, rr in r_mat:
                rr_sel = rr[idx]
                r_out[name] = g1_flat * rr_sel * g2_flat
                rz_out[name] = g1z_flat * rr_sel * g2_flat
        else:
            outer_g = np.outer(g1_flat, g2_flat)
            outer_gz = np.outer(g1z_flat, g2_flat)
            for name, rr in r_mat:
                sel = rr[ind1 - 1][:, ind2 - 1]
                r_out[name] = sel * outer_g
                rz_out[name] = sel * outer_gz

        return r_out, rz_out, kz

    def _bemsolve_batch(self,
            enei: float,
            kpar_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Vectorized bemsolve over array of kpar values.
        # Returns par shape (M, 2n, 2n) and perp shape (M, 4n, 4n).
        kpar_arr = np.asarray(kpar_arr)
        M = kpar_arr.size
        k0 = 2 * np.pi / enei

        eps_vals = np.empty(len(self.eps), dtype = complex)
        k_vals = np.empty(len(self.eps), dtype = complex)
        for i, eps_func in enumerate(self.eps):
            eps_vals[i], k_vals[i] = eps_func(enei)

        # kz shape (M, neps)
        kz = m_sqrt_c(k_vals[np.newaxis, :] ** 2 - kpar_arr[:, np.newaxis] ** 2)
        kz = kz * np.sign(np.imag(kz + 1e-10j))

        n = len(self.z)
        G0 = 2j * np.pi / kz  # (M, neps)

        if n > 1:
            dz = np.abs(np.diff(self.z))  # (n-1,)
            G = 2j * np.pi / kz[:, 1:-1] * m_exp_c(1j * kz[:, 1:-1] * dz[np.newaxis, :])  # (M, n-1)
        else:
            G = np.zeros((M, 0), dtype = complex)

        # Parallel surface current
        siz = 2 * n
        lhs = np.zeros((M, siz, siz), dtype = complex)
        rhs_mat = np.zeros((M, siz, siz), dtype = complex)

        i1 = np.arange(1, 2 * n, 2)
        i2 = np.arange(0, 2 * n, 2)
        eq1 = np.arange(0, 2 * n, 2)
        eq2 = np.arange(1, 2 * n, 2)

        for idx in range(n):
            lhs[:, eq1[idx], i1[idx]] = G0[:, idx + 1]
            lhs[:, eq1[idx], i2[idx]] = -G0[:, idx]
            rhs_mat[:, eq1[idx], i1[idx]] = -1
            rhs_mat[:, eq1[idx], i2[idx]] = 1

        if n > 1:
            for idx in range(n - 1):
                lhs[:, eq1[idx + 1], i1[idx]] = -G[:, idx]
                lhs[:, eq1[idx], i2[idx + 1]] = G[:, idx]

        for idx in range(n):
            lhs[:, eq2[idx], i1[idx]] = 2j * np.pi
            lhs[:, eq2[idx], i2[idx]] = 2j * np.pi
            rhs_mat[:, eq2[idx], i1[idx]] = kz[:, idx + 1]
            rhs_mat[:, eq2[idx], i2[idx]] = kz[:, idx]

        if n > 1:
            for idx in range(n - 1):
                lhs[:, eq2[idx + 1], i1[idx]] = -kz[:, idx + 1] * G[:, idx]
                lhs[:, eq2[idx], i2[idx + 1]] = -kz[:, idx + 1] * G[:, idx]

        par = np.linalg.solve(lhs, rhs_mat)  # (M, 2n, 2n)

        # Perpendicular surface current and surface charge
        siz = 4 * n
        lhs = np.zeros((M, siz, siz), dtype = complex)
        rhs_mat = np.zeros((M, siz, siz), dtype = complex)

        i1 = np.arange(2, 4 * n, 4)
        i2 = np.arange(0, 4 * n, 4)
        j1 = np.arange(3, 4 * n, 4)
        j2 = np.arange(1, 4 * n, 4)

        eq1 = np.arange(0, 4 * n, 4)
        eq2 = np.arange(1, 4 * n, 4)
        eq3 = np.arange(2, 4 * n, 4)
        eq4 = np.arange(3, 4 * n, 4)

        for idx in range(n):
            lhs[:, eq1[idx], i1[idx]] = G0[:, idx + 1]
            lhs[:, eq1[idx], i2[idx]] = -G0[:, idx]
            rhs_mat[:, eq1[idx], i1[idx]] = -1
            rhs_mat[:, eq1[idx], i2[idx]] = 1

            lhs[:, eq2[idx], j1[idx]] = G0[:, idx + 1]
            lhs[:, eq2[idx], j2[idx]] = -G0[:, idx]
            rhs_mat[:, eq2[idx], j1[idx]] = -1
            rhs_mat[:, eq2[idx], j2[idx]] = 1

            lhs[:, eq3[idx], i1[idx]] = 2j * np.pi * eps_vals[idx + 1]
            lhs[:, eq3[idx], i2[idx]] = 2j * np.pi * eps_vals[idx]
            lhs[:, eq3[idx], j1[idx]] = k0 * G0[:, idx + 1] * eps_vals[idx + 1]
            lhs[:, eq3[idx], j2[idx]] = -k0 * G0[:, idx] * eps_vals[idx]
            rhs_mat[:, eq3[idx], i1[idx]] = kz[:, idx + 1] * eps_vals[idx + 1]
            rhs_mat[:, eq3[idx], i2[idx]] = kz[:, idx] * eps_vals[idx]
            rhs_mat[:, eq3[idx], j1[idx]] = -k0 * eps_vals[idx + 1]
            rhs_mat[:, eq3[idx], j2[idx]] = k0 * eps_vals[idx]

            lhs[:, eq4[idx], j1[idx]] = 2j * np.pi
            lhs[:, eq4[idx], j2[idx]] = 2j * np.pi
            lhs[:, eq4[idx], i1[idx]] = k0 * G0[:, idx + 1] * eps_vals[idx + 1]
            lhs[:, eq4[idx], i2[idx]] = -k0 * G0[:, idx] * eps_vals[idx]
            rhs_mat[:, eq4[idx], j1[idx]] = kz[:, idx + 1]
            rhs_mat[:, eq4[idx], j2[idx]] = kz[:, idx]
            rhs_mat[:, eq4[idx], i1[idx]] = -k0 * eps_vals[idx + 1]
            rhs_mat[:, eq4[idx], i2[idx]] = k0 * eps_vals[idx]

        if n > 1:
            for idx in range(n - 1):
                lhs[:, eq1[idx + 1], i1[idx]] = -G[:, idx]
                lhs[:, eq1[idx], i2[idx + 1]] = G[:, idx]

                lhs[:, eq2[idx + 1], j1[idx]] = -G[:, idx]
                lhs[:, eq2[idx], j2[idx + 1]] = G[:, idx]

                lhs[:, eq3[idx + 1], i1[idx]] = -kz[:, idx + 1] * eps_vals[idx + 1] * G[:, idx]
                lhs[:, eq3[idx], i2[idx + 1]] = -kz[:, idx + 1] * eps_vals[idx + 1] * G[:, idx]
                lhs[:, eq3[idx + 1], j1[idx]] = -k0 * eps_vals[idx + 1] * G[:, idx]
                lhs[:, eq3[idx], j2[idx + 1]] = k0 * eps_vals[idx + 1] * G[:, idx]

                lhs[:, eq4[idx + 1], j1[idx]] = -kz[:, idx + 1] * G[:, idx]
                lhs[:, eq4[idx], j2[idx + 1]] = -kz[:, idx + 1] * G[:, idx]
                lhs[:, eq4[idx + 1], i1[idx]] = -k0 * eps_vals[idx + 1] * G[:, idx]
                lhs[:, eq4[idx], i2[idx + 1]] = k0 * eps_vals[idx + 1] * G[:, idx]

        perp = np.linalg.solve(lhs, rhs_mat)  # (M, 4n, 4n)

        return par, perp

    def _reflection_full_batch(self,
            kpar_arr: np.ndarray,
            ctx: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
        # Vectorized multi-layer reflection over array of kpar.
        # Returns refl[name], reflz[name] shape (M, n_pos), and kz shape (M, neps).
        # Only supports the same_size (ind1.shape==ind2.shape) case used by
        # _intbessel_batch / _inthankel_batch in the Green integral loop.
        enei = ctx['enei']
        k_vals = ctx['k_vals']
        ind1 = ctx['ind1']
        ind2 = ctx['ind2']
        z1 = ctx['z1']
        z2 = ctx['z2']

        kpar_arr = np.asarray(kpar_arr)
        M = kpar_arr.size

        # MATLAB intbessel.m: kz = sqrt(...); kz = kz * sign(imag(kz + 1e-10i))
        kz = m_sqrt_c(k_vals[np.newaxis, :] ** 2 - kpar_arr[:, np.newaxis] ** 2)
        kz = kz * np.sign(np.imag(kz + 1e-10j))  # (M, neps)

        k1z = kz[:, ind1 - 1]  # (M, n_pos)
        k2z = kz[:, ind2 - 1]  # (M, n_pos)

        n_pos = len(ind2)
        n_z = len(self.z)

        z_lower = np.append(self.z, -1e100)
        z_upper = np.append(1e100, self.z)
        dn1 = np.abs(z1 - z_lower[ind1 - 1])  # (n_pos,)
        dn2 = np.abs(z2 - z_lower[ind2 - 1])
        up1 = np.abs(z1 - z_upper[ind1 - 1])
        up2 = np.abs(z2 - z_upper[ind2 - 1])

        # Build excitation matrix (M, 2*n_z + 2, n_pos), then drop endpoints.
        exc = np.zeros((M, 2 * n_z + 2, n_pos), dtype = complex)
        fac = 2j * np.pi / k2z  # (M, n_pos)
        pos_idx = np.arange(n_pos)
        exp_dn2 = m_exp_c(1j * k2z * dn2[np.newaxis, :])
        exp_up2 = m_exp_c(1j * k2z * up2[np.newaxis, :])
        for j in range(n_pos):
            exc[:, 2 * ind2[j] - 1, j] += fac[:, j] * exp_dn2[:, j]
            exc[:, 2 * ind2[j] - 2, j] += fac[:, j] * exp_up2[:, j]
        exc = exc[:, 1:-1, :]  # (M, 2n_z, n_pos)

        par, perp = self._bemsolve_batch(enei, kpar_arr)  # par (M, 2n, 2n), perp (M, 4n, 4n)

        exp_dn1 = m_exp_c(1j * k1z * dn1[np.newaxis, :])  # (M, n_pos)
        exp_up1 = m_exp_c(1j * k1z * up1[np.newaxis, :])

        # Helper: evaluate h2[ind1-1, pos] + h1[ind1-1, pos] where h1/h2 are (M, n_z+1, n_pos)
        # and select the row per-position using ind1.
        def _combine(h1_arr, h2_arr, sign):
            i1_idx = ind1 - 1  # (n_pos,)
            # advanced indexing: pick (M, n_pos) by h[M, i1[j], j]
            h2_sel = h2_arr[:, i1_idx, pos_idx]  # (M, n_pos)
            h1_sel = h1_arr[:, i1_idx, pos_idx]
            return h2_sel * exp_dn1 + sign * h1_sel * exp_up1

        r: Dict[str, np.ndarray] = {}
        rz: Dict[str, np.ndarray] = {}

        # Parallel current
        y = par @ exc  # (M, 2n, n_pos)
        h1_p = np.zeros((M, n_z + 1, n_pos), dtype = complex)
        h2_p = np.zeros((M, n_z + 1, n_pos), dtype = complex)
        h1_p[:, 1:, :] = y[:, 1::2, :]
        h2_p[:, :-1, :] = y[:, 0::2, :]
        r['p'] = _combine(h1_p, h2_p, +1)
        rz['p'] = _combine(h1_p, h2_p, -1)

        # Surface charge
        exc2 = np.zeros((M, 2 * exc.shape[1], n_pos), dtype = complex)
        exc2[:, 0::2, :] = exc
        y = perp @ exc2  # (M, 4n, n_pos)
        sig1 = np.zeros((M, n_z + 1, n_pos), dtype = complex)
        sig2 = np.zeros((M, n_z + 1, n_pos), dtype = complex)
        sig1[:, 1:, :] = y[:, 2::4, :]
        sig2[:, :-1, :] = y[:, 0::4, :]
        r['ss'] = _combine(sig1, sig2, +1)
        rz['ss'] = _combine(sig1, sig2, -1)

        h1_s = np.zeros((M, n_z + 1, n_pos), dtype = complex)
        h2_s = np.zeros((M, n_z + 1, n_pos), dtype = complex)
        h1_s[:, 1:, :] = y[:, 3::4, :]
        h2_s[:, :-1, :] = y[:, 1::4, :]
        r['hs'] = _combine(h1_s, h2_s, +1)
        rz['hs'] = _combine(h1_s, h2_s, -1)

        # Perpendicular current
        exc2 = np.zeros((M, 2 * exc.shape[1], n_pos), dtype = complex)
        exc2[:, 1::2, :] = exc
        y = perp @ exc2
        sig1 = np.zeros((M, n_z + 1, n_pos), dtype = complex)
        sig2 = np.zeros((M, n_z + 1, n_pos), dtype = complex)
        sig1[:, 1:, :] = y[:, 2::4, :]
        sig2[:, :-1, :] = y[:, 0::4, :]
        r['sh'] = _combine(sig1, sig2, +1)
        rz['sh'] = _combine(sig1, sig2, -1)

        h1_h = np.zeros((M, n_z + 1, n_pos), dtype = complex)
        h2_h = np.zeros((M, n_z + 1, n_pos), dtype = complex)
        h1_h[:, 1:, :] = y[:, 3::4, :]
        h2_h[:, :-1, :] = y[:, 1::4, :]
        r['hh'] = _combine(h1_h, h2_h, +1)
        rz['hh'] = _combine(h1_h, h2_h, -1)

        return r, rz, kz

    def _reflection_subs_batch(self,
            kpar_arr: np.ndarray,
            ctx: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
        # Vectorized substrate reflection over array of kpar.
        # kpar_arr shape (M,). Returns refl[name] shape depending on same_size:
        #   same_size: (M, n_flat); not same_size: (M, n1, n2).
        # kz shape (M, 2).
        eps_vals = ctx['eps_vals']
        k_vals = ctx['k_vals']
        k0 = ctx['k0']
        ind1 = ctx['ind1']
        ind2 = ctx['ind2']
        abs_z1 = ctx['abs_z1']
        abs_z2 = ctx['abs_z2']
        sign_z1 = ctx['sign_z1']
        same_size = ctx['same_size_refl']

        kpar_arr = np.asarray(kpar_arr)
        M = kpar_arr.size

        # kz shape (M, 2) — small (2 layers); compute on CPU with MATLAB-bitexact
        # m_sqrt_c so that the reflection coefficients keep their reference values.
        kz = m_sqrt_c(k_vals[np.newaxis, :] ** 2 - kpar_arr[:, np.newaxis] ** 2)
        kz = kz * np.sign(np.imag(kz + 1e-10j))

        k1z = kz[:, 0]
        k2z = kz[:, 1]
        eps1 = eps_vals[0]
        eps2 = eps_vals[1]

        rr_p = (k1z - k2z) / (k2z + k1z)
        Delta = (k2z + k1z) * (eps1 * k2z + eps2 * k1z)

        # safe ratios (avoid divide-by-zero at kz=0)
        k1z_safe = np.where(np.abs(k1z) > 1e-30, k1z, 1e-30 + 0j)
        k2z_safe = np.where(np.abs(k2z) > 1e-30, k2z, 1e-30 + 0j)
        ratio_12 = k1z / k2z_safe
        ratio_21 = k2z / k1z_safe

        r_ss_11 = (k1z + k2z) * (2 * eps1 * k1z - eps2 * k1z - eps1 * k2z) / Delta
        r_ss_22 = (k2z + k1z) * (2 * eps2 * k2z - eps1 * k2z - eps2 * k1z) / Delta
        r_hs_11 = -2 * k0 * (eps2 - eps1) * eps1 * k1z / Delta
        r_hs_22 = -2 * k0 * (eps1 - eps2) * eps2 * k2z / Delta
        r_sh_11 = -2 * k0 * (eps2 - eps1) * k1z / Delta
        r_sh_22 = -2 * k0 * (eps1 - eps2) * k2z / Delta
        r_hh_11 = (k1z - k2z) * (2 * eps1 * k1z - eps2 * k1z + eps1 * k2z) / Delta
        r_hh_22 = (k2z - k1z) * (2 * eps2 * k2z - eps1 * k2z + eps2 * k1z) / Delta

        mats: Dict[str, np.ndarray] = {}
        # r_p: [[rr, 1+rr], [1-rr, -rr]]
        r_p = np.empty((M, 2, 2), dtype = complex)
        r_p[:, 0, 0] = rr_p
        r_p[:, 0, 1] = 1 + rr_p
        r_p[:, 1, 0] = 1 - rr_p
        r_p[:, 1, 1] = -rr_p
        mats['p'] = r_p

        # mat1 pattern for ss, hh: [[r1, rat12*(r2+1)], [rat21*(r1+1), r2]]
        def _mat1(r1, r2):
            m = np.empty((M, 2, 2), dtype = complex)
            m[:, 0, 0] = r1
            m[:, 0, 1] = ratio_12 * (r2 + 1)
            m[:, 1, 0] = ratio_21 * (r1 + 1)
            m[:, 1, 1] = r2
            return m

        # mat2 pattern for hs, sh: [[r1, -rat12*r2], [rat21*r1, -r2]]
        def _mat2(r1, r2):
            m = np.empty((M, 2, 2), dtype = complex)
            m[:, 0, 0] = r1
            m[:, 0, 1] = -ratio_12 * r2
            m[:, 1, 0] = ratio_21 * r1
            m[:, 1, 1] = -r2
            return m

        mats['ss'] = _mat1(r_ss_11, r_ss_22)
        mats['hs'] = _mat2(r_hs_11, r_hs_22)
        mats['sh'] = _mat2(r_sh_11, r_sh_22)
        mats['hh'] = _mat1(r_hh_11, r_hh_22)

        # Propagation factors
        kz_i1 = kz[:, ind1 - 1]  # (M, n1)
        kz_i2 = kz[:, ind2 - 1]  # (M, n2)
        if abs_z1.ndim == 2:
            abs_z1f = abs_z1[:, 0]
            abs_z2f = abs_z2[:, 0]
            sign_z1f = sign_z1[:, 0]
        else:
            abs_z1f = abs_z1
            abs_z2f = abs_z2
            sign_z1f = sign_z1

        # MATLAB-bitexact complex exp on CPU.  The big elementwise outer-product
        # below is what dominates _reflection_subs_batch wall time and is a
        # candidate for GPU dispatch, but g1/g2 themselves are small (M*n).
        # In GPU mode the bit-exact MATLAB libmwmathutil exp is replaced by
        # numpy's vectorized cexp (still ULP-equivalent for our demos) so that
        # ctypes call overhead does not dominate the (M, n) propagation step.
        n_flat_total = (M * (ind1.size * ind2.size if not same_size else ind1.size))
        use_gpu = layer_gpu_active(n_flat_total)
        xp, asnumpy, _ = get_layer_xp(n_flat_total)
        if use_gpu:
            g1 = np.exp(1j * kz_i1 * abs_z1f)
            g2 = np.exp(1j * kz_i2 * abs_z2f)
        else:
            g1 = m_exp_c(1j * kz_i1 * abs_z1f)  # (M, n1)
            g2 = m_exp_c(1j * kz_i2 * abs_z2f)  # (M, n2)
        g1z = g1 * sign_z1f

        refl_out: Dict[str, np.ndarray] = {}
        reflz_out: Dict[str, np.ndarray] = {}
        if same_size:
            i1m = ind1 - 1
            i2m = ind2 - 1
            if use_gpu:
                g1_d = xp.asarray(g1)
                g2_d = xp.asarray(g2)
                g1z_d = xp.asarray(g1z)
                for name, rr in mats.items():
                    rr_d = xp.asarray(rr)
                    rr_sel_d = rr_d[:, i1m, i2m]
                    base_d = rr_sel_d * g2_d
                    refl_out[name] = asnumpy(g1_d * base_d)
                    reflz_out[name] = asnumpy(g1z_d * base_d)
            else:
                for name, rr in mats.items():
                    rr_sel = rr[:, i1m, i2m]
                    base = rr_sel * g2
                    refl_out[name] = g1 * base
                    reflz_out[name] = g1z * base
        else:
            if use_gpu:
                g1_d = xp.asarray(g1)
                g2_d = xp.asarray(g2)
                g1z_d = xp.asarray(g1z)
                outer_g_d = g1_d[:, :, None] * g2_d[:, None, :]
                outer_gz_d = g1z_d[:, :, None] * g2_d[:, None, :]
                for name, rr in mats.items():
                    sel = rr[:, ind1 - 1, :][:, :, ind2 - 1]
                    sel_d = xp.asarray(sel)
                    refl_out[name] = asnumpy(sel_d * outer_g_d)
                    reflz_out[name] = asnumpy(sel_d * outer_gz_d)
            else:
                for name, rr in mats.items():
                    sel = rr[:, ind1 - 1, :][:, :, ind2 - 1]
                    outer_g = g1[:, :, np.newaxis] * g2[:, np.newaxis, :]
                    outer_gz = g1z[:, :, np.newaxis] * g2[:, np.newaxis, :]
                    refl_out[name] = sel * outer_g
                    reflz_out[name] = sel * outer_gz

        return refl_out, reflz_out, kz

    def _intbessel_batch(self,
            kpar_arr: np.ndarray,
            ctx: Dict[str, Any],
            ind: np.ndarray,
            weights: Optional[np.ndarray] = None) -> np.ndarray:
        # Batched _intbessel over an array of kpar values.  Returns (M, 15*n)
        # when ``weights`` is None.  When ``weights`` is provided (shape (M,)),
        # the routine performs ``sum_m weights[m] * y[m, :]`` directly and
        # returns a (15*n,) vector — this lets the GPU path keep the dense
        # intermediate on-device and avoid one extra round-trip.
        kpar_arr = np.asarray(kpar_arr)
        M = kpar_arr.size
        n = len(ind)

        if ctx['is_subs']:
            refl, reflz, kz = self._reflection_subs_batch(kpar_arr, ctx)
        elif ctx.get('same_size_refl', False):
            # Batched multi-layer reflection (same_size case only).
            refl, reflz, kz = self._reflection_full_batch(kpar_arr, ctx)
        else:
            # Fallback: scalar loop (not-same-size case in multi-layer).
            y = np.empty((M, 15 * n), dtype = complex)
            for m in range(M):
                y[m, :] = self._intbessel_ctx(kpar_arr[m], ctx, ind)
            return y

        kz_full = kz[:, ctx['kz_expand_idx']]  # (M, n_flat)
        kz_ind = kz_full[:, ind]

        r_ind = ctx['r_flat'][ind]
        arg = kpar_arr[:, np.newaxis] * r_ind[np.newaxis, :]
        # Bessel accepts complex directly; small cost increase for complex kpar.
        j0 = besselj(0, arg)
        j1 = besselj(1, arg)

        # Wave 49: match MATLAB intbessel.m FP ordering exactly.
        # MATLAB: 1i * j0 .* (rr .* (kpar ./ kz))  - inner factor (rr*kpar/kz) first.
        kpar_col = kpar_arr[:, np.newaxis]
        kpar_sq_col = (kpar_arr ** 2)[:, np.newaxis]

        names = ('p', 'ss', 'hs', 'sh', 'hh')

        # Lane C GPU acceleration: the outer (M, n) = O(M * n_flat) elementwise
        # multiply / assemble dominates this routine.  Move everything onto the
        # device when MNPBEM_GPU_LAYER is active and the workload is large.
        use_gpu = layer_gpu_active(M * n)
        xp, asnumpy, _ = get_layer_xp(M * n)
        if use_gpu:
            j0_d = xp.asarray(j0)
            j1_d = xp.asarray(j1)
            kpar_col_d = xp.asarray(kpar_col)
            kpar_sq_col_d = xp.asarray(kpar_sq_col)
            kz_ind_d = xp.asarray(kz_ind)
            if weights is not None:
                # Reduce on-device.  We accumulate weighted sums per name
                # block to avoid materializing the full (M, 15*n) intermediate.
                w_d = xp.asarray(weights)
                acc = xp.zeros(15 * n, dtype = xp.complex128)
                for iname, name in enumerate(names):
                    rr = refl[name]
                    rrz = reflz[name]
                    if rr.ndim > 2:
                        rr = rr.reshape(M, -1)[:, ind]
                        rrz = rrz.reshape(M, -1)[:, ind]
                    else:
                        rr = rr[:, ind]
                        rrz = rrz[:, ind]
                    rr_d = xp.asarray(rr)
                    rrz_d = xp.asarray(rrz)
                    base = iname * 3 * n
                    block_a = 1j * (j0_d * (rr_d * (kpar_col_d / kz_ind_d)))
                    block_b = 1j * (j1_d * (rr_d * (-kpar_sq_col_d / kz_ind_d)))
                    block_c = 1j * (j0_d * (1j * (rrz_d * kpar_col_d)))
                    acc[base:base + n] = (w_d[:, None] * block_a).sum(axis = 0)
                    acc[base + n:base + 2 * n] = (w_d[:, None] * block_b).sum(axis = 0)
                    acc[base + 2 * n:base + 3 * n] = (w_d[:, None] * block_c).sum(axis = 0)
                return asnumpy(acc)
            y_d = xp.empty((M, 15 * n), dtype = xp.complex128)
            for iname, name in enumerate(names):
                rr = refl[name]
                rrz = reflz[name]
                if rr.ndim > 2:
                    rr = rr.reshape(M, -1)[:, ind]
                    rrz = rrz.reshape(M, -1)[:, ind]
                else:
                    rr = rr[:, ind]
                    rrz = rrz[:, ind]
                rr_d = xp.asarray(rr)
                rrz_d = xp.asarray(rrz)
                base = iname * 3 * n
                y_d[:, base:base + n] = 1j * (j0_d * (rr_d * (kpar_col_d / kz_ind_d)))
                y_d[:, base + n:base + 2 * n] = 1j * (j1_d * (rr_d * (-kpar_sq_col_d / kz_ind_d)))
                y_d[:, base + 2 * n:base + 3 * n] = 1j * (j0_d * (1j * (rrz_d * kpar_col_d)))
            return asnumpy(y_d)

        y = np.empty((M, 15 * n), dtype = complex)
        for iname, name in enumerate(names):
            rr = refl[name]
            rrz = reflz[name]
            if rr.ndim > 2:
                # outer-product case: flatten (n1, n2) -> (n1*n2). Also take [ind].
                rr = rr.reshape(M, -1)[:, ind]
                rrz = rrz.reshape(M, -1)[:, ind]
            else:
                rr = rr[:, ind]
                rrz = rrz[:, ind]
            base = iname * 3 * n
            # 1i * j0 .* (rr .* (kpar ./ kz))
            y[:, base:base + n] = 1j * (j0 * (rr * (kpar_col / kz_ind)))
            # 1i * j1 .* (rr .* (-kpar^2 ./ kz))
            y[:, base + n:base + 2 * n] = 1j * (j1 * (rr * (-kpar_sq_col / kz_ind)))
            # 1i * j0 .* (1i * rrz .* kpar)
            y[:, base + 2 * n:base + 3 * n] = 1j * (j0 * (1j * (rrz * kpar_col)))

        if weights is not None:
            # CPU reduction matches the original caller-side weighted-sum path.
            return (np.asarray(weights)[:, np.newaxis] * y).sum(axis = 0)
        return y

    def _inthankel_batch(self,
            kpar_arr: np.ndarray,
            ctx: Dict[str, Any],
            ind: np.ndarray,
            weights: Optional[np.ndarray] = None) -> np.ndarray:
        # Batched _inthankel over an array of complex kpar.  Returns (M, 15*n)
        # when ``weights`` is None, otherwise the weighted on-device sum
        # ``sum_m weights[m] * y[m, :]`` of shape (15*n,).
        kpar_arr = np.asarray(kpar_arr)
        M = kpar_arr.size
        n = len(ind)

        kpar1 = kpar_arr
        kpar2 = np.conj(kpar_arr)

        if ctx['is_subs']:
            refl1, refl1z, kz1 = self._reflection_subs_batch(kpar1, ctx)
            refl2, refl2z, kz2 = self._reflection_subs_batch(kpar2, ctx)
        elif ctx.get('same_size_refl', False):
            refl1, refl1z, kz1 = self._reflection_full_batch(kpar1, ctx)
            refl2, refl2z, kz2 = self._reflection_full_batch(kpar2, ctx)
        else:
            y = np.empty((M, 15 * n), dtype = complex)
            for m in range(M):
                y[m, :] = self._inthankel_ctx(kpar_arr[m], ctx, ind)
            return y

        kz1_full = kz1[:, ctx['kz_expand_idx']]
        kz2_full = kz2[:, ctx['kz_expand_idx']]
        kz1_ind = kz1_full[:, ind]
        kz2_ind = kz2_full[:, ind]

        r_ind = ctx['r_flat'][ind]
        arg = kpar_arr[:, np.newaxis] * r_ind[np.newaxis, :]
        h0 = hankel1(0, arg)
        h1 = hankel1(1, arg)
        h0_conj = np.conj(h0)
        h1_conj = np.conj(h1)

        # Wave 49: match MATLAB inthankel.m FP ordering exactly.
        kpar1_col = kpar1[:, np.newaxis]
        kpar2_col = kpar2[:, np.newaxis]
        kpar1_sq_col = (kpar1 ** 2)[:, np.newaxis]
        kpar2_sq_col = (kpar2 ** 2)[:, np.newaxis]

        names = ('p', 'ss', 'hs', 'sh', 'hh')

        # Lane C GPU acceleration — same pattern as _intbessel_batch.  hankel1
        # is not exposed by cupyx.scipy.special, so we evaluate it on the CPU
        # and only ship the dense (M, n_flat) elementwise/reduce work to the
        # device.
        use_gpu = layer_gpu_active(M * n)
        xp, asnumpy, _ = get_layer_xp(M * n)
        if use_gpu:
            h0_d = xp.asarray(h0)
            h1_d = xp.asarray(h1)
            h0c_d = xp.asarray(h0_conj)
            h1c_d = xp.asarray(h1_conj)
            kpar1_col_d = xp.asarray(kpar1_col)
            kpar2_col_d = xp.asarray(kpar2_col)
            kpar1_sq_col_d = xp.asarray(kpar1_sq_col)
            kpar2_sq_col_d = xp.asarray(kpar2_sq_col)
            kz1_ind_d = xp.asarray(kz1_ind)
            kz2_ind_d = xp.asarray(kz2_ind)
            if weights is not None:
                w_d = xp.asarray(weights)
                acc = xp.zeros(15 * n, dtype = xp.complex128)
                for iname, name in enumerate(names):
                    rr1 = refl1[name]
                    rr1z = refl1z[name]
                    rr2 = refl2[name]
                    rr2z = refl2z[name]
                    if rr1.ndim > 2:
                        rr1 = rr1.reshape(M, -1)[:, ind]
                        rr1z = rr1z.reshape(M, -1)[:, ind]
                        rr2 = rr2.reshape(M, -1)[:, ind]
                        rr2z = rr2z.reshape(M, -1)[:, ind]
                    else:
                        rr1 = rr1[:, ind]
                        rr1z = rr1z[:, ind]
                        rr2 = rr2[:, ind]
                        rr2z = rr2z[:, ind]
                    rr1_d = xp.asarray(rr1)
                    rr1z_d = xp.asarray(rr1z)
                    rr2_d = xp.asarray(rr2)
                    rr2z_d = xp.asarray(rr2z)
                    base = iname * 3 * n
                    block_a = (
                        0.5j * (h0_d * (rr1_d * (kpar1_col_d / kz1_ind_d)))
                        - 0.5j * (h0c_d * (rr2_d * (kpar2_col_d / kz2_ind_d)))
                    )
                    block_b = (
                        0.5j * (h1_d * (rr1_d * (-kpar1_sq_col_d / kz1_ind_d)))
                        - 0.5j * (h1c_d * (rr2_d * (-kpar2_sq_col_d / kz2_ind_d)))
                    )
                    block_c = (
                        0.5j * (h0_d * (1j * (rr1z_d * kpar1_col_d)))
                        - 0.5j * (h0c_d * (1j * (rr2z_d * kpar2_col_d)))
                    )
                    acc[base:base + n] = (w_d[:, None] * block_a).sum(axis = 0)
                    acc[base + n:base + 2 * n] = (w_d[:, None] * block_b).sum(axis = 0)
                    acc[base + 2 * n:base + 3 * n] = (w_d[:, None] * block_c).sum(axis = 0)
                return asnumpy(acc)
            y_d = xp.empty((M, 15 * n), dtype = xp.complex128)
            for iname, name in enumerate(names):
                rr1 = refl1[name]
                rr1z = refl1z[name]
                rr2 = refl2[name]
                rr2z = refl2z[name]
                if rr1.ndim > 2:
                    rr1 = rr1.reshape(M, -1)[:, ind]
                    rr1z = rr1z.reshape(M, -1)[:, ind]
                    rr2 = rr2.reshape(M, -1)[:, ind]
                    rr2z = rr2z.reshape(M, -1)[:, ind]
                else:
                    rr1 = rr1[:, ind]
                    rr1z = rr1z[:, ind]
                    rr2 = rr2[:, ind]
                    rr2z = rr2z[:, ind]
                rr1_d = xp.asarray(rr1)
                rr1z_d = xp.asarray(rr1z)
                rr2_d = xp.asarray(rr2)
                rr2z_d = xp.asarray(rr2z)
                base = iname * 3 * n
                y_d[:, base:base + n] = (
                    0.5j * (h0_d * (rr1_d * (kpar1_col_d / kz1_ind_d)))
                    - 0.5j * (h0c_d * (rr2_d * (kpar2_col_d / kz2_ind_d)))
                )
                y_d[:, base + n:base + 2 * n] = (
                    0.5j * (h1_d * (rr1_d * (-kpar1_sq_col_d / kz1_ind_d)))
                    - 0.5j * (h1c_d * (rr2_d * (-kpar2_sq_col_d / kz2_ind_d)))
                )
                y_d[:, base + 2 * n:base + 3 * n] = (
                    0.5j * (h0_d * (1j * (rr1z_d * kpar1_col_d)))
                    - 0.5j * (h0c_d * (1j * (rr2z_d * kpar2_col_d)))
                )
            return asnumpy(y_d)

        y = np.empty((M, 15 * n), dtype = complex)
        for iname, name in enumerate(names):
            rr1 = refl1[name]
            rr1z = refl1z[name]
            rr2 = refl2[name]
            rr2z = refl2z[name]
            if rr1.ndim > 2:
                rr1 = rr1.reshape(M, -1)[:, ind]
                rr1z = rr1z.reshape(M, -1)[:, ind]
                rr2 = rr2.reshape(M, -1)[:, ind]
                rr2z = rr2z.reshape(M, -1)[:, ind]
            else:
                rr1 = rr1[:, ind]
                rr1z = rr1z[:, ind]
                rr2 = rr2[:, ind]
                rr2z = rr2z[:, ind]
            base = iname * 3 * n
            y[:, base:base + n] = (
                0.5j * (h0 * (rr1 * (kpar1_col / kz1_ind)))
                - 0.5j * (h0_conj * (rr2 * (kpar2_col / kz2_ind)))
            )
            y[:, base + n:base + 2 * n] = (
                0.5j * (h1 * (rr1 * (-kpar1_sq_col / kz1_ind)))
                - 0.5j * (h1_conj * (rr2 * (-kpar2_sq_col / kz2_ind)))
            )
            y[:, base + 2 * n:base + 3 * n] = (
                0.5j * (h0 * (1j * (rr1z * kpar1_col)))
                - 0.5j * (h0_conj * (1j * (rr2z * kpar2_col)))
            )

        if weights is not None:
            return (np.asarray(weights)[:, np.newaxis] * y).sum(axis = 0)
        return y

    # Cache Gauss-Legendre nodes/weights per order.
    _GL_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def _gl_nodes_weights(cls, n: int) -> Tuple[np.ndarray, np.ndarray]:
        if n not in cls._GL_CACHE:
            cls._GL_CACHE[n] = np.polynomial.legendre.leggauss(n)
        return cls._GL_CACHE[n]

    @staticmethod
    def _gl_panels(a: float, b: float, n_panels: int, order: int) -> Tuple[np.ndarray, np.ndarray]:
        # Composite Gauss-Legendre nodes/weights over [a, b] split into n_panels
        # equal sub-intervals, each with `order`-point GL rule.
        nodes_ref, weights_ref = LayerStructure._gl_nodes_weights(order)
        edges = mlinspace(a, b, n_panels + 1)
        xs_all = np.empty(n_panels * order)
        ws_all = np.empty(n_panels * order)
        for i in range(n_panels):
            lo, hi = edges[i], edges[i + 1]
            xs_all[i * order:(i + 1) * order] = 0.5 * (hi - lo) * nodes_ref + 0.5 * (hi + lo)
            ws_all[i * order:(i + 1) * order] = 0.5 * (hi - lo) * weights_ref
        return xs_all, ws_all

    def _integrate_semiellipse(self,
            enei: float,
            pos: Dict[str, Any],
            n1: int,
            n_names: int) -> np.ndarray:
        # Integration along semi-ellipse in complex kr-plane.
        # Wave 33: dispatch to ODE backend (MATLAB ode45 mirror) when use_ode
        # is set, otherwise use composite Gauss-Legendre (high accuracy).
        if self.use_ode:
            return self._integrate_semiellipse_ode(enei, pos, n1, n_names)

        ctx = self._build_integrate_context(enei, pos)
        k1max = np.max(np.real(ctx['k_vals'])) + ctx['k0']
        ind_full = np.arange(len(ctx['r_flat']))
        semi = self.semi

        # semi-ellipse integrand is smooth; 4 panels x 40 order = 160 pts suffice.
        # Higher resolution available via gl_semi_panels/gl_semi_order options
        # for near-interface cases (demospecret13, sub-nm gap).
        xs, ws = self._gl_panels(0.0, np.pi, self.gl_semi_panels, self.gl_semi_order)

        kr_arr = k1max * (1 - mcos(xs) - 1j * semi * msin(xs))
        dkr_arr = k1max * (msin(xs) - 1j * semi * mcos(xs))

        return self._intbessel_batch(kr_arr, ctx, ind_full, weights = ws * dkr_arr)

    def _integrate_semiellipse_ode(self,
            enei: float,
            pos: Dict[str, Any],
            n1: int,
            n_names: int) -> np.ndarray:
        # Wave 33: solve_ivp RK45 mirroring MATLAB ode45.
        # MATLAB green.m: ode45(@(x,y) fun(...,1), [0, 1e-3, pi], y1, op)
        # with op = odeset('AbsTol', 1e-6, 'InitialStep', 1e-3).
        # MATLAB ode45 default MaxStep = abs(t1-t0)/10 (= pi/10 here).
        # Wave 48: when use_matlab_ode45 is set, use complex-y matlab_ode45
        # (bit-exact MATLAB step controller).
        ctx = self._build_integrate_context(enei, pos)
        k1max = float(np.max(np.real(ctx['k_vals'])) + ctx['k0'])
        ind_full = np.arange(len(ctx['r_flat']))
        n_state = 15 * len(ind_full)
        semi = self.semi

        if self.use_matlab_ode45:
            def rhs_c(x, y):
                kr = k1max * (1 - np.cos(x) - 1j * semi * np.sin(x))
                dkr = k1max * (np.sin(x) - 1j * semi * np.cos(x))
                val = self._intbessel_ctx(kr, ctx, ind_full)
                return dkr * val

            y0 = np.zeros(n_state, dtype = complex)
            return matlab_ode45(rhs_c, [0.0, 1e-3, np.pi], y0,
                                atol = self.atol, rtol = self.rtol,
                                initial_step = self.initial_step)

        def rhs(x, y):
            kr = k1max * (1 - np.cos(x) - 1j * semi * np.sin(x))
            dkr = k1max * (np.sin(x) - 1j * semi * np.cos(x))
            val = self._intbessel_ctx(kr, ctx, ind_full)
            out = dkr * val
            return np.concatenate([out.real, out.imag])

        y0 = np.zeros(2 * n_state)
        sol = solve_ivp(rhs, (0.0, np.pi), y0, method = 'RK45',
                        atol = self.atol, rtol = self.rtol,
                        first_step = self.initial_step,
                        max_step = np.pi / 10,
                        t_eval = [np.pi])
        y_end = sol.y[:, -1]
        return y_end[:n_state] + 1j * y_end[n_state:]

    def _integrate_real(self,
            enei: float,
            pos: Dict[str, Any],
            ind: np.ndarray,
            n: int,
            n_names: int) -> np.ndarray:
        # Real kr-axis integration. Original ODE went x: 1 -> 1e-10.
        # We integrate forward over [1e-10, 1] with composite GL (more panels
        # near the small-x end where integrand oscillates) and negate.
        if self.use_ode:
            return self._integrate_real_ode(enei, pos, ind, n, n_names)

        ctx = self._build_integrate_context(enei, pos)
        k1max = np.max(np.real(ctx['k_vals'])) + ctx['k0']

        # Use logarithmic panel boundaries to capture oscillations near x->0.
        # Break [1e-10, 1] into panels with geometrically increasing widths.
        order = self.gl_real_order
        n_panels = self.gl_real_panels
        edges = np.concatenate(([1e-10], np.logspace(-9, 0, n_panels)))
        nodes_ref, weights_ref = self._gl_nodes_weights(order)
        xs = np.empty((len(edges) - 1) * order)
        ws = np.empty((len(edges) - 1) * order)
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            xs[i * order:(i + 1) * order] = 0.5 * (hi - lo) * nodes_ref + 0.5 * (hi + lo)
            ws[i * order:(i + 1) * order] = 0.5 * (hi - lo) * weights_ref

        kr_arr = 2 * k1max / xs
        fac_arr = -2 * k1max / (xs ** 2)

        return -self._intbessel_batch(kr_arr, ctx, ind, weights = ws * fac_arr)

    def _integrate_real_ode(self,
            enei: float,
            pos: Dict[str, Any],
            ind: np.ndarray,
            n: int,
            n_names: int) -> np.ndarray:
        # Wave 33: solve_ivp RK45 mirroring MATLAB ode45 along real kr-axis.
        # MATLAB green.m: ode45(@(x,y) fun(...,2,ind2), [1, 1e-3, 1e-10], y2, op)
        # We integrate from 1 -> 1e-10 (forward in MATLAB time direction).
        ctx = self._build_integrate_context(enei, pos)
        k1max = float(np.max(np.real(ctx['k_vals'])) + ctx['k0'])
        n_state = 15 * len(ind)

        if self.use_matlab_ode45:
            def rhs_c(x, y):
                kr = 2.0 * k1max / x
                val = self._intbessel_ctx(kr, ctx, ind)
                return -2.0 * k1max * val / (x * x)

            y0 = np.zeros(n_state, dtype = complex)
            return matlab_ode45(rhs_c, [1.0, 1e-3, 1e-10], y0,
                                atol = self.atol, rtol = self.rtol,
                                initial_step = self.initial_step)

        def rhs(x, y):
            kr = 2.0 * k1max / x
            val = self._intbessel_ctx(kr, ctx, ind)
            out = -2.0 * k1max * val / (x * x)
            return np.concatenate([out.real, out.imag])

        y0 = np.zeros(2 * n_state)
        # Integrate from 1 to 1e-10 (decreasing x). first_step is signed.
        # MATLAB MaxStep = abs(1 - 1e-10)/10 ~ 0.1.
        sol = solve_ivp(rhs, (1.0, 1e-10), y0, method = 'RK45',
                        atol = self.atol, rtol = self.rtol,
                        first_step = self.initial_step,
                        max_step = 0.1,
                        t_eval = [1e-10])
        y_end = sol.y[:, -1]
        return y_end[:n_state] + 1j * y_end[n_state:]

    def _integrate_imag(self,
            enei: float,
            pos: Dict[str, Any],
            ind: np.ndarray,
            n: int,
            n_names: int) -> np.ndarray:
        if self.use_ode:
            return self._integrate_imag_ode(enei, pos, ind, n, n_names)

        ctx = self._build_integrate_context(enei, pos)
        k1max = np.max(np.real(ctx['k_vals'])) + ctx['k0']

        order = self.gl_imag_order
        n_panels = self.gl_imag_panels
        edges = np.concatenate(([1e-10], np.logspace(-9, 0, n_panels)))
        nodes_ref, weights_ref = self._gl_nodes_weights(order)
        xs = np.empty((len(edges) - 1) * order)
        ws = np.empty((len(edges) - 1) * order)
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            xs[i * order:(i + 1) * order] = 0.5 * (hi - lo) * nodes_ref + 0.5 * (hi + lo)
            ws[i * order:(i + 1) * order] = 0.5 * (hi - lo) * weights_ref

        kr_arr = 2 * k1max * (1 - 1j + 1j / xs)
        fac_arr = -2j * k1max / (xs ** 2)

        return -self._inthankel_batch(kr_arr, ctx, ind, weights = ws * fac_arr)

    def _integrate_imag_ode(self,
            enei: float,
            pos: Dict[str, Any],
            ind: np.ndarray,
            n: int,
            n_names: int) -> np.ndarray:
        # Wave 33: solve_ivp RK45 mirroring MATLAB ode45 along imag kr-axis.
        # MATLAB green.m: ode45(@(x,y) fun(...,3,ind3), [1, 1e-3, 1e-10], y3, op)
        ctx = self._build_integrate_context(enei, pos)
        k1max = float(np.max(np.real(ctx['k_vals'])) + ctx['k0'])
        n_state = 15 * len(ind)

        if self.use_matlab_ode45:
            def rhs_c(x, y):
                kr = 2.0 * k1max * (1 - 1j + 1j / x)
                val = self._inthankel_ctx(kr, ctx, ind)
                return -2j * k1max * val / (x * x)

            y0 = np.zeros(n_state, dtype = complex)
            return matlab_ode45(rhs_c, [1.0, 1e-3, 1e-10], y0,
                                atol = self.atol, rtol = self.rtol,
                                initial_step = self.initial_step)

        def rhs(x, y):
            kr = 2.0 * k1max * (1 - 1j + 1j / x)
            val = self._inthankel_ctx(kr, ctx, ind)
            out = -2j * k1max * val / (x * x)
            return np.concatenate([out.real, out.imag])

        y0 = np.zeros(2 * n_state)
        sol = solve_ivp(rhs, (1.0, 1e-10), y0, method = 'RK45',
                        atol = self.atol, rtol = self.rtol,
                        first_step = self.initial_step,
                        max_step = 0.1,
                        t_eval = [1e-10])
        y_end = sol.y[:, -1]
        return y_end[:n_state] + 1j * y_end[n_state:]

    def _intbessel_ctx(self,
            kpar: complex,
            ctx: Dict[str, Any],
            ind: np.ndarray) -> np.ndarray:
        # Fast _intbessel using precomputed ctx.
        if ctx['is_subs']:
            refl, reflz, kz = self._reflection_subs_ctx(kpar, ctx)
        else:
            pos = ctx['pos']
            refl, reflz = self.reflection(ctx['enei'], kpar, pos)
            k_vals = ctx['k_vals']
            kz = m_sqrt_c(k_vals ** 2 - kpar ** 2)
            kz = kz * np.sign(np.imag(kz + 1e-10j))

        r_flat = ctx['r_flat']
        kz_full = kz[ctx['kz_expand_idx']]
        kz_ind = kz_full[ind]

        r_ind = r_flat[ind]
        arg = np.real(kpar * r_ind) if np.isreal(kpar) else kpar * r_ind
        j0 = besselj(0, arg)
        j1 = besselj(1, arg)

        n = len(ind)
        y = np.empty(15 * n, dtype = complex)

        # Wave 49: match MATLAB intbessel.m FP ordering exactly.
        # MATLAB: y = 1i * (j0 .* (rr .* (kpar ./ kz)))   [Eq. for num(i,1)]
        # Compute factor = rr * kpar / kz first, then j0 * factor, then 1i * (...)
        kpar_sq = kpar ** 2

        names = list(refl.keys())
        for iname, name in enumerate(names):
            rr = refl[name]
            rrz = reflz[name]
            if rr.ndim > 1:
                rr = rr.ravel()
                rrz = rrz.ravel()
            rr_ind = rr[ind] if rr.size > 1 else rr
            rrz_ind = rrz[ind] if rrz.size > 1 else rrz

            base = iname * 3 * n
            # MATLAB: 1i * j0 .* (rr .* (kpar ./ kz))
            y[base:base + n] = 1j * (j0 * (rr_ind * (kpar / kz_ind)))
            # MATLAB: 1i * j1 .* (rr .* (-kpar^2 ./ kz))
            y[base + n:base + 2 * n] = 1j * (j1 * (rr_ind * (-kpar_sq / kz_ind)))
            # MATLAB: 1i * j0 .* (1i * rrz .* kpar) = -j0 .* (rrz .* kpar)
            y[base + 2 * n:base + 3 * n] = 1j * (j0 * (1j * (rrz_ind * kpar)))

        if np.isreal(kpar) and not np.iscomplexobj(y):
            return np.real(y)
        # Avoid np.all(isreal(...)) scan; default to complex.
        return y

    def _inthankel_ctx(self,
            kpar: complex,
            ctx: Dict[str, Any],
            ind: np.ndarray) -> np.ndarray:
        kpar1 = kpar
        kpar2 = np.conj(kpar)

        if ctx['is_subs']:
            refl1, refl1z, kz1 = self._reflection_subs_ctx(kpar1, ctx)
            refl2, refl2z, kz2 = self._reflection_subs_ctx(kpar2, ctx)
        else:
            pos = ctx['pos']
            refl1, refl1z = self.reflection(ctx['enei'], kpar1, pos)
            refl2, refl2z = self.reflection(ctx['enei'], kpar2, pos)
            k_vals = ctx['k_vals']
            kz1 = m_sqrt_c(k_vals ** 2 - kpar1 ** 2)
            kz1 = kz1 * np.sign(np.imag(kz1 + 1e-10j))
            kz2 = m_sqrt_c(k_vals ** 2 - kpar2 ** 2)
            kz2 = kz2 * np.sign(np.imag(kz2 + 1e-10j))

        kz1_full = kz1[ctx['kz_expand_idx']]
        kz2_full = kz2[ctx['kz_expand_idx']]
        kz1_ind = kz1_full[ind]
        kz2_ind = kz2_full[ind]

        r_ind = ctx['r_flat'][ind]
        h0 = hankel1(0, kpar * r_ind)
        h1 = hankel1(1, kpar * r_ind)
        h0_conj = np.conj(h0)
        h1_conj = np.conj(h1)

        n = len(ind)
        y = np.empty(15 * n, dtype = complex)

        # Wave 49: match MATLAB inthankel.m FP ordering exactly.
        # MATLAB: 0.5i * h0 .* (rr1 .* (kpar1 ./ kz1)) - 0.5i * conj(h0) .* (rr2 .* (kpar2 ./ kz2))
        kpar1_sq = kpar1 ** 2
        kpar2_sq = kpar2 ** 2

        names = list(refl1.keys())
        for iname, name in enumerate(names):
            rr1 = refl1[name]
            rr1z = refl1z[name]
            rr2 = refl2[name]
            rr2z = refl2z[name]
            if rr1.ndim > 1:
                rr1 = rr1.ravel()
                rr1z = rr1z.ravel()
                rr2 = rr2.ravel()
                rr2z = rr2z.ravel()
            rr1_ind = rr1[ind] if rr1.size > 1 else rr1
            rr1z_ind = rr1z[ind] if rr1z.size > 1 else rr1z
            rr2_ind = rr2[ind] if rr2.size > 1 else rr2
            rr2z_ind = rr2z[ind] if rr2z.size > 1 else rr2z

            base = iname * 3 * n
            y[base:base + n] = (
                0.5j * (h0 * (rr1_ind * (kpar1 / kz1_ind)))
                - 0.5j * (h0_conj * (rr2_ind * (kpar2 / kz2_ind)))
            )
            y[base + n:base + 2 * n] = (
                0.5j * (h1 * (rr1_ind * (-kpar1_sq / kz1_ind)))
                - 0.5j * (h1_conj * (rr2_ind * (-kpar2_sq / kz2_ind)))
            )
            y[base + 2 * n:base + 3 * n] = (
                0.5j * (h0 * (1j * (rr1z_ind * kpar1)))
                - 0.5j * (h0_conj * (1j * (rr2z_ind * kpar2)))
            )

        return y

    def _intbessel(self,
            enei: float,
            kpar: complex,
            pos: Dict[str, Any],
            ind: Optional[np.ndarray] = None) -> np.ndarray:
        # Backward-compatible: build context on demand.
        ctx = self._build_integrate_context(enei, pos)
        if ind is None:
            ind = np.arange(len(ctx['r_flat']))
        return self._intbessel_ctx(kpar, ctx, ind)

    def _inthankel(self,
            enei: float,
            kpar: complex,
            pos: Dict[str, Any],
            ind: Optional[np.ndarray] = None) -> np.ndarray:
        ctx = self._build_integrate_context(enei, pos)
        if ind is None:
            ind = np.arange(len(ctx['r_flat']))
        return self._inthankel_ctx(kpar, ctx, ind)

    def bemsolve(self,
            enei: float,
            kpar: Union[float, complex]) -> Tuple[np.ndarray, np.ndarray]:
        # MATLAB: bemsolve.m
        # Waxenegger et al., Comp. Phys. Commun. 193, 138 (2015)
        k0 = 2 * np.pi / enei

        eps_vals = np.empty(len(self.eps), dtype = complex)
        k_vals = np.empty(len(self.eps), dtype = complex)
        for i, eps_func in enumerate(self.eps):
            eps_vals[i], k_vals[i] = eps_func(enei)

        kz = m_sqrt_c(k_vals ** 2 - kpar ** 2)
        kz = kz * np.sign(np.imag(kz + 1e-10j))

        n = len(self.z)
        G0 = 2j * np.pi / kz

        # Interlayer Green function
        if n > 1:
            G = 2j * np.pi / kz[1:-1] * m_exp_c(1j * kz[1:-1] * np.abs(np.diff(self.z)))
        else:
            G = np.array([], dtype = complex)

        # Parallel surface current
        siz = 2 * n
        lhs = np.zeros((siz, siz), dtype = complex)
        rhs_mat = np.zeros((siz, siz), dtype = complex)

        i1 = np.arange(1, 2 * n, 2)  # h2 indices (0-based)
        i2 = np.arange(0, 2 * n, 2)  # h1 indices
        eq1 = np.arange(0, 2 * n, 2)  # continuity of A
        eq2 = np.arange(1, 2 * n, 2)  # continuity of dA

        # Continuity of vector potential [Eq. (13a)]
        for idx in range(n):
            lhs[eq1[idx], i1[idx]] = G0[idx + 1]
            lhs[eq1[idx], i2[idx]] = -G0[idx]

            rhs_mat[eq1[idx], i1[idx]] = -1
            rhs_mat[eq1[idx], i2[idx]] = 1

        if n > 1:
            for idx in range(n - 1):
                lhs[eq1[idx + 1], i1[idx]] = -G[idx]
                lhs[eq1[idx], i2[idx + 1]] = G[idx]

        # Continuity of derivative [Eq. (13b)]
        for idx in range(n):
            lhs[eq2[idx], i1[idx]] = 2j * np.pi
            lhs[eq2[idx], i2[idx]] = 2j * np.pi

            rhs_mat[eq2[idx], i1[idx]] = kz[idx + 1]
            rhs_mat[eq2[idx], i2[idx]] = kz[idx]

        if n > 1:
            for idx in range(n - 1):
                lhs[eq2[idx + 1], i1[idx]] = -kz[idx + 1] * G[idx]
                lhs[eq2[idx], i2[idx + 1]] = -kz[idx + 1] * G[idx]

        par = np.linalg.solve(lhs, rhs_mat)

        # Perpendicular surface current and surface charge
        siz = 4 * n
        lhs = np.zeros((siz, siz), dtype = complex)
        rhs_mat = np.zeros((siz, siz), dtype = complex)

        # Indices for surface charge (i1,j1) and current (i2,j2)
        i1 = np.arange(2, 4 * n, 4)  # sig1
        i2 = np.arange(0, 4 * n, 4)  # sig2
        j1 = np.arange(3, 4 * n, 4)  # h1
        j2 = np.arange(1, 4 * n, 4)  # h2

        eq1 = np.arange(0, 4 * n, 4)  # continuity of phi
        eq2 = np.arange(1, 4 * n, 4)  # continuity of A
        eq3 = np.arange(2, 4 * n, 4)  # continuity of D
        eq4 = np.arange(3, 4 * n, 4)  # continuity of dA

        for idx in range(n):
            # Eq. (14a): continuity of scalar potential
            lhs[eq1[idx], i1[idx]] = G0[idx + 1]
            lhs[eq1[idx], i2[idx]] = -G0[idx]
            rhs_mat[eq1[idx], i1[idx]] = -1
            rhs_mat[eq1[idx], i2[idx]] = 1

            # Eq. (14b): continuity of vector potential
            lhs[eq2[idx], j1[idx]] = G0[idx + 1]
            lhs[eq2[idx], j2[idx]] = -G0[idx]
            rhs_mat[eq2[idx], j1[idx]] = -1
            rhs_mat[eq2[idx], j2[idx]] = 1

            # Eq. (14c): continuity of dielectric displacement
            lhs[eq3[idx], i1[idx]] = 2j * np.pi * eps_vals[idx + 1]
            lhs[eq3[idx], i2[idx]] = 2j * np.pi * eps_vals[idx]
            lhs[eq3[idx], j1[idx]] = k0 * G0[idx + 1] * eps_vals[idx + 1]
            lhs[eq3[idx], j2[idx]] = -k0 * G0[idx] * eps_vals[idx]
            rhs_mat[eq3[idx], i1[idx]] = kz[idx + 1] * eps_vals[idx + 1]
            rhs_mat[eq3[idx], i2[idx]] = kz[idx] * eps_vals[idx]
            rhs_mat[eq3[idx], j1[idx]] = -k0 * eps_vals[idx + 1]
            rhs_mat[eq3[idx], j2[idx]] = k0 * eps_vals[idx]

            # Eq. (14d): continuity of derivative of vector potential
            lhs[eq4[idx], j1[idx]] = 2j * np.pi
            lhs[eq4[idx], j2[idx]] = 2j * np.pi
            lhs[eq4[idx], i1[idx]] = k0 * G0[idx + 1] * eps_vals[idx + 1]
            lhs[eq4[idx], i2[idx]] = -k0 * G0[idx] * eps_vals[idx]
            rhs_mat[eq4[idx], j1[idx]] = kz[idx + 1]
            rhs_mat[eq4[idx], j2[idx]] = kz[idx]
            rhs_mat[eq4[idx], i1[idx]] = -k0 * eps_vals[idx + 1]
            rhs_mat[eq4[idx], i2[idx]] = k0 * eps_vals[idx]

        if n > 1:
            for idx in range(n - 1):
                # Cross-coupling terms
                lhs[eq1[idx + 1], i1[idx]] = -G[idx]
                lhs[eq1[idx], i2[idx + 1]] = G[idx]

                lhs[eq2[idx + 1], j1[idx]] = -G[idx]
                lhs[eq2[idx], j2[idx + 1]] = G[idx]

                lhs[eq3[idx + 1], i1[idx]] = -kz[idx + 1] * eps_vals[idx + 1] * G[idx]
                lhs[eq3[idx], i2[idx + 1]] = -kz[idx + 1] * eps_vals[idx + 1] * G[idx]
                lhs[eq3[idx + 1], j1[idx]] = -k0 * eps_vals[idx + 1] * G[idx]
                lhs[eq3[idx], j2[idx + 1]] = k0 * eps_vals[idx + 1] * G[idx]

                lhs[eq4[idx + 1], j1[idx]] = -kz[idx + 1] * G[idx]
                lhs[eq4[idx], j2[idx + 1]] = -kz[idx + 1] * G[idx]
                lhs[eq4[idx + 1], i1[idx]] = -k0 * eps_vals[idx + 1] * G[idx]
                lhs[eq4[idx], i2[idx + 1]] = k0 * eps_vals[idx + 1] * G[idx]

        perp = np.linalg.solve(lhs, rhs_mat)

        return par, perp

    def tabspace(self,
            *args: Any,
            **options: Any) -> Union[Dict[str, Any], list]:
        # MATLAB: tabspace.m
        if len(args) > 0 and isinstance(args[0], (int, float, np.ndarray)):
            return self._tabspace1(args[0], args[1], args[2], **options)
        else:
            return self._tabspace2(args[0], *args[1:], **options)

    def _tabspace1(self,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray,
            **options: Any) -> Dict[str, Any]:
        # MATLAB: private/tabspace1.m
        r = np.asarray(r, dtype = float)
        z1 = np.asarray(z1, dtype = float)
        z2 = np.asarray(z2, dtype = float)

        rmod = options.get('rmod', 'log')
        zmod = options.get('zmod', 'log')

        # Slightly reduce zmin for tabulation
        zmin_orig = self.zmin
        self.zmin = 0.999 * self.zmin

        tab = {}
        # Table for radii
        tab['r'] = self._linlogspace(max(r[0], self.rmin), r[1], int(r[2]), rmod)

        if z1.size == 1:
            tab['z1'] = z1
        else:
            z1_sorted = np.sort(np.asarray(self.round_z(z1[0:2])).ravel())
            if np.abs(z1_sorted[0] - z1_sorted[1]) < 1e-3:
                z1_sorted = self._expand_z(z1_sorted)
            tab['z1'] = self._zlinlogspace(z1_sorted[0], z1_sorted[1], int(z1[2]), zmod)

        if z2.size == 1:
            tab['z2'] = z2
        else:
            z2_sorted = np.sort(np.asarray(self.round_z(z2[0:2])).ravel())
            if np.abs(z2_sorted[0] - z2_sorted[1]) < 1e-3:
                z2_sorted = self._expand_z(z2_sorted)
            tab['z2'] = self._zlinlogspace(z2_sorted[0], z2_sorted[1], int(z2[2]), zmod)

        self.zmin = zmin_orig
        return tab

    def _tabspace2(self,
            p: Any,
            *args: Any,
            **options: Any) -> list:
        # MATLAB: private/tabspace2.m
        # MATLAB API: tabspace(layer, {p, pt}, [pt_extra,] options).
        # The cell {p, pt} contributes to BOTH pos1 (z1 source) and pos2
        # (z2 source); pt_extra is added only to pos1.
        # Python convenience call tabspace(p, pt, ...) with non-list `p`
        # mirrors the cell pattern: both p and pt go to pos1=pos2.
        # If a second ComPoint-like arg is given, it acts as pt_extra.
        pt = None
        pt_extra = None
        remaining = list(args)
        if len(remaining) > 0 and not isinstance(remaining[0], str):
            pt = remaining[0]
            remaining = remaining[1:]
        if len(remaining) > 0 and not isinstance(remaining[0], str):
            pt_extra = remaining[0]
            remaining = remaining[1:]

        scale = options.get('scale', 1.05)
        nr = options.get('nr', 30)
        nz = options.get('nz', 30)
        n_layers = len(self.z) + 1

        # If user passed `(p, pt)` with single particle and ComPoint pt,
        # promote to cell-style `[p, pt]` for primary positions.
        if not isinstance(p, list):
            primary = [p]
            if pt is not None:
                primary.append(pt)
                pt = None  # consumed into primary
        else:
            primary = list(p)

        # Collect positions from primary particles/points
        positions = []
        for particle in primary:
            if hasattr(particle, 'verts'):
                positions.append(particle.verts)
            elif hasattr(particle, 'pos'):
                positions.append(particle.pos)

        pos1 = np.vstack(positions)
        pos2 = pos1.copy()

        # MATLAB tabspace2.m line 50-56: if pt arg present, pt.pos is
        # appended to pos1 (NOT pos2) once per primary particle. Mirror
        # this for the optional pt_extra arg (or pt when p was a list).
        extra = pt_extra if pt_extra is not None else pt
        if extra is not None and hasattr(extra, 'pos'):
            n_primary = len(primary)
            extra_pos = np.tile(extra.pos, (n_primary, 1))
            pos1 = np.vstack([pos1, extra_pos])

        # Get limits
        ir, iz1, iz2 = self._limits(pos1, pos2)

        # Generate tables
        tabs = []
        if isinstance(nz, (int, float)):
            nz_arr = [nz] * n_layers
        else:
            nz_arr = list(nz)

        for i1 in range(n_layers):
            for i2 in range(n_layers):
                if iz1[i1][i2] is not None:
                    # Adjust z-values
                    z1_adj, z2_adj = self._adjust(iz1[i1][i2], iz2[i1][i2], scale = scale)

                    # Scale radii
                    r_range = ir[i1][i2]
                    r_mean = np.mean(r_range)
                    r_range = np.maximum(
                        [r_mean + 0.5 * scale * (r_range[0] - r_range[1]),
                         r_mean - 0.5 * scale * (r_range[0] - r_range[1])], 0)

                    tab = self._tabspace1(
                        np.array([r_range[0], r_range[1], nr]),
                        np.array([z1_adj[0], z1_adj[1], nz_arr[i1]]),
                        np.array([z2_adj[0], z2_adj[1], nz_arr[i2]]) if len(z2_adj) > 1
                        else np.array([z2_adj[0]]),
                        **options)
                    tabs.append(tab)

        return tabs

    def _adjust(self,
            z1: np.ndarray,
            z2: np.ndarray,
            scale: float = 1.05,
            range_mode: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        # MATLAB: private/adjust.m
        i1, _ = self.indlayer(np.array([z1[0]]))
        i2, _ = self.indlayer(np.array([z2[0]]))
        i1 = int(np.asarray(i1).flat[0])
        i2 = int(np.asarray(i2).flat[0])

        n = len(self.z) + 1
        zlayer = np.empty((n, 2))
        for i in range(n):
            zlayer[i, 0] = self.z[i] if i < len(self.z) else -np.inf
            zlayer[i, 1] = self.z[i - 1] if i > 0 else np.inf

        z1 = self._enlarge(z1, zlayer[i1 - 1, :], scale, range_mode)
        z2 = self._enlarge(z2, zlayer[i2 - 1, :], scale, range_mode)

        # Handle uppermost and lowermost layer
        if i1 == 1 and i2 == 1:
            z1 = z1 + z2 - self.z[0]
            z2 = np.array([self.z[0] + 1e-10])
        elif i1 == n and i2 == n:
            z1 = z1 + z2 - self.z[-1]
            z2 = np.array([self.z[-1] - 1e-10])

        return z1, z2

    def _enlarge(self,
            z: np.ndarray,
            zlayer: np.ndarray,
            scale: float,
            range_mode: Optional[str]) -> np.ndarray:
        z = np.asarray(z, dtype = float).copy()
        if np.abs(z[0] - z[1]) < 1e-10:
            z = np.array([z[0] - 1e-2, z[0] + 1e-2])
        else:
            mean_z = np.mean(z)
            z = np.array([mean_z + 0.5 * scale * (z[0] - z[1]),
                          mean_z - 0.5 * scale * (z[0] - z[1])])

        # NOTE: use <= / >= so that values exactly on a layer boundary are
        # nudged inward by 1e-10. Without this nudge the downstream
        # _zlinlogspace evaluates log10(zmin - obj.z[i]) = log10(0) = -inf and
        # produces NaN tabulation grids (Issue 3 — metal substrate trigger
        # case where the particle bottom sits exactly on the interface).
        if z[0] <= zlayer[0]:
            z[0] = zlayer[0] + 1e-10
        if z[1] >= zlayer[1]:
            z[1] = zlayer[1] - 1e-10

        if range_mode == 'full':
            if not np.isinf(zlayer[0]):
                z[0] = zlayer[0] + 1e-10
            if not np.isinf(zlayer[1]):
                z[1] = zlayer[1] - 1e-10

        return z

    def _expand_z(self,
            z: np.ndarray) -> np.ndarray:
        _, ind = self.mindist(np.array([z[0]]))
        ind = int(ind) - 1
        z_out = np.sort([z[0] + np.sign(z[0] - self.z[ind]) * 0.1 * self.zmin, z[1]])
        return z_out

    def _limits(self,
            pos1: np.ndarray,
            pos2: np.ndarray) -> Tuple[list, list, list]:
        # Radial distance between points
        diff_xy = pos1[:, 0:2][:, np.newaxis, :] - pos2[:, 0:2][np.newaxis, :, :]
        r = msqrt(np.sum(diff_xy ** 2, axis = 2))

        z1 = pos1[:, 2]
        ind1, _ = self.indlayer(z1)
        z2 = pos2[:, 2]
        ind2, _ = self.indlayer(z2)

        n_layers = len(self.z) + 1
        ir = [[None] * n_layers for _ in range(n_layers)]
        iz1 = [[None] * n_layers for _ in range(n_layers)]
        iz2 = [[None] * n_layers for _ in range(n_layers)]

        for i1 in range(1, n_layers + 1):
            for i2 in range(1, n_layers + 1):
                mask1 = (ind1 == i1)
                mask2 = (ind2 == i2)
                if np.any(mask1) and np.any(mask2):
                    r_sub = r[mask1][:, mask2]
                    ir[i1 - 1][i2 - 1] = np.array([np.min(r_sub), np.max(r_sub)])
                    iz1[i1 - 1][i2 - 1] = np.array([np.min(z1[mask1]), np.max(z1[mask1])])
                    iz2[i1 - 1][i2 - 1] = np.array([np.min(z2[mask2]), np.max(z2[mask2])])

        return ir, iz1, iz2

    @staticmethod
    def _logspace_matlab(a: float, b: float, n: int) -> np.ndarray:
        # MATLAB-compatible logspace: logspace(-Inf, b, 2) returns [0, 10^b].
        # numpy.logspace returns NaN for -inf endpoints; handle explicitly.
        if np.isinf(a) or np.isinf(b):
            lin = np.empty(n)
            lin[0] = a
            lin[-1] = b
            if n > 2:
                lin[1:-1] = np.nan
            out = np.empty(n)
            for i in range(n):
                if np.isneginf(lin[i]):
                    out[i] = 0.0
                elif np.isposinf(lin[i]):
                    out[i] = np.inf
                elif np.isnan(lin[i]):
                    out[i] = np.nan
                else:
                    out[i] = 10.0 ** lin[i]
            return out
        return np.logspace(a, b, n)

    @staticmethod
    def _linlogspace(xmin: float,
            xmax: float,
            n: int,
            key: str,
            x0: float = 0.0) -> np.ndarray:
        if key == 'lin':
            return mlinspace(xmin, xmax, n)
        else:
            return x0 + LayerStructure._logspace_matlab(
                mlog10(xmin - x0), mlog10(xmax - x0), n)

    def _zlinlogspace(self,
            zmin: float,
            zmax: float,
            n: int,
            key: str) -> np.ndarray:
        if key == 'lin':
            return mlinspace(zmin, zmax, n)

        medium, _ = self.indlayer(np.array([zmin]))
        medium = int(np.atleast_1d(medium).ravel()[0])

        if medium == 1:
            # Upper layer
            return self.z[0] + LayerStructure._logspace_matlab(
                mlog10(zmin - self.z[0]),
                mlog10(zmax - self.z[0]), n)
        elif medium == len(self.z) + 1:
            # Lower layer
            z = self.z[-1] - LayerStructure._logspace_matlab(
                mlog10(self.z[-1] - zmax),
                mlog10(self.z[-1] - zmin), n)
            return z[::-1]
        else:
            # Intermediate layer
            zup = self.z[medium - 2]
            zlo = self.z[medium - 1]
            zmin_scaled = 2 * (zmin - zlo) / (zup - zlo) - 1
            zmax_scaled = 2 * (zmax - zlo) / (zup - zlo) - 1
            z_scaled = mtanh(mlinspace(np.arctanh(zmin_scaled), np.arctanh(zmax_scaled), n))
            return 0.5 * (zup + zlo) + 0.5 * z_scaled * (zup - zlo)

    @staticmethod
    def options(**kwargs: Any) -> Dict[str, Any]:
        opt = {
            'ztol': 2e-2,
            'rmin': 1e-2,
            'zmin': 1e-2,
            'semi': 0.1,
            'ratio': 2.0,
            'atol': 1e-6,
            'initial_step': 1e-3,
        }
        opt.update(kwargs)
        return opt

    def __repr__(self) -> str:
        return 'LayerStructure(n_layers={}, z={})'.format(len(self.z), self.z.tolist())
