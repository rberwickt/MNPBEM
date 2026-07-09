"""
Mie-Gans theory for ellipsoidal particles (quasistatic approximation).

MATLAB reference: Mie/@miegans/
"""

import numpy as np
from scipy.integrate import quad


def _matlab_quad(f, a, b, tol=1e-6):
    """MATLAB-compatible adaptive Simpson quadrature (port of MATLAB's quad.m,
    based on Gander & Gautschi, BIT 40 (2000) 84-101).
    """
    # Initialize with three unequal subintervals.
    h = 0.13579 * (b - a)
    x = np.array([a, a + h, a + 2 * h, (a + b) / 2, b - 2 * h, b - h, b])
    y = np.array([f(xi) for xi in x])
    if not np.isfinite(y[0]):
        y[0] = f(a + np.finfo(float).eps * (b - a))
    if not np.isfinite(y[6]):
        y[6] = f(b - np.finfo(float).eps * (b - a))
    hmin = np.spacing(b - a) / 1024.0

    def quadstep(a_, b_, fa, fc, fb, tol_, depth):
        if depth > 200:
            return (b_ - a_) / 6 * (fa + 4 * fc + fb)
        h_ = b_ - a_
        c_ = 0.5 * (a_ + b_)
        d_ = 0.5 * (a_ + c_)
        e_ = 0.5 * (c_ + b_)
        fd = f(d_)
        fe = f(e_)
        Q1 = (h_ / 6) * (fa + 4 * fc + fb)
        Q2 = (h_ / 12) * (fa + 4 * fd + 2 * fc + 4 * fe + fb)
        Q = Q2 + (Q2 - Q1) / 15.0
        if not np.isfinite(Q):
            return Q
        if abs(Q2 - Q) <= tol_:
            return Q
        if abs(h_) < hmin or c_ == a_ or c_ == b_:
            return Q
        return (quadstep(a_, c_, fa, fd, fc, tol_, depth + 1)
                + quadstep(c_, b_, fc, fe, fb, tol_, depth + 1))

    Q1 = quadstep(x[0], x[2], y[0], y[1], y[2], tol, 0)
    Q2 = quadstep(x[2], x[4], y[2], y[3], y[4], tol, 0)
    Q3 = quadstep(x[4], x[6], y[4], y[5], y[6], tol, 0)
    return Q1 + Q2 + Q3


def _get_eps_value(eps_func, enei):
    """Extract dielectric constant from eps_func(enei).

    Handles both:
    - Functions returning scalar/array eps value
    - Functions returning (eps, k) tuple
    """
    result = eps_func(enei)
    if isinstance(result, tuple):
        return result[0]
    return result


class MieGans(object):
    """Mie-Gans theory for ellipsoidal particle (quasistatic approximation).

    MATLAB: @miegans

    Parameters
    ----------
    epsin : callable
        Dielectric function inside ellipsoid. epsin(enei) -> eps or (eps, k).
    epsout : callable
        Dielectric function outside ellipsoid. epsout(enei) -> eps or (eps, k).
    ax : ndarray, shape (3,)
        Ellipsoid semi-axis diameters (full axes, not semi-axes) in nm.
    """

    def __init__(self, epsin, epsout, ax):
        self.epsin = epsin
        self.epsout = epsout
        self.ax = np.asarray(ax, dtype=float)
        self._compute_depolarization()

    def _compute_depolarization(self):
        """Compute depolarization factors L1, L2, L3.

        MATLAB: @miegans/init.m
        van de Hulst, Sec. 6.32

        MATLAB uses full axes (diameters) in integrand and tol=1e-8 with
        adaptive Simpson; we mirror the same algorithm and parameterization
        so that the L_i values are bit-comparable.
        """
        a, b, c = self.ax  # MATLAB feeds full axes directly

        def f1(s):
            return a * b * c / 2 / ((s + a**2)**1.5 * np.sqrt(s + b**2) * np.sqrt(s + c**2))

        def f2(s):
            return a * b * c / 2 / (np.sqrt(s + a**2) * (s + b**2)**1.5 * np.sqrt(s + c**2))

        def f3(s):
            return a * b * c / 2 / (np.sqrt(s + a**2) * np.sqrt(s + b**2) * (s + c**2)**1.5)

        upper = 1e5 * max(self.ax)
        tol = 1e-8
        self._L1 = _matlab_quad(f1, 0.0, upper, tol=tol)
        self._L2 = _matlab_quad(f2, 0.0, upper, tol=tol)
        self._L3 = _matlab_quad(f3, 0.0, upper, tol=tol)

    def _polarizabilities(self, enei):
        """Compute per-axis polarizabilities.

        MATLAB: @miegans/scattering.m lines 14-21
        """
        epsb = _get_eps_value(self.epsout, np.array([0.0]))
        if isinstance(epsb, np.ndarray):
            epsb = complex(epsb.ravel()[0])
        epsi = _get_eps_value(self.epsin, enei)
        epsz = epsi / epsb

        vol = 4 * np.pi / 3 * np.prod(self.ax / 2)
        a1 = vol / (4 * np.pi) / (self._L1 + 1.0 / (epsz - 1))
        a2 = vol / (4 * np.pi) / (self._L2 + 1.0 / (epsz - 1))
        a3 = vol / (4 * np.pi) / (self._L3 + 1.0 / (epsz - 1))

        nb = np.sqrt(epsb)
        k = 2 * np.pi / enei * nb
        return a1, a2, a3, k

    def extinction(self, enei, pol=None):
        """Extinction cross section.

        MATLAB: @miegans/extinction.m
        If pol is None, returns orientation-averaged extinction.
        """
        if pol is None:
            return (self.extinction(enei, np.array([1., 0., 0.]))
                  + self.extinction(enei, np.array([0., 1., 0.]))
                  + self.extinction(enei, np.array([0., 0., 1.]))) / 3.0
        return self.scattering(enei, pol) + self.absorption(enei, pol)

    def scattering(self, enei, pol):
        """Scattering cross section.

        MATLAB: @miegans/scattering.m
        """
        enei = np.asarray(enei, dtype=float)
        pol = np.asarray(pol, dtype=float)
        a1, a2, a3, k = self._polarizabilities(enei)
        sca = (8 * np.pi / 3 * k**4
               * (np.abs(a1 * pol[0])**2
                  + np.abs(a2 * pol[1])**2
                  + np.abs(a3 * pol[2])**2))
        return np.real(sca)

    def absorption(self, enei, pol):
        """Absorption cross section.

        MATLAB: @miegans/absorption.m
        """
        enei = np.asarray(enei, dtype=float)
        pol = np.asarray(pol, dtype=float)
        a1, a2, a3, k = self._polarizabilities(enei)
        abso = 4 * np.pi * k * np.imag(a1 * pol[0] + a2 * pol[1] + a3 * pol[2])
        return np.real(abso)

    def __repr__(self):
        return "MieGans(ax={})".format(self.ax.tolist())
