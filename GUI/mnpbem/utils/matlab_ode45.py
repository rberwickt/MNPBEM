"""MATLAB ode45 1:1 reimplementation in Python (Wave 48).

This module mirrors MATLAB's ode45.m step controller, error norm, and
initial-step logic exactly so that ODE-driven Sommerfeld integrations match
MATLAB bit-similar (within machine ULP for elementary ops).

Reference: /usr/local/MATLAB/R2025b/toolbox/matlab/funfun/ode45.m
"""

import numpy as np

# Dormand-Prince DP4(5) coefficients (identical to MATLAB ode45.m)
_A2 = 1.0 / 5.0
_A3 = 3.0 / 10.0
_A4 = 4.0 / 5.0
_A5 = 8.0 / 9.0

_B11 = 1.0 / 5.0
_B21 = 3.0 / 40.0
_B31 = 44.0 / 45.0
_B41 = 19372.0 / 6561.0
_B51 = 9017.0 / 3168.0
_B61 = 35.0 / 384.0
_B22 = 9.0 / 40.0
_B32 = -56.0 / 15.0
_B42 = -25360.0 / 2187.0
_B52 = -355.0 / 33.0
_B33 = 32.0 / 9.0
_B43 = 64448.0 / 6561.0
_B53 = 46732.0 / 5247.0
_B63 = 500.0 / 1113.0
_B44 = -212.0 / 729.0
_B54 = 49.0 / 176.0
_B64 = 125.0 / 192.0
_B55 = -5103.0 / 18656.0
_B65 = -2187.0 / 6784.0
_B66 = 11.0 / 84.0

_E1 = 71.0 / 57600.0
_E3 = -71.0 / 16695.0
_E4 = 71.0 / 1920.0
_E5 = -17253.0 / 339200.0
_E6 = 22.0 / 525.0
_E7 = -1.0 / 40.0

_POW = 1.0 / 5.0


def _eps(x):
    # MATLAB eps(x) returns the spacing to next double (= np.spacing for finite x).
    return np.spacing(np.abs(x)) if x != 0.0 else np.spacing(1.0) * 0.0 + np.finfo(float).eps * 0.0 + 2.220446049250313e-16


def matlab_ode45(rhs, tspan, y0, *, atol=1e-6, rtol=1e-3, initial_step=None,
                 max_step=None, min_step=0.0):
    """MATLAB ode45 1:1 reimplementation.

    Parameters
    ----------
    rhs : callable
        rhs(t, y) -> dy/dt. Operates on real ndarray (shape (neq,)).
    tspan : array_like
        [t0, ..., tfinal]. Output is returned only at the final time.
    y0 : ndarray
        Initial state (real, 1D).
    atol : float or ndarray
        Absolute tolerance (matches MATLAB AbsTol).
    rtol : float
        Relative tolerance (matches MATLAB RelTol).
    initial_step : float or None
        MATLAB InitialStep. If None, computed from y'(t0).
    max_step : float or None
        MATLAB MaxStep. Default = 0.1 * |tfinal-t0|.
    min_step : float
        MATLAB MinStep. Default 0.

    Returns
    -------
    y_final : ndarray
        Solution at tspan[-1] (same shape as y0).
    """
    tspan = np.asarray(tspan, dtype=float).ravel()
    y0 = np.asarray(y0).ravel()
    if np.iscomplexobj(y0):
        y0 = y0.astype(complex)
    else:
        y0 = y0.astype(float)
    t0 = float(tspan[0])
    tfinal = float(tspan[-1])
    neq = y0.size

    htspan = abs(float(tspan[1]) - t0) if tspan.size >= 2 else abs(tfinal - t0)
    tdir = np.sign(tfinal - t0)
    tlen = abs(tfinal - t0)

    rtol = float(rtol)
    eps_d = np.finfo(float).eps
    if rtol < 100.0 * eps_d:
        rtol = 100.0 * eps_d

    atol_arr = np.atleast_1d(np.asarray(atol, dtype=float))
    threshold = atol_arr / rtol  # scalar or (neq,)

    safehmax = 16.0 * eps_d * max(abs(t0), abs(tfinal))
    if max_step is None:
        hmax = max(0.1 * tlen, safehmax)
    else:
        hmax = min(tlen, float(max_step))

    userhmin = float(min_step)
    htry = None if initial_step is None else abs(float(initial_step))

    t = t0
    y = y0.copy()

    f0 = rhs(t, y)
    f1 = f0.copy()

    # Compute initial step
    if htry is None:
        absh = min(hmax, htspan)
        denom = np.maximum(np.abs(y), threshold)
        rh = np.linalg.norm(f0 / denom, ord=np.inf) / (0.8 * rtol ** _POW)
        if absh * rh > 1.0:
            absh = 1.0 / rh
        absh = max(absh, userhmin)
    else:
        absh = min(hmax, max(userhmin, htry))

    done = False

    while not done:
        # Recompute hmin/hmax at current t (MATLAB lines 272-275)
        tinystep = 16.0 * np.spacing(abs(t) if t != 0.0 else 1.0)
        # Note: MATLAB eps(0) ~ 4.94e-324 but tinystep at t=0 is 16*eps(0).
        # Use np.spacing(t) which returns 4.94e-324 for t=0 - this matches.
        if t == 0.0:
            tinystep = 16.0 * 4.9406564584124654e-324
        else:
            tinystep = 16.0 * np.spacing(abs(t))
        hmin = max(tinystep, userhmin)
        hmax_eff = max(tinystep, hmax)
        absh = min(hmax_eff, max(hmin, absh))
        h = tdir * absh

        # Stretch step within 10% of tfinal
        if 1.1 * absh >= abs(tfinal - t):
            h = tfinal - t
            absh = abs(h)
            done = True

        nofailed = True
        while True:
            y2 = y + h * (_B11 * f1)
            t2 = t + h * _A2
            f2 = rhs(t2, y2)

            y3 = y + h * (_B21 * f1 + _B22 * f2)
            t3 = t + h * _A3
            f3 = rhs(t3, y3)

            y4 = y + h * (_B31 * f1 + _B32 * f2 + _B33 * f3)
            t4 = t + h * _A4
            f4 = rhs(t4, y4)

            y5 = y + h * (_B41 * f1 + _B42 * f2 + _B43 * f3 + _B44 * f4)
            t5 = t + h * _A5
            f5 = rhs(t5, y5)

            y6 = y + h * (_B51 * f1 + _B52 * f2 + _B53 * f3 + _B54 * f4 + _B55 * f5)
            t6 = t + h
            f6 = rhs(t6, y6)

            tnew = t + h
            if done:
                tnew = tfinal  # hit endpoint exactly
            h = tnew - t  # purify h

            ynew = y + h * (_B61 * f1 + _B63 * f3 + _B64 * f4 + _B65 * f5 + _B66 * f6)
            f7 = rhs(tnew, ynew)

            # Error estimate (MATLAB normcontrol=off path)
            fE = f1 * _E1 + f3 * _E3 + f4 * _E4 + f5 * _E5 + f6 * _E6 + f7 * _E7
            denom = np.maximum(np.maximum(np.abs(y), np.abs(ynew)), threshold)
            err = absh * np.linalg.norm(fE / denom, ord=np.inf)

            if err > rtol:
                # Failed step
                if absh <= hmin:
                    raise RuntimeError(
                        f"matlab_ode45: integration tolerance not met at t={t:e}, hmin={hmin:e}"
                    )
                if nofailed:
                    nofailed = False
                    absh = max(hmin, absh * max(0.1, 0.8 * (rtol / err) ** _POW))
                else:
                    absh = max(hmin, 0.5 * absh)
                h = tdir * absh
                done = False
            else:
                break

        if done:
            t = tnew
            y = ynew
            break

        # Compute next h (no failures path)
        if nofailed:
            temp = 1.25 * (err / rtol) ** _POW
            if temp > 0.2:
                absh = absh / temp
            else:
                absh = 5.0 * absh

        # Advance
        t = tnew
        y = ynew
        f1 = f7  # FSAL

    return y
