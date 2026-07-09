"""
Spherical harmonics and related functions.

MATLAB reference: Mie/spharm.m, Mie/sphtable.m, Mie/vecspharm.m
"""

import numpy as np
from scipy.special import lpmv, factorial


def sphtable(lmax, key='z'):
    """Table of spherical harmonic degrees and orders.

    MATLAB: sphtable.m

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree.
    key : str
        'z' (default): only m = -1, 0, 1 per l (3*lmax entries).
        'full': m = -l..l per l (sum(2l+1) entries).

    Returns
    -------
    ltab : ndarray, shape (N,), dtype int64
        Table of spherical harmonic degrees.
    mtab : ndarray, shape (N,), dtype int64
        Table of spherical harmonic orders.
    """
    ltab_list = []
    mtab_list = []
    for l in range(1, lmax + 1):
        if key == 'z':
            m_vals = np.array([-1, 0, 1])
        else:
            m_vals = np.arange(-l, l + 1)
        ltab_list.append(np.full(len(m_vals), l))
        mtab_list.append(m_vals)
    ltab = np.hstack(ltab_list).astype(np.int64)
    mtab = np.hstack(mtab_list).astype(np.int64)
    return ltab, mtab


def spharm(ltab, mtab, theta, phi):
    """Spherical harmonics Y_l^m(theta, phi).

    MATLAB: spharm.m

    Uses the convention:
        Y_l^m = c * P_l^|m|(cos(theta)) * exp(i*|m|*phi)
        Y_l^{-m} = (-1)^m * conj(Y_l^m)

    where c = sqrt((2l+1)/(4*pi) * (l-|m|)!/(l+|m|)!)

    Parameters
    ----------
    ltab : array_like
        Spherical harmonic degrees.
    mtab : array_like
        Spherical harmonic orders.
    theta : array_like
        Polar angles (paired with phi).
    phi : array_like
        Azimuthal angles (paired with theta).

    Returns
    -------
    y : ndarray, shape (len(ltab), len(theta)), complex128
        Spherical harmonics evaluated at each (theta, phi) pair.
    """
    ltab = np.asarray(ltab).ravel()
    mtab = np.asarray(mtab).ravel()
    theta = np.asarray(theta, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()

    n_lm = len(ltab)
    n_pts = len(theta)
    y = np.zeros((n_lm, n_pts), dtype=complex)

    cos_theta = np.cos(theta)

    for l_val in np.unique(ltab):
        # Compute full associated Legendre functions for this l
        # scipy.special.lpmv(m, l, x) gives P_l^m(x) (unnormalized)
        indices = np.where(ltab == l_val)[0]
        # Filter indices where |m| <= l
        valid = indices[np.abs(mtab[indices]) <= l_val]

        for i in valid:
            m = mtab[i]
            abs_m = abs(m)
            # Associated Legendre polynomial P_l^|m|(cos(theta))
            p = lpmv(abs_m, l_val, cos_theta)
            # Normalization
            c = np.sqrt(
                (2 * l_val + 1) / (4 * np.pi)
                * factorial(l_val - abs_m, exact=True)
                / factorial(l_val + abs_m, exact=True)
            )
            # Y_l^|m|
            y[i, :] = c * p * np.exp(1j * abs_m * phi)
            # Correct for negative m
            if m < 0:
                y[i, :] = (-1) ** abs(m) * np.conj(y[i, :])

    return y


def vecspharm(ltab, mtab, theta, phi):
    """Vector spherical harmonics X_lm (Jackson eq. 9.119).

    MATLAB: vecspharm.m

    X_lm = (1/sqrt(l(l+1))) * L x Y_lm

    where L is the angular momentum operator.

    Parameters
    ----------
    ltab : array_like
        Spherical harmonic degrees.
    mtab : array_like
        Spherical harmonic orders.
    theta : array_like
        Polar angles.
    phi : array_like
        Azimuthal angles.

    Returns
    -------
    x : ndarray, shape (n_lm, n_pts, 3), complex128
        Vector spherical harmonics.
    y : ndarray, shape (n_lm, n_pts), complex128
        Scalar spherical harmonics (same as spharm output).
    """
    ltab = np.asarray(ltab).ravel()
    mtab = np.asarray(mtab).ravel()

    l = ltab.astype(float)
    m = mtab.astype(float)

    # Scalar spherical harmonics
    y = spharm(ltab, mtab, theta, phi)
    yp = spharm(ltab, mtab + 1, theta, phi)
    ym = spharm(ltab, mtab - 1, theta, phi)

    n_pts = y.shape[1]
    norm = 1.0 / np.sqrt(l * (l + 1))  # (n_lm,)

    # L+ Y_lm = sqrt((l-m)(l+m+1)) Y_l^{m+1}
    lpy = (norm * np.sqrt((l - m) * (l + m + 1)))[:, np.newaxis] * yp
    # L- Y_lm = sqrt((l+m)(l-m+1)) Y_l^{m-1}
    lmy = (norm * np.sqrt((l + m) * (l - m + 1)))[:, np.newaxis] * ym
    # Lz Y_lm = m Y_lm
    lzy = (norm * m)[:, np.newaxis] * y

    # Vector spherical harmonics in (r, theta, phi) basis
    # X = (L+ Y * [1, -i, 0] + L- Y * [1, i, 0]) / 2 + Lz Y * [0, 0, 1]
    # This gives components in (x, y, z)-like spherical basis
    x = np.zeros((len(ltab), n_pts, 3), dtype=complex)
    x[:, :, 0] = lpy / 2 + lmy / 2          # component 1
    x[:, :, 1] = -1j * lpy / 2 + 1j * lmy / 2  # component 2
    x[:, :, 2] = lzy                          # component 3

    return x, y
