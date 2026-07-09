"""
Full retarded Mie theory for spherical particles.

MATLAB reference: Mie/@mieret/
"""

import numpy as np
from scipy.special import jv, yv, factorial
from scipy.special import kv as besselk

from .spherical_harmonics import sphtable
from ..misc.units import FINE, BOHR, HARTREE


def _riccatibessel(z, ltab):
    """Riccati-Bessel functions.

    MATLAB: @mieret/private/riccatibessel.m
    Abramowitz & Stegun, Chapter 10.

    Parameters
    ----------
    z : complex
        Argument.
    ltab : array_like
        Angular momentum components.

    Returns
    -------
    j : ndarray
        Spherical Bessel function of first kind j_l(z).
    h : ndarray
        Spherical Hankel function h_l(z) = j_l + i*y_l.
    zjp : ndarray
        [z*j_l(z)]' derivative.
    zhp : ndarray
        [z*h_l(z)]' derivative.
    """
    ltab = np.asarray(ltab).ravel()
    lmax = int(np.max(ltab))

    # Compute for l = 1..lmax
    l_arr = np.arange(1, lmax + 1)

    # Spherical Bessel functions via half-integer Bessel functions
    # j_l(z) = sqrt(pi/(2z)) * J_{l+0.5}(z)
    # y_l(z) = sqrt(pi/(2z)) * Y_{l+0.5}(z)
    prefactor = np.sqrt(np.pi / (2 * z))

    j_vals = prefactor * jv(l_arr + 0.5, z)
    y_vals = prefactor * yv(l_arr + 0.5, z)

    # l=0 values for recurrence
    j0 = np.sin(z) / z
    y0 = -np.cos(z) / z
    h0 = j0 + 1j * y0

    h_vals = j_vals + 1j * y_vals

    # Derivatives via recurrence: [z*f_l(z)]' = z*f_{l-1}(z) - l*f_l(z)
    j_prev = np.hstack([[j0], j_vals[:-1]])
    h_prev = np.hstack([[h0], h_vals[:-1]])

    zjp_vals = z * j_prev - l_arr * j_vals
    zhp_vals = z * h_prev - l_arr * h_vals

    # Map to ltab indices (ltab values are 1-based)
    j = j_vals[ltab - 1]
    h = h_vals[ltab - 1]
    zjp = zjp_vals[ltab - 1]
    zhp = zhp_vals[ltab - 1]

    return j, h, zjp, zhp


def _miecoefficients(k, diameter, epsr, mur, ltab):
    """Mie coefficients (Bohren & Huffman 1983).

    MATLAB: @mieret/private/miecoefficients.m

    Parameters
    ----------
    k : complex
        Wavevector outside sphere.
    diameter : float
        Sphere diameter.
    epsr : complex
        Relative dielectric constant of sphere.
    mur : float
        Relative magnetic permeability of sphere.
    ltab : array_like
        Angular momentum components.

    Returns
    -------
    a, b, c, d : ndarray
        Mie coefficients for outside (a, b) and inside (c, d) fields.
    """
    ltab = np.asarray(ltab).ravel()
    nr = np.sqrt(epsr * mur)

    x_in = nr * k * diameter / 2
    x_out = k * diameter / 2

    j1, _, zjp1, _ = _riccatibessel(x_in, ltab)
    j2, h2, zjp2, zhp2 = _riccatibessel(x_out, ltab)

    # Outside field coefficients
    a = (nr**2 * j1 * zjp2 - mur * j2 * zjp1) / (nr**2 * j1 * zhp2 - mur * h2 * zjp1)
    b = (mur * j1 * zjp2 - j2 * zjp1) / (mur * j1 * zhp2 - h2 * zjp1)

    # Inside field coefficients
    c = (mur * j2 * zhp2 - mur * h2 * zjp2) / (mur * j1 * zhp2 - h2 * zjp1)
    d = (mur * nr * j2 * zhp2 - mur * nr * h2 * zjp2) / (nr**2 * j1 * zhp2 - mur * h2 * zjp1)

    return a, b, c, d


def _get_eps_scalar(eps_func, enei):
    """Extract scalar dielectric constant."""
    result = eps_func(enei)
    if isinstance(result, tuple):
        return result[0]
    return result


def _get_eps_and_k(eps_func, enei):
    """Extract (eps, k) from eps_func."""
    result = eps_func(enei)
    if isinstance(result, tuple):
        return result[0], result[1]
    eps = result
    k = 2 * np.pi / enei * np.sqrt(complex(eps))
    return eps, k


def _lglnodes(n):
    """Legendre-Gauss-Lobatto nodes and weights.

    MATLAB: @mieret/private/aeels.m -> lglnodes
    """
    n1 = n + 1
    x = np.cos(np.pi * np.arange(n + 1) / n)
    P = np.zeros((n1, n1))

    x_old = np.full_like(x, 2.0)
    while np.max(np.abs(x - x_old)) > np.finfo(float).eps:
        x_old = x.copy()
        P[:, 0] = 1.0
        P[:, 1] = x
        for k_idx in range(2, n + 1):
            P[:, k_idx] = ((2 * k_idx - 1) * x * P[:, k_idx - 1]
                           - (k_idx - 1) * P[:, k_idx - 2]) / k_idx
        x = x_old - (x * P[:, n] - P[:, n - 1]) / (n1 * P[:, n])

    w = 2.0 / (n * n1 * P[:, n]**2)
    return x, w


def _double_factorial(n):
    """Compute n!! (double factorial)."""
    n = int(n)
    if n <= 0:
        return 1
    if n % 2 == 0:
        result = 1
        for i in range(2, n + 1, 2):
            result *= i
        return result
    else:
        result = 1
        for i in range(1, n + 1, 2):
            result *= i
        return result


def _aeels(ltab, mtab, beta):
    """EELS spherical harmonics coefficients.

    MATLAB: @mieret/private/aeels.m
    Garcia de Abajo, Phys. Rev. B 59, 3095 (1999).

    Returns
    -------
    ce : ndarray
        Electric expansion coefficient (Eq. 31).
    cm : ndarray
        Magnetic expansion coefficient (Eq. 30).
    """
    from scipy.special import legendre as legendre_poly
    from numpy.polynomial.legendre import legval

    ltab = np.asarray(ltab).ravel()
    mtab = np.asarray(mtab).ravel()

    # LGL integration nodes
    x_nodes, w_nodes = _lglnodes(100)

    gamma = 1.0 / np.sqrt(1 - beta**2)

    # Build factorial table
    max_idx = int(np.max(ltab + np.abs(mtab))) + 2
    fac_table = np.array([float(factorial(i, exact=True)) for i in range(max_idx)])

    a = np.zeros(len(ltab), dtype=complex)

    for l_val in np.unique(ltab):
        # Compute Legendre polynomials at nodes
        # legendre(l, x) in MATLAB gives associated Legendre functions
        # We need P_l^m(x) for each m
        from scipy.special import lpmv

        for m_val in range(0, l_val + 1):
            aa = 0.0 + 0j
            alpha = np.sqrt(
                (2 * l_val + 1) / (4 * np.pi)
                * fac_table[l_val - m_val] / fac_table[l_val + m_val]
            )

            p_m = lpmv(m_val, l_val, x_nodes)

            for j in range(m_val, l_val + 1):
                if (j + m_val) % 2 == 0:
                    # Integral (A7)
                    integrand = (p_m
                                 * (1 - x_nodes**2)**(j / 2)
                                 * x_nodes**(l_val - j))
                    I = (-1)**m_val * np.sum(w_nodes * integrand)

                    # C factor (A9)
                    C = (1j**(l_val - j) * alpha * _double_factorial(2 * l_val + 1)
                         / (2**j * fac_table[l_val - j]
                            * fac_table[(j - m_val) // 2]
                            * fac_table[(j + m_val) // 2]) * I)

                    aa += C / (beta**(l_val + 1) * gamma**j)

            # Assign to indices
            i1 = np.where((ltab == l_val) & (mtab == m_val))[0]
            i2 = np.where((ltab == l_val) & (mtab == -m_val))[0]
            a[i1] = aa
            a[i2] = (-1)**m_val * aa

    # Expansion coefficient b (Eq. 15)
    b = np.zeros(len(ltab), dtype=complex)
    ltab_f = ltab.astype(float)
    mtab_f = mtab.astype(float)

    # ip: indices where m != l (can shift up)
    ip = np.where(mtab != ltab)[0]
    # im: indices where m != -l (can shift down)
    im = np.where(mtab != -ltab)[0]

    # For ip (m < l), a[ip+1] corresponds to (l, m+1)
    # Note: this assumes sphtable ordering where consecutive entries
    # within same l are m, m+1, ...
    if len(ip) > 0:
        b[ip] += (a[ip + 1]
                  * np.sqrt((ltab_f[ip] + mtab_f[ip] + 1)
                            * (ltab_f[ip] - mtab_f[ip])))
    if len(im) > 0:
        b[im] -= (a[im - 1]
                   * np.sqrt((ltab_f[im] - mtab_f[im] + 1)
                             * (ltab_f[im] + mtab_f[im])))

    # Magnetic and electric expansion coefficients
    cm = (1.0 / (ltab_f * (ltab_f + 1)) * np.abs(2 * beta * mtab_f * a)**2)
    ce = (1.0 / (ltab_f * (ltab_f + 1)) * np.abs(b / gamma)**2)

    return ce, cm


class MieRet(object):
    """Full retarded Mie theory for spherical particles.

    MATLAB: @mieret

    Parameters
    ----------
    epsin : callable
        Dielectric function inside sphere. Must return (eps, k) tuple.
    epsout : callable
        Dielectric function outside sphere. Must return (eps, k) tuple.
    diameter : float
        Sphere diameter in nm.
    lmax : int
        Maximum angular momentum (default 20).
    """

    def __init__(self, epsin, epsout, diameter, lmax=20):
        self.epsin = epsin
        self.epsout = epsout
        self.diameter = diameter
        self._ltab, self._mtab = sphtable(lmax)

    def extinction(self, enei):
        """Extinction cross section.

        MATLAB: @mieret/extinction.m
        """
        enei = np.asarray(enei, dtype=float).ravel()

        epsb = _get_eps_scalar(self.epsout, 0.0)
        if isinstance(epsb, np.ndarray):
            epsb = complex(epsb.ravel()[0])
        nb = np.sqrt(epsb)

        l_unique, ind = np.unique(self._ltab, return_index=True)

        ext = np.zeros(len(enei))
        for i in range(len(enei)):
            k = 2 * np.pi / enei[i] * nb
            epsz = _get_eps_scalar(self.epsin, enei[i]) / epsb
            a, b, _, _ = _miecoefficients(k, self.diameter, epsz, 1.0, self._ltab)
            ext[i] = np.real(2 * np.pi / k**2
                      * np.sum((2 * l_unique + 1) * np.real(a[ind] + b[ind])))

        return np.real(ext)

    def scattering(self, enei):
        """Scattering cross section.

        MATLAB: @mieret/scattering.m
        """
        enei = np.asarray(enei, dtype=float).ravel()

        epsb = _get_eps_scalar(self.epsout, 0.0)
        if isinstance(epsb, np.ndarray):
            epsb = complex(epsb.ravel()[0])
        nb = np.sqrt(epsb)

        l_unique, ind = np.unique(self._ltab, return_index=True)

        sca = np.zeros(len(enei))
        for i in range(len(enei)):
            k = 2 * np.pi / enei[i] * nb
            epsz = _get_eps_scalar(self.epsin, enei[i]) / epsb
            a, b, _, _ = _miecoefficients(k, self.diameter, epsz, 1.0, self._ltab)
            sca[i] = np.real(2 * np.pi / k**2
                      * np.sum((2 * l_unique + 1)
                               * (np.abs(a[ind])**2 + np.abs(b[ind])**2)))

        return np.real(sca)

    def absorption(self, enei):
        """Absorption cross section.

        MATLAB: @mieret/absorption.m
        """
        return self.extinction(enei) - self.scattering(enei)

    def decayrate(self, enei, z):
        """Total and radiative decay rate.

        MATLAB: @mieret/decayrate.m
        Kim et al., Surf. Science 195, 1 (1988).

        Parameters
        ----------
        enei : float
            Wavelength (single value).
        z : ndarray
            Dipole positions on z axis (nm).

        Returns
        -------
        tot : ndarray, shape (len(z), 2)
            Total decay rate for [x, z] orientations.
        rad : ndarray, shape (len(z), 2)
            Radiative decay rate for [x, z] orientations.
        """
        enei = np.asarray(enei, dtype=float)
        assert enei.size == 1, "decayrate requires a single wavelength"
        enei_val = float(enei.ravel()[0])

        z = np.asarray(z, dtype=float).ravel()

        epsb, k = _get_eps_and_k(self.epsout, enei_val)
        if isinstance(epsb, np.ndarray):
            epsb = complex(epsb.ravel()[0])
        if isinstance(k, np.ndarray):
            k = complex(k.ravel()[0])
        epsin, kin = _get_eps_and_k(self.epsin, enei_val)
        if isinstance(epsin, np.ndarray):
            epsin = complex(epsin.ravel()[0])
        if isinstance(kin, np.ndarray):
            kin = complex(kin.ravel()[0])

        tot = np.zeros((len(z), 2))
        rad = np.zeros((len(z), 2))

        l_unique = np.unique(self._ltab)
        l = l_unique.astype(float)

        # Mie coefficients at sphere surface
        j1, h1, zjp1, zhp1 = _riccatibessel(0.5 * k * self.diameter, l_unique)
        j2, _, zjp2, _ = _riccatibessel(0.5 * kin * self.diameter, l_unique)

        # Modified Mie coefficients (Kim et al. Eq. 11)
        A = (j1 * zjp2 - j2 * zjp1) / (j2 * zhp1 - h1 * zjp2)
        B = (epsb * j1 * zjp2 - epsin * j2 * zjp1) / (epsin * j2 * zhp1 - epsb * h1 * zjp2)

        for iz in range(len(z)):
            y = k * z[iz]
            j, h, zjp, zhp = _riccatibessel(y, l_unique)

            # Total decay rates (Eqs. 17, 19)
            tot[iz, 0] = 1 + 1.5 * np.real(np.sum(
                (l + 0.5) * (B * (zhp / y)**2 + A * h**2)))
            tot[iz, 1] = 1 + 1.5 * np.real(np.sum(
                (2 * l + 1) * l * (l + 1) * B * (h / y)**2))

            # Radiative decay rates (Eqs. 18, 20)
            rad[iz, 0] = 0.75 * np.sum(
                (2 * l + 1) * (np.abs(j + A * h)**2
                                + np.abs((zjp + B * zhp) / y)**2))
            rad[iz, 1] = 1.5 * np.sum(
                (2 * l + 1) * l * (l + 1)
                * np.abs((j + B * h) / y)**2)

        tot = np.real(tot)
        rad = np.real(rad)
        return tot, rad

    def loss(self, b, enei, beta=0.7):
        """EELS loss probability (retarded).

        MATLAB: @mieret/loss.m
        Garcia de Abajo, Phys. Rev. B 59, 3095 (1999).

        Parameters
        ----------
        b : ndarray
            Impact parameter (distance from sphere surface, nm).
        enei : ndarray
            Wavelengths in vacuum (nm).
        beta : float
            Electron velocity / speed of light (default 0.7).

        Returns
        -------
        prob : ndarray, shape (len(b), len(enei))
            EELS probability.
        prad : ndarray, shape (len(b), len(enei))
            Photon emission probability.
        """
        b = np.asarray(b, dtype=float).ravel()
        enei = np.asarray(enei, dtype=float).ravel()
        assert np.all(b > 0), "Impact parameter must be positive"

        b_total = 0.5 * self.diameter + b
        gamma = 1.0 / np.sqrt(1 - beta**2)

        prob = np.zeros((len(b), len(enei)))
        prad = np.zeros((len(b), len(enei)))

        lmax = int(np.max(self._ltab))
        l_full, m_full = sphtable(lmax, 'full')
        ce, cm = _aeels(l_full, m_full, beta)
        m_f = m_full.astype(float)

        for ien in range(len(enei)):
            epsb, k = _get_eps_and_k(self.epsout, enei[ien])
            if isinstance(epsb, np.ndarray):
                epsb = complex(epsb.ravel()[0])
            if isinstance(k, np.ndarray):
                k = complex(k.ravel()[0])

            epsz = _get_eps_scalar(self.epsin, enei[ien]) / epsb

            # Mie coefficients -> te, tm (Garcia notation)
            te, tm = _miecoefficients(np.real(k), self.diameter, epsz, 1.0, l_full)[:2]
            te = 1j * te
            tm = 1j * tm

            for ib in range(len(b)):
                K = besselk(m_f, np.real(k) * b_total[ib] / (beta * gamma))
                prob[ib, ien] = np.real(
                    np.sum(K**2 * (cm * np.imag(tm) + ce * np.imag(te))) / np.real(k))
                prad[ib, ien] = np.real(
                    np.sum(K**2 * (cm * np.abs(tm)**2 + ce * np.abs(te)**2)) / np.real(k))

        # Convert to units of 1/eV
        prob = FINE**2 / (BOHR * HARTREE) * prob
        prad = FINE**2 / (BOHR * HARTREE) * prad
        return prob, prad

    def __repr__(self):
        return "MieRet(diameter={})".format(self.diameter)
