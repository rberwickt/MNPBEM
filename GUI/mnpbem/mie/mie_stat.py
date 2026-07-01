"""
Quasistatic Mie theory for spherical particles.

MATLAB reference: Mie/@miestat/
"""

import numpy as np
from scipy.special import factorial
from scipy.special import kv as besselk

from .spherical_harmonics import sphtable, spharm, vecspharm
from ..misc.units import FINE, BOHR, HARTREE


def _get_eps_scalar(eps_func, enei):
    """Extract scalar dielectric constant from eps_func(enei)."""
    result = eps_func(enei)
    if isinstance(result, tuple):
        return result[0]
    return result


def _get_eps_and_k(eps_func, enei):
    """Extract (eps, k) from eps_func(enei)."""
    result = eps_func(enei)
    if isinstance(result, tuple):
        return result[0], result[1]
    eps = result
    k = 2 * np.pi / enei * np.sqrt(complex(eps))
    return eps, k


class MieStat(object):
    """Quasistatic Mie theory for spherical particles.

    MATLAB: @miestat

    Parameters
    ----------
    epsin : callable
        Dielectric function inside sphere.
    epsout : callable
        Dielectric function outside sphere.
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

        MATLAB: @miestat/extinction.m
        """
        return self.scattering(enei) + self.absorption(enei)

    def scattering(self, enei):
        """Scattering cross section.

        MATLAB: @miestat/scattering.m
        """
        enei = np.asarray(enei, dtype=float)
        epsb = _get_eps_scalar(self.epsout, 0.0)
        if isinstance(epsb, np.ndarray):
            epsb = complex(epsb.ravel()[0])
        nb = np.sqrt(epsb)
        k = 2 * np.pi / enei * nb
        epsz = _get_eps_scalar(self.epsin, enei) / epsb

        a = self.diameter / 2
        alpha = (epsz - 1) / (epsz + 2) * a**3
        return np.real(8 * np.pi / 3 * k**4 * np.abs(alpha)**2)

    def absorption(self, enei):
        """Absorption cross section.

        MATLAB: @miestat/absorption.m
        """
        enei = np.asarray(enei, dtype=float)
        epsb = _get_eps_scalar(self.epsout, 0.0)
        if isinstance(epsb, np.ndarray):
            epsb = complex(epsb.ravel()[0])
        nb = np.sqrt(epsb)
        k = 2 * np.pi / enei * nb
        epsz = _get_eps_scalar(self.epsin, enei) / epsb

        a = self.diameter / 2
        alpha = (epsz - 1) / (epsz + 2) * a**3
        return np.real(4 * np.pi * k * np.imag(alpha))

    def decayrate(self, enei, z):
        """Total and radiative decay rate for oscillating dipole.

        MATLAB: @miestat/decayrate.m

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

        epsb = _get_eps_scalar(self.epsout, 0.0)
        if isinstance(epsb, np.ndarray):
            epsb = complex(epsb.ravel()[0])
        nb = np.sqrt(epsb)

        # Free-space radiative decay rate (Wigner-Weisskopf)
        gamma0 = 4.0 / 3 * nb * (2 * np.pi / enei_val)**3
        epsz = _get_eps_scalar(self.epsin, enei_val) / epsb

        tot = np.zeros((len(z), 2))
        rad = np.zeros((len(z), 2))

        ltab = self._ltab
        mtab = self._mtab

        # Dipole orientations: [1,0,0] (x) and [0,0,1] (z)
        dips = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])]

        for iz in range(len(z)):
            for idip in range(2):
                dip = dips[idip]
                pos = np.array([0.0, 0.0, z[iz]]) / self.diameter

                # Spherical harmonics coefficients for dipole
                adip = _adipole(pos, dip, ltab, mtab)

                # Induced dipole moment
                indip = _dipole(ltab, mtab, adip, 1.0, epsz)
                rad[iz, idip] = np.linalg.norm(dip + indip)**2

                # Induced electric field (normalized by epsb * diameter^3)
                efield = _field(ltab, mtab, pos, adip, 1.0, epsz)
                efield = efield / (epsb * self.diameter**3)
                tot[iz, idip] = (rad[iz, idip]
                                 + np.imag(efield @ dip) / (gamma0 / 2))

        return tot, rad

    def loss(self, b, enei, beta=0.7):
        """EELS loss probability (quasistatic).

        MATLAB: @miestat/loss.m
        Garcia de Abajo, Rev. Mod. Phys. 82, 209 (2010), Eq. (31).

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
        """
        b = np.asarray(b, dtype=float).ravel()
        enei = np.asarray(enei, dtype=float).ravel()
        assert np.all(b > 0), "Impact parameter must be positive"

        a = 0.5 * self.diameter
        b_total = a + b  # add sphere radius

        lmax = int(np.max(self._ltab))
        l_full, m_full = sphtable(lmax, 'full')
        l_f = l_full.astype(float)
        m_f = m_full.astype(float)

        prob = np.zeros((len(b), len(enei)))

        for ien in range(len(enei)):
            epsb, k = _get_eps_and_k(self.epsout, enei[ien])
            if isinstance(epsb, np.ndarray):
                epsb = complex(epsb.ravel()[0])
            if isinstance(k, np.ndarray):
                k = complex(k.ravel()[0])
            epsz_val = _get_eps_scalar(self.epsin, enei[ien]) / epsb

            # Polarizability for each l
            alpha = (l_f * epsz_val - l_f) / (l_f * epsz_val + l_f + 1) * a**3
            # Prefactor
            fac = ((np.real(k) * a / beta)**(2 * l_f)
                   / (factorial(l_f + m_f, exact=False) * factorial(l_f - m_f, exact=False)))

            for ib in range(len(b)):
                K = besselk(m_f, np.real(k) * b_total[ib] / beta)
                prob[ib, ien] = np.real(np.sum(fac * K**2 * np.imag(alpha)))

        # Convert to units of 1/eV
        prob = 4 * FINE**2 / (np.pi * HARTREE * BOHR * beta**2 * a**2) * prob
        return prob

    def __repr__(self):
        return "MieStat(diameter={})".format(self.diameter)


def _adipole(pos, dip, ltab, mtab):
    """Spherical harmonics coefficients for a dipole.

    MATLAB: @miestat/private/adipole.m
    Jackson eq. (4.1)

    Parameters
    ----------
    pos : ndarray, shape (3,)
        Position of dipole (normalized by diameter).
    dip : ndarray, shape (3,)
        Dipole orientation vector.
    ltab, mtab : ndarray
        Spherical harmonic degree/order tables.

    Returns
    -------
    a : ndarray
        Expansion coefficients.
    """
    # Convert to spherical coordinates
    # MATLAB: cart2sph(x, y, z) returns (azimuth, elevation, r)
    x, y, z_val = pos
    r = np.sqrt(x**2 + y**2 + z_val**2)
    phi_angle = np.arctan2(y, x)
    # MATLAB cart2sph returns elevation (from xy-plane), not polar angle
    elevation = np.arctan2(z_val, np.sqrt(x**2 + y**2))
    theta_angle = np.pi / 2 - elevation

    e = pos / r  # unit vector

    ltab_f = ltab.astype(float)

    # Spherical harmonics at dipole position
    x_vec, y_sph = vecspharm(ltab, mtab,
                              np.array([theta_angle]),
                              np.array([phi_angle]))
    # y_sph shape: (n_lm, 1), x_vec shape: (n_lm, 1, 3)

    y_sph = y_sph[:, 0]  # (n_lm,)
    x_vec = x_vec[:, 0, :]  # (n_lm, 3)

    e_dot_dip = np.dot(e, dip)
    cross_e_dip = np.cross(e, dip)

    a = -((ltab_f + 1) * e_dot_dip * np.conj(y_sph)
          + 1j * np.sqrt(ltab_f * (ltab_f + 1))
          * (np.conj(x_vec) @ cross_e_dip)) / r**(ltab_f + 2)

    return a


def _dipole(ltab, mtab, a, diameter, epsr):
    """Induced dipole moment of sphere.

    MATLAB: @miestat/private/dipole.m
    Jackson eq. (4.5)

    Parameters
    ----------
    ltab, mtab : ndarray
        Tables.
    a : ndarray
        Expansion coefficients.
    diameter : float
        Sphere diameter (normalized, typically 1.0).
    epsr : complex
        Relative dielectric constant.

    Returns
    -------
    dip : ndarray, shape (3,)
        Dipole moment vector.
    """
    ltab_f = ltab.astype(float)

    # Static Mie coefficients
    c = ((1 - epsr) * ltab_f / ((1 + epsr) * ltab_f + 1)
         * (diameter / 2)**(2 * ltab_f + 1) * a)

    # Extract l=1 components
    mask_l1_m0 = (ltab == 1) & (mtab == 0)
    mask_l1_mp = (ltab == 1) & (mtab == 1)
    mask_l1_mm = (ltab == 1) & (mtab == -1)

    qz = np.sqrt(4 * np.pi / 3) * np.sum(c[mask_l1_m0])
    qp = -np.sqrt(4 * np.pi / 3) * np.sum(c[mask_l1_mp])
    qm = np.sqrt(4 * np.pi / 3) * np.sum(c[mask_l1_mm])

    dip = (qz * np.array([0, 0, 1], dtype=complex)
           + (qp * np.array([1, 1j, 0]) + qm * np.array([1, -1j, 0])) / np.sqrt(2))
    return dip


def _field(ltab, mtab, pos, a, diameter, epsr):
    """Electric field from quasistatic Mie theory.

    MATLAB: @miestat/private/field.m

    Parameters
    ----------
    ltab, mtab : ndarray
        Tables.
    pos : ndarray, shape (3,) or (n, 3)
        Vertices (normalized by diameter).
    a : ndarray
        Expansion coefficients.
    diameter : float
        Sphere diameter (normalized, typically 1.0).
    epsr : complex
        Relative dielectric constant.

    Returns
    -------
    e : ndarray, shape (3,) or (n, 3)
        Electric field.
    """
    ltab_f = ltab.astype(float)

    # Static Mie coefficients
    c = ((1 - epsr) * ltab_f / ((1 + epsr) * ltab_f + 1)
         * (diameter / 2)**(2 * ltab_f + 1) * a)

    pos = np.atleast_2d(pos)
    nverts = pos.shape[0]
    r = np.sqrt(np.sum(pos**2, axis=1))
    unit = pos / r[:, np.newaxis]
    r_mean = np.mean(r)

    # Convert to spherical coordinates
    elevation = np.arctan2(pos[:, 2], np.sqrt(pos[:, 0]**2 + pos[:, 1]**2))
    theta = np.pi / 2 - elevation
    phi_angle = np.arctan2(pos[:, 1], pos[:, 0])

    x_vec, y_sph = vecspharm(ltab, mtab, theta, phi_angle)
    # y_sph: (n_lm, nverts), x_vec: (n_lm, nverts, 3)

    # Scalar potential prefactor
    fac = 4 * np.pi / (2 * ltab_f + 1) * c / r_mean**(ltab_f + 2)

    # Electric field: radial + angular parts
    # Radial part
    radial_coeff = y_sph.T @ (fac * (ltab_f + 1))  # (nverts,)
    e_radial = radial_coeff[:, np.newaxis] * unit  # (nverts, 3)

    # Angular part from vector spherical harmonics
    angular_fac = 1j * np.sqrt(ltab_f * (ltab_f + 1)) * fac  # (n_lm,)
    # Sum over lm: angular_fac[lm] * x_vec[lm, pt, comp]
    angular_sum = np.einsum('i,ijk->jk', angular_fac, x_vec)  # (nverts, 3)
    e_angular = np.cross(unit, angular_sum)

    e = e_radial + e_angular

    if nverts == 1:
        return e[0]
    return e
