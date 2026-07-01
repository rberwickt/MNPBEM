"""
Drude model dielectric function.

Matches MATLAB MNPBEM @epsdrude implementation exactly.
"""

import numpy as np

from ..utils.constants import EV2NM


class EpsDrude(object):
    """
    Drude model dielectric function.

    Formula:
        eps = eps0 - wp^2 / (w * (w + i*gammad))

    where w is the photon energy in eV.

    Parameters
    ----------
    eps0 : float
        Background dielectric constant (high-frequency limit)
    wp : float
        Plasma frequency in eV
    gammad : float
        Damping rate in eV

    Examples
    --------
    >>> # Gold (approximate Drude parameters)
    >>> eps_au = EpsDrude(9.5, 8.95, 0.069)
    >>> eps_val, k = eps_au(500)  # at 500 nm

    >>> # Use predefined metal
    >>> eps_au = EpsDrude.gold()
    >>> eps_ag = EpsDrude.silver()

    Notes
    -----
    MATLAB equivalent: @epsdrude class

    MATLAB-compatible Drude parameters (calculated from Jellium model):
    - Gold (Au):   eps0=10,   wp=9.071 eV, gammad=0.066 eV
    - Silver (Ag): eps0=3.3,  wp=9.071 eV, gammad=0.022 eV
    - Aluminum (Al): eps0=1.0, wp=15.826 eV, gammad=1.060 eV

    Use EpsDrude.gold(), EpsDrude.silver(), or EpsDrude.aluminum() for
    MATLAB-compatible parameters, or specify custom values directly.
    """

    def __init__(self, eps0, wp, gammad, name=None):
        """
        Initialize Drude dielectric function.

        Parameters
        ----------
        eps0 : float
            Background dielectric constant
        wp : float
            Plasma frequency in eV
        gammad : float
            Damping rate in eV
        name : str, optional
            Material name (e.g., 'Au', 'Ag')
        """
        self.eps0 = eps0
        self.wp = wp
        self.gammad = gammad
        self.name = name

    def __call__(self, enei):
        """
        Get dielectric constant and wavenumber.

        MATLAB: subsref.m
            w = eV2nm / enei
            eps = eps0 - wp^2 / (w * (w + 1i*gammad))
            k = 2*pi / enei * sqrt(eps)

        Parameters
        ----------
        enei : float or array_like
            Light wavelength in vacuum (nm)

        Returns
        -------
        eps : complex or ndarray
            Drude dielectric function
        k : complex or ndarray
            Wavenumber in medium (1/nm)
        """
        enei = np.asarray(enei, dtype=float)

        # Convert wavelength to photon energy in eV
        # MATLAB: w = eV2nm / enei
        w = EV2NM / enei

        # Drude formula
        # MATLAB: eps = eps0 - wp^2 / (w * (w + 1i*gammad))
        eps = self.eps0 - self.wp**2 / (w * (w + 1j * self.gammad))

        # Wavenumber: k = 2π/λ × √ε
        k = 2 * np.pi / enei * np.sqrt(eps)

        return eps, k

    def wavenumber(self, enei):
        """
        Get wavenumber in medium.

        Parameters
        ----------
        enei : float or array_like
            Light wavelength in vacuum (nm)

        Returns
        -------
        k : complex or ndarray
            Wavenumber in medium (1/nm)
        """
        _, k = self(enei)
        return k

    @classmethod
    def gold(cls):
        """
        Create Drude model for gold (Au) - MATLAB compatible.

        Uses MATLAB @epsdrude/init.m calculation (Jellium model).

        Returns
        -------
        EpsDrude
            Gold dielectric function
        """
        return cls._init_from_matlab_model('Au')

    @classmethod
    def silver(cls):
        """
        Create Drude model for silver (Ag) - MATLAB compatible.

        Uses MATLAB @epsdrude/init.m calculation (Jellium model).

        Returns
        -------
        EpsDrude
            Silver dielectric function
        """
        return cls._init_from_matlab_model('Ag')

    @classmethod
    def aluminum(cls):
        """
        Create Drude model for aluminum (Al) - MATLAB compatible.

        Uses MATLAB @epsdrude/init.m calculation (Jellium model).

        Returns
        -------
        EpsDrude
            Aluminum dielectric function
        """
        return cls._init_from_matlab_model('Al')

    @classmethod
    def _init_from_matlab_model(cls, name):
        """
        Initialize Drude parameters using MATLAB @epsdrude/init.m calculation.

        This replicates the exact calculation from MATLAB init.m:
        - Jellium model with electron gas parameter rs
        - Atomic units conversion
        - Density calculation: density = 3 / (4*pi*rs^3)
        - Plasma frequency: wp = sqrt(4*pi*density)

        Parameters
        ----------
        name : str
            Material name: 'Au', 'gold', 'Ag', 'silver', 'Al', or 'aluminum'

        Returns
        -------
        EpsDrude
            Drude dielectric function with MATLAB-calculated parameters
        """
        # MATLAB init.m line 5-6: atomic units
        hartree = 27.2116  # 2 * Rydberg in eV
        tunit = 0.66 / hartree  # time unit in fs

        # MATLAB init.m line 8-23: switch case
        if name in ['Au', 'gold']:
            rs = 3                  # electron gas parameter
            eps0 = 10               # background dielectric constant
            gammad = tunit / 10     # Drude relaxation rate
        elif name in ['Ag', 'silver']:
            rs = 3
            eps0 = 3.3
            gammad = tunit / 30
        elif name in ['Al', 'aluminum']:
            rs = 2.07
            eps0 = 1
            gammad = 1.06 / hartree
        else:
            raise ValueError("Material name unknown: {}".format(name))

        # MATLAB init.m line 25-28: density and plasmon energy
        density = 3 / (4 * np.pi * rs ** 3)  # density in atomic units
        wp = np.sqrt(4 * np.pi * density)    # plasmon energy

        # MATLAB init.m line 31-32: save values (convert to eV)
        gammad_ev = gammad * hartree
        wp_ev = wp * hartree

        return cls(eps0=eps0, wp=wp_ev, gammad=gammad_ev, name=name)

    def __repr__(self):
        if self.name:
            return "EpsDrude('{}', eps0 = {}, wp = {}, gammad = {})".format(self.name, self.eps0, self.wp, self.gammad)
        return "EpsDrude(eps0 = {}, wp = {}, gammad = {})".format(self.eps0, self.wp, self.gammad)

    def __str__(self):
        name_str = " ({})".format(self.name) if self.name else ""
        return "Drude dielectric function{}: eps = {} - {}^2/(w(w+i{}))".format(name_str, self.eps0, self.wp, self.gammad)
