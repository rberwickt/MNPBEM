"""
Tabulated dielectric function with interpolation.
"""

import numpy as np
from scipy.interpolate import CubicSpline
import os
from ..utils.constants import EV2NM
from ..utils.gpu import _CUPY_OK, _cp


def _to_host(x):
    if _CUPY_OK and isinstance(x, _cp.ndarray):
        return _cp.asnumpy(x)
    return x


class EpsTable(object):
    """
    Interpolate from tabulated values of dielectric function.

    Reads tabulated dielectric function data from file and provides
    wavelength-dependent dielectric constant through spline interpolation.

    Parameters
    ----------
    filename : str
        Path to data file or filename in the data directory.
        File format: "energy(eV) n k" per line
        - energy: photon energy in eV
        - n: refractive index (real part)
        - k: refractive index (imaginary part)

    Available data files:
        - 'gold.dat', 'silver.dat' : Johnson & Christy
        - 'goldpalik.dat', 'silverpalik.dat', 'copperpalik.dat' : Palik

    Examples
    --------
    >>> # Gold from Johnson & Christy
    >>> eps_gold = EpsTable('gold.dat')
    >>>
    >>> # Get dielectric function at 500 nm
    >>> eps_val, k = eps_gold(500)
    >>>
    >>> # Get at multiple wavelengths
    >>> wavelengths = np.linspace(400, 700, 100)
    >>> eps_array, k_array = eps_gold(wavelengths)
    """

    def __init__(self, filename):
        """
        Initialize tabulated dielectric function.

        Parameters
        ----------
        filename : str
            Path to data file or filename
        """
        # Find the file
        if os.path.exists(filename):
            filepath = filename
        else:
            # Try in the data directory
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    "Material data file not found: {}\n"
                    "Tried: {}".format(filename, filepath)
                )

        # Read data file
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line.startswith('%') or line.startswith('#') or not line:
                    continue
                try:
                    values = [float(x) for x in line.split()]
                    if len(values) >= 3:
                        data.append(values[:3])
                except ValueError:
                    continue

        if not data:
            raise ValueError("No valid data found in {}".format(filepath))

        data = np.array(data)
        ene_ev = data[:, 0]  # Energy in eV
        n = data[:, 1]       # Real part of refractive index
        k = data[:, 2]       # Imaginary part of refractive index

        # Convert energy from eV to wavelength in nm
        self.enei = EV2NM / ene_ev

        # Create splines for interpolation (wavelength in nm)
        # Note: wavelengths are in reverse order (high to low energy)
        # Need to sort for interpolation
        sort_idx = np.argsort(self.enei)
        self.enei = self.enei[sort_idx]
        n = n[sort_idx]
        k = k[sort_idx]

        # Cubic spline interpolation
        self.ni = CubicSpline(self.enei, n)
        self.ki = CubicSpline(self.enei, k)

        # Store filename for reference
        self.filename = os.path.basename(filepath)

    def __call__(self, enei):
        """
        Interpolate dielectric function and wavenumber.

        Parameters
        ----------
        enei : float or array_like
            Light wavelength in vacuum (nm)

        Returns
        -------
        eps : complex or ndarray
            Interpolated dielectric function: ε = (n + ik)²
        k : complex or ndarray
            Wavenumber in medium (1/nm): k = 2π/λ × √ε
        """
        enei = np.asarray(_to_host(enei))

        # Check if wavelengths are in valid range
        enei_min, enei_max = self.enei.min(), self.enei.max()
        if np.any(enei < enei_min) or np.any(enei > enei_max):
            raise ValueError(
                "Wavelength out of range. Valid range: "
                "{:.1f} - {:.1f} nm, "
                "requested: {:.1f} - {:.1f} nm".format(enei_min, enei_max, enei.min(), enei.max())
            )

        # Interpolate refractive index
        ni = self.ni(enei)
        ki = self.ki(enei)

        # Compute dielectric function: ε = (n + ik)²
        n_complex = ni + 1j * ki
        eps = n_complex ** 2

        # Compute wavenumber: k = 2π/λ × √ε
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

    def refractive_index(self, enei):
        """
        Get complex refractive index.

        Parameters
        ----------
        enei : float or array_like
            Light wavelength in vacuum (nm)

        Returns
        -------
        n : complex or ndarray
            Complex refractive index: n + ik
        """
        enei = np.asarray(_to_host(enei))
        ni = self.ni(enei)
        ki = self.ki(enei)
        return ni + 1j * ki

    def __repr__(self):
        return "EpsTable('{}')".format(self.filename)

    def __str__(self):
        return (
            "Tabulated dielectric function from {}\n"
            "Wavelength range: {:.1f} - {:.1f} nm".format(self.filename, self.enei.min(), self.enei.max())
        )
