"""
Constant dielectric function.
"""

import numpy as np


class EpsConst(object):
    """
    Constant dielectric function.

    Represents a medium with a constant, wavelength-independent dielectric constant.

    Parameters
    ----------
    eps : float or complex
        Dielectric constant value

    Examples
    --------
    >>> # Vacuum
    >>> eps_vacuum = EpsConst(1.0)
    >>>
    >>> # Water
    >>> eps_water = EpsConst(1.33**2)
    >>>
    >>> # Get dielectric function at 500 nm
    >>> eps_val, k = eps_vacuum(500)
    """

    def __init__(self, eps):
        """
        Initialize constant dielectric function.

        Parameters
        ----------
        eps : float or complex
            Dielectric constant value
        """
        if eps is None:
            raise ValueError("EpsConst: 'eps' must be a numeric value, got None.")
        if not isinstance(eps, (int, float, complex, np.integer, np.floating, np.complexfloating)):
            raise TypeError(
                "EpsConst: 'eps' must be a numeric (int/float/complex) value, "
                "got {!r}.".format(type(eps).__name__))
        self.eps = eps

    def __call__(self, enei):
        """
        Get dielectric constant and wavenumber.

        Parameters
        ----------
        enei : float or array_like
            Light wavelength in vacuum (nm)

        Returns
        -------
        eps : float or complex or ndarray
            Dielectric constant (same shape as enei)
        k : float or complex or ndarray
            Wavenumber in medium (1/nm)
        """
        enei = np.asarray(enei)

        # Dielectric constant (broadcast to enei shape)
        eps = np.full_like(enei, self.eps, dtype=complex)

        # Wavenumber: k = 2π/λ × √ε  (use complex eps so negative real eps gives imaginary k)
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
        k : float or complex or ndarray
            Wavenumber in medium (1/nm)
        """
        enei = np.asarray(enei)
        eps_complex = np.array(self.eps, dtype = complex)
        return 2 * np.pi / enei * np.sqrt(eps_complex)

    def __repr__(self):
        return "EpsConst(eps = {})".format(self.eps)

    def __str__(self):
        return "Constant dielectric function: eps = {}".format(self.eps)
