"""
Dielectric function factory and user-supplied function wrapper.

MATLAB: epsfun.m
"""

import numpy as np
from .eps_const import EpsConst
from .eps_table import EpsTable
from .eps_drude import EpsDrude


class EpsFun(object):
    """
    Dielectric function using user-supplied function.

    Wraps a callable that computes the dielectric function from wavelength.

    Parameters
    ----------
    fun : callable
        Function for evaluation: eps = fun(enei)
        Must accept wavelength in nm and return complex dielectric function.
    key : str, optional
        Input unit type: 'nm' for wavelengths (default) or 'eV' for energies.

    Examples
    --------
    >>> # Custom Lorentz oscillator
    >>> def lorentz(enei):
    ...     w = 1240.0 / enei
    ...     return 1.0 + 1.0 / (3.0 - w**2 - 0.1j * w)
    >>> eps = EpsFun(lorentz)
    >>> eps_val, k = eps(500)

    Notes
    -----
    MATLAB equivalent: @epsfun class
    """

    # eV to nm conversion (MATLAB Misc/units.m: eV2nm = 1 / 8.0655477e-4)
    _EV2NM = 1.0 / 8.0655477e-4

    def __init__(self, fun, key = 'nm'):
        """
        Initialize user-supplied dielectric function.

        Parameters
        ----------
        fun : callable
            Function for evaluation: eps = fun(enei)
        key : str, optional
            'nm' for wavelength input, 'eV' for energy input
        """
        if not callable(fun):
            raise TypeError('fun must be callable')
        if key not in ('nm', 'eV'):
            raise ValueError("key must be 'nm' or 'eV', got '{}'".format(key))

        self.fun = fun
        self.key = key

    def __call__(self, enei):
        """
        Get dielectric constant and wavenumber.

        Parameters
        ----------
        enei : float or array_like
            Light wavelength in vacuum (nm)

        Returns
        -------
        eps : complex or ndarray
            Dielectric function
        k : complex or ndarray
            Wavenumber in medium (1/nm)
        """
        enei = np.asarray(enei, dtype = float)

        # evaluate dielectric function
        if self.key == 'nm':
            eps = self.fun(enei)
        else:
            # convert wavelength to eV before calling function
            eps = self.fun(self._EV2NM / enei)

        eps = np.asarray(eps, dtype = complex)

        # wavenumber: k = 2pi / lambda * sqrt(eps)
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

    def __repr__(self):
        return "EpsFun(fun = {}, key = '{}')".format(self.fun, self.key)

    def __str__(self):
        return "User-supplied dielectric function (key = '{}')".format(self.key)


def epsfun(arg):
    """
    Create dielectric function from various inputs.

    Convenience factory function that creates the appropriate dielectric
    function object based on the type of input.

    Parameters
    ----------
    arg : numeric, str, or callable
        - numeric value (int, float, complex) -> EpsConst
        - string filename (e.g. 'gold.dat') -> EpsTable
        - string 'drude:gold', 'drude:silver', 'drude:aluminum' -> EpsDrude
        - callable function -> EpsFun

    Returns
    -------
    eps_func : EpsConst, EpsTable, EpsDrude, or EpsFun
        Dielectric function object

    Examples
    --------
    >>> # Constant dielectric function (vacuum)
    >>> eps = epsfun(1.0)

    >>> # From tabulated data file
    >>> eps = epsfun('gold.dat')

    >>> # Drude model for gold
    >>> eps = epsfun('drude:gold')

    >>> # User-supplied function
    >>> eps = epsfun(lambda enei: 1.0 + 0.5j * enei / 500)

    Notes
    -----
    MATLAB equivalent: epsfun.m (constructor dispatching)
    """
    # numeric value -> EpsConst
    if isinstance(arg, (int, float, complex, np.integer, np.floating, np.complexfloating)):
        return EpsConst(arg)

    # numpy scalar -> EpsConst
    if isinstance(arg, np.ndarray) and arg.ndim == 0:
        return EpsConst(arg.item())

    # string -> EpsTable or EpsDrude
    if isinstance(arg, str):
        # check for drude model specification
        if arg.startswith('drude:'):
            material = arg[6:].strip().lower()
            drude_map = {
                'gold': EpsDrude.gold,
                'au': EpsDrude.gold,
                'silver': EpsDrude.silver,
                'ag': EpsDrude.silver,
                'aluminum': EpsDrude.aluminum,
                'aluminium': EpsDrude.aluminum,
                'al': EpsDrude.aluminum,
            }
            if material not in drude_map:
                raise ValueError(
                    "Unknown Drude material: '{}'. "
                    "Available: {}".format(material, list(drude_map.keys()))
                )
            return drude_map[material]()

        # otherwise treat as filename for EpsTable
        return EpsTable(arg)

    # callable -> EpsFun
    if callable(arg):
        return EpsFun(arg)

    raise TypeError(
        "Cannot create dielectric function from type '{}'. "
        "Expected numeric, string, or callable.".format(type(arg).__name__)
    )
