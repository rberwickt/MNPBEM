"""
Factory function for Mie solver.

MATLAB reference: Mie/miesolver.m
"""

from .mie_stat import MieStat
from .mie_ret import MieRet


def mie_solver(epsin, epsout, diameter, sim='stat', lmax=20):
    """Create Mie solver based on simulation type.

    MATLAB: miesolver.m

    Parameters
    ----------
    epsin : callable
        Dielectric function inside sphere.
    epsout : callable
        Dielectric function outside sphere.
    diameter : float
        Sphere diameter in nm.
    sim : str
        'stat' for quasistatic, 'ret' for retarded.
    lmax : int
        Maximum angular momentum (default 20).

    Returns
    -------
    mie : MieStat or MieRet
        Mie solver object.
    """
    if sim == 'stat':
        return MieStat(epsin, epsout, diameter, lmax=lmax)
    elif sim == 'ret':
        return MieRet(epsin, epsout, diameter, lmax=lmax)
    else:
        raise ValueError("Unknown simulation type: '{}'. Use 'stat' or 'ret'.".format(sim))
