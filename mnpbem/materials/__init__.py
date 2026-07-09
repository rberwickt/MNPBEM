"""
Material dielectric functions.

Classes:
- EpsConst: Constant dielectric function
- EpsTable: Tabulated dielectric function with interpolation
- EpsDrude: Drude model dielectric function
- EpsFun: User-supplied dielectric function
- EpsNonlocal: Hydrodynamic Drude nonlocal cover-layer dielectric function

Functions:
- epsfun: Convenience factory for creating dielectric functions
- make_nonlocal_pair: Build (eps_metal, eps_nonlocal_cover) pair for
  hydrodynamic Drude simulations.
"""

from .eps_const import EpsConst
from .eps_table import EpsTable
from .eps_drude import EpsDrude
from .epsfun import EpsFun, epsfun
from .eps_nonlocal import EpsNonlocal, make_nonlocal_pair

__all__ = [
    "EpsConst",
    "EpsTable",
    "EpsDrude",
    "EpsFun",
    "epsfun",
    "EpsNonlocal",
    "make_nonlocal_pair",
]
