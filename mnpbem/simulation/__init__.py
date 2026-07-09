"""
Simulation module for MNPBEM.

This module provides excitation sources for BEM simulations:
- PlaneWaveStat: Plane wave excitation for quasistatic simulations
- PlaneWaveRet: Plane wave excitation for retarded simulations
- DipoleStat: Dipole excitation for quasistatic simulations
- DipoleRet: Dipole excitation for retarded simulations
- PlaneWaveStatMirror: Plane wave excitation (quasistatic + mirror symmetry)
- PlaneWaveRetMirror: Plane wave excitation (retarded + mirror symmetry)
- DipoleStatMirror: Dipole excitation (quasistatic + mirror symmetry)
- DipoleRetMirror: Dipole excitation (retarded + mirror symmetry)
- EELSBase: Base class for EELS simulations
- EELSStat: EELS excitation for quasistatic simulations
- EELSRet: EELS excitation for retarded simulations
- PlaneWaveStatLayer: Plane wave excitation (quasistatic + layer structure)
- PlaneWaveRetLayer: Plane wave excitation (retarded + layer structure)
- DipoleStatLayer: Dipole excitation (quasistatic + layer structure)
- DipoleRetLayer: Dipole excitation (retarded + layer structure)

Matches MATLAB MNPBEM Simulation module exactly.
"""

from .planewave_stat import PlaneWaveStat
from .planewave_ret import PlaneWaveRet
from .dipole_stat import DipoleStat
from .dipole_ret import DipoleRet
from .planewave_stat_mirror import PlaneWaveStatMirror
from .planewave_ret_mirror import PlaneWaveRetMirror
from .dipole_stat_mirror import DipoleStatMirror
from .dipole_ret_mirror import DipoleRetMirror
from .eels_base import EELSBase
from .eels_stat import EELSStat
from .eels_ret import EELSRet
from .planewave_stat_layer import PlaneWaveStatLayer
from .planewave_ret_layer import PlaneWaveRetLayer
from .dipole_stat_layer import DipoleStatLayer
from .dipole_ret_layer import DipoleRetLayer
from .meshfield import MeshField
from .retarded_utils import scattering, extinction, absorption
from .dipole_factory import dipole
from .planewave_factory import planewave
from .electronbeam_factory import electronbeam

__all__ = [
    "PlaneWaveStat",
    "PlaneWaveRet",
    "DipoleStat",
    "DipoleRet",
    "PlaneWaveStatMirror",
    "PlaneWaveRetMirror",
    "DipoleStatMirror",
    "DipoleRetMirror",
    "EELSBase",
    "EELSStat",
    "EELSRet",
    "PlaneWaveStatLayer",
    "PlaneWaveRetLayer",
    "DipoleStatLayer",
    "DipoleRetLayer",
    "MeshField",
    "scattering",
    "extinction",
    "absorption",
    "dipole",
    "planewave",
    "electronbeam",
]
