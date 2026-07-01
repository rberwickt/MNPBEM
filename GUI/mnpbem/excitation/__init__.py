"""
Excitation module for MNPBEM.

Provides plane wave, dipole, and EELS excitation classes for BEM simulations.

Note: All excitation classes are implemented in the simulation module.
This module re-exports them for API compatibility.
"""

# Import all excitation classes from simulation module
from ..simulation import (
    PlaneWaveStat,
    PlaneWaveRet,
    DipoleStat,
    DipoleRet,
    PlaneWaveStatMirror,
    PlaneWaveRetMirror,
    DipoleStatMirror,
    DipoleRetMirror,
    EELSBase,
    EELSStat,
    EELSRet,
    PlaneWaveStatLayer,
    PlaneWaveRetLayer,
    DipoleStatLayer,
    DipoleRetLayer,
)

__all__ = [
    'PlaneWaveStat',
    'PlaneWaveRet',
    'DipoleStat',
    'DipoleRet',
    'PlaneWaveStatMirror',
    'PlaneWaveRetMirror',
    'DipoleStatMirror',
    'DipoleRetMirror',
    'EELSBase',
    'EELSStat',
    'EELSRet',
    'PlaneWaveStatLayer',
    'PlaneWaveRetLayer',
    'DipoleStatLayer',
    'DipoleRetLayer',
]
