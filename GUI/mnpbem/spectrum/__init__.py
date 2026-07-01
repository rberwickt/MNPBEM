"""
Spectrum module for computing far-fields and scattering cross sections.

Provides:
- SpectrumRet: For full Maxwell equations (retarded case)
- SpectrumStat: For quasistatic approximation
- SpectrumRetLayer: For retarded case with layer structure
- SpectrumStatLayer: For quasistatic case with layer structure
"""

from .spectrum_ret import SpectrumRet
from .spectrum_stat import SpectrumStat
from .spectrum_ret_layer import SpectrumRetLayer
from .spectrum_stat_layer import SpectrumStatLayer
from .spectrum_factory import spectrum

__all__ = [
    'SpectrumRet',
    'SpectrumStat',
    'SpectrumRetLayer',
    'SpectrumStatLayer',
    'spectrum',
]
