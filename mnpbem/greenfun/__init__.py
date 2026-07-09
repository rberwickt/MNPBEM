"""
Green's functions for electromagnetic boundary element method.

Classes:
- GreenStat: Standalone quasistatic Green function G=1/r
- CompGreenStat: Composite Green function (quasistatic)
- CompGreenRet: Composite Green function (retarded)
- CompStruct: Structure for compound of points or particles
- CompGreenStatMirror: Composite Green function (quasistatic + mirror symmetry)
- CompGreenRetMirror: Composite Green function (retarded + mirror symmetry)
- CompGreenStatLayer: Composite Green function (quasistatic + layer)
- CompGreenRetLayer: Composite Green function (retarded + layer)
- CompGreenTabLayer: Composite Green function (retarded + tabulated layer)
- GreenRetLayer: Reflected Green function for layer structure
- GreenTabLayer: Tabulated Green function for layer structure
- ClusterTree: Cluster tree for hierarchical matrix bisection
- HMatrix: Hierarchical matrix with low-rank approximation
- ACACompGreenStat: ACA-accelerated composite Green function (quasistatic)
- ACACompGreenRet: ACA-accelerated composite Green function (retarded)
- ACACompGreenRetLayer: ACA-accelerated composite Green function (retarded + layer)
"""

from .greenstat import GreenStat
from .compgreen_stat import CompGreenStat, CompStruct
from .compgreen_ret import CompGreenRet
from .compgreen_stat_mirror import CompGreenStatMirror
from .compgreen_ret_mirror import CompGreenRetMirror
from .compgreen_stat_layer import CompGreenStatLayer
from .compgreen_ret_layer import CompGreenRetLayer
from .compgreentab_layer import CompGreenTabLayer
from .greenret_layer import GreenRetLayer
from .greentab_layer import GreenTabLayer
from .clustertree import ClusterTree
from .hmatrix import HMatrix
from .aca_compgreen_stat import ACACompGreenStat
from .aca_compgreen_ret import ACACompGreenRet
from .aca_compgreen_ret_layer import ACACompGreenRetLayer
from .greenfunction import greenfunction

__all__ = [
    "GreenStat",
    "CompGreenStat",
    "CompGreenRet",
    "CompStruct",
    "CompGreenStatMirror",
    "CompGreenRetMirror",
    "CompGreenStatLayer",
    "CompGreenRetLayer",
    "CompGreenTabLayer",
    "GreenRetLayer",
    "GreenTabLayer",
    "ClusterTree",
    "HMatrix",
    "ACACompGreenStat",
    "ACACompGreenRet",
    "ACACompGreenRetLayer",
    "greenfunction",
]
