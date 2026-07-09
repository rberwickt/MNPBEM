"""
BEM solvers for electromagnetic boundary element method.

Classes:
- BEMStat: BEM solver (quasistatic approximation)
- BEMRet: BEM solver (retarded/full Maxwell)
- BEMStatMirror: BEM solver (quasistatic + mirror symmetry)
- BEMRetMirror: BEM solver (retarded + mirror symmetry)
- BEMStatEig: BEM solver (quasistatic eigenmode expansion)
- BEMStatEigMirror: BEM solver (quasistatic eigenmode + mirror symmetry)
- BEMLayerMirror: BEM solver (layer + mirror symmetry, not implemented)
- BEMStatLayer: BEM solver (quasistatic + layer structure)
- BEMRetLayer: BEM solver (retarded + layer structure)
- BEMIter: Base class for iterative BEM solvers
- BEMStatIter: Iterative BEM solver (quasistatic approximation)
- BEMRetIter: Iterative BEM solver (retarded/full Maxwell)
- BEMRetLayerIter: Iterative BEM solver (retarded + layer structure)
"""

from .bem_stat import BEMStat
from .bem_ret import BEMRet
from .bem_stat_mirror import BEMStatMirror
from .bem_ret_mirror import BEMRetMirror
from .bem_stat_eig import BEMStatEig
from .bem_stat_eig_mirror import BEMStatEigMirror
from .bem_layer_mirror import BEMLayerMirror
from .bem_stat_layer import BEMStatLayer
from .bem_ret_layer import BEMRetLayer
from .bem_iter import BEMIter
from .bem_stat_iter import BEMStatIter
from .bem_ret_iter import BEMRetIter
from .bem_ret_layer_iter import BEMRetLayerIter
from .plasmonmode import plasmonmode
from .solver_factory import create_solver
from .bembase import BemBase

__all__ = [
    "BEMStat",
    "BEMRet",
    "BEMStatMirror",
    "BEMRetMirror",
    "BEMStatEig",
    "BEMStatEigMirror",
    "BEMLayerMirror",
    "BEMStatLayer",
    "BEMRetLayer",
    "BEMIter",
    "BEMStatIter",
    "BEMRetIter",
    "BEMRetLayerIter",
    "plasmonmode",
    "create_solver",
    "BemBase",
]
