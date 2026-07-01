"""
Miscellaneous utilities for MNPBEM.

MATLAB: Misc/
"""

from .math_utils import matmul, inner, outer, matcross, vec_norm, vec_normalize, spdiag
from .distance_utils import pdist2, bradius, bdist2, distmin3
from .units import EV2NM, BOHR, HARTREE, FINE
from .options import bemoptions, getbemoptions, getfields
from .shapes import Tri, Quad
from .gauss_legendre import lglnodes, lgwt
from .igrid import IGrid2, IGrid3
from .valarray import ValArray
from .vecarray import VecArray
from .quadface_misc import QuadFace, triangle_unit_set, trisubdivide
from .bemplot import BemPlot
from .plotting import arrowplot, coneplot, coneplot2, mycolormap, particlecursor
from .misc_utils import nettable, patchcurvature, memsize, round_left, Mem, multi_waitbar


__all__ = [
    # math_utils
    'matmul', 'inner', 'outer', 'matcross',
    'vec_norm', 'vec_normalize', 'spdiag',
    # distance_utils
    'pdist2', 'bradius', 'bdist2', 'distmin3',
    # units
    'EV2NM', 'BOHR', 'HARTREE', 'FINE',
    # options
    'bemoptions', 'getbemoptions', 'getfields',
    # shapes
    'Tri', 'Quad',
    # gauss_legendre
    'lglnodes', 'lgwt',
    # igrid
    'IGrid2', 'IGrid3',
    # valarray
    'ValArray',
    # vecarray
    'VecArray',
    # quadface
    'QuadFace', 'triangle_unit_set', 'trisubdivide',
    # bemplot
    'BemPlot',
    # plotting
    'arrowplot', 'coneplot', 'coneplot2', 'mycolormap', 'particlecursor',
    # misc_utils
    'nettable', 'patchcurvature', 'memsize', 'round_left',
    'Mem', 'multi_waitbar',
]
