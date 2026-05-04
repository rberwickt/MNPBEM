"""
MNPBEM - Metallic Nanoparticle Boundary Element Method
Python implementation of the MATLAB MNPBEM toolbox

Main modules:
- materials: Dielectric functions (EpsConst, EpsTable, EpsDrude)
- geometry: Particle geometries and mesh generation
- greenfun: Green's functions (static and retarded)
- bem: BEM solvers
- simulation: External excitations (plane wave, dipole, EELS)
- spectrum: Far-field and scattering cross sections
- mie: Mie theory for spherical and ellipsoidal particles
- misc: Math, distance, plotting, and other utilities
"""

__version__ = "1.6.2"

# Materials: dielectric functions
from .materials import EpsConst, EpsTable, EpsDrude, EpsFun, epsfun, EpsNonlocal, make_nonlocal_pair

# Geometry: particles, mesh generators, and connectivity
from .geometry import (
    Particle,
    ComParticle,
    ComParticleMirror,
    CompStructMirror,
    Point,
    ComPoint,
    EdgeProfile,
    Polygon,
    Polygon3,
    trisphere,
    trirod,
    tricube,
    tritorus,
    trispheresegment,
    trispherescale,
    tripolygon,
    fvgrid,
    connect,
)

# Green's functions: static, retarded, mirror, layer, ACA
from .greenfun import (
    GreenStat,
    CompGreenStat,
    CompGreenRet,
    CompStruct,
    CompGreenStatMirror,
    CompGreenRetMirror,
    CompGreenStatLayer,
    CompGreenRetLayer,
    CompGreenTabLayer,
    GreenRetLayer,
    GreenTabLayer,
    ClusterTree,
    HMatrix,
    ACACompGreenStat,
    ACACompGreenRet,
    ACACompGreenRetLayer,
    greenfunction,
)

# BEM: abstract base class
from .bem import BemBase

# BEM solvers: static, retarded, mirror, layer, iterative
from .bem import (
    BEMStat,
    BEMRet,
    BEMStatMirror,
    BEMRetMirror,
    BEMStatEig,
    BEMStatEigMirror,
    BEMLayerMirror,
    BEMStatLayer,
    BEMRetLayer,
    BEMIter,
    BEMStatIter,
    BEMRetIter,
    BEMRetLayerIter,
    plasmonmode,
)

# Simulation: plane wave, dipole, EELS excitations
from .simulation import (
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
    MeshField,
    dipole,
    planewave,
    electronbeam,
)

# Spectrum: far-field and cross section calculations
from .spectrum import (
    SpectrumRet,
    SpectrumStat,
    SpectrumRetLayer,
    SpectrumStatLayer,
    spectrum,
)

# Mie theory: spherical harmonics, Mie solvers
from .mie import (
    spharm,
    sphtable,
    vecspharm,
    MieGans,
    MieStat,
    MieRet,
    mie_solver,
)

# Misc: math, distance, units, options, shapes, plotting, etc.
from .misc import (
    matmul,
    inner,
    outer,
    matcross,
    vec_norm,
    vec_normalize,
    spdiag,
    pdist2,
    bradius,
    bdist2,
    distmin3,
    EV2NM,
    BOHR,
    HARTREE,
    FINE,
    bemoptions,
    getbemoptions,
    getfields,
    Tri,
    Quad,
    lglnodes,
    lgwt,
    IGrid2,
    IGrid3,
    ValArray,
    VecArray,
    QuadFace,
    triangle_unit_set,
    trisubdivide,
    BemPlot,
    arrowplot,
    coneplot,
    coneplot2,
    mycolormap,
    particlecursor,
    nettable,
    patchcurvature,
    memsize,
    round_left,
    Mem,
    multi_waitbar,
)

# Utils: parallel computation
from .utils import (
    compute_spectrum,
    compute_spectrum_parallel,
)

__all__ = [
    # Materials
    "EpsConst",
    "EpsTable",
    "EpsDrude",
    "EpsFun",
    "epsfun",
    "EpsNonlocal",
    "make_nonlocal_pair",
    # Geometry
    "Particle",
    "ComParticle",
    "ComParticleMirror",
    "CompStructMirror",
    "Point",
    "ComPoint",
    "EdgeProfile",
    "Polygon",
    "Polygon3",
    "trisphere",
    "trirod",
    "tricube",
    "tritorus",
    "trispheresegment",
    "trispherescale",
    "tripolygon",
    "fvgrid",
    "connect",
    # Green's functions
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
    # BEM base
    "BemBase",
    # BEM solvers
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
    # Simulation
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
    "dipole",
    "planewave",
    "electronbeam",
    # Spectrum
    "SpectrumRet",
    "SpectrumStat",
    "SpectrumRetLayer",
    "SpectrumStatLayer",
    "spectrum",
    # Mie theory
    "spharm",
    "sphtable",
    "vecspharm",
    "MieGans",
    "MieStat",
    "MieRet",
    "mie_solver",
    # Misc: math utilities
    "matmul",
    "inner",
    "outer",
    "matcross",
    "vec_norm",
    "vec_normalize",
    "spdiag",
    # Misc: distance utilities
    "pdist2",
    "bradius",
    "bdist2",
    "distmin3",
    # Misc: units / constants
    "EV2NM",
    "BOHR",
    "HARTREE",
    "FINE",
    # Misc: options
    "bemoptions",
    "getbemoptions",
    "getfields",
    # Misc: shapes
    "Tri",
    "Quad",
    # Misc: Gauss-Legendre
    "lglnodes",
    "lgwt",
    # Misc: grids
    "IGrid2",
    "IGrid3",
    # Misc: arrays
    "ValArray",
    "VecArray",
    # Misc: quadface
    "QuadFace",
    "triangle_unit_set",
    "trisubdivide",
    # Misc: plotting
    "BemPlot",
    "arrowplot",
    "coneplot",
    "coneplot2",
    "mycolormap",
    "particlecursor",
    # Misc: other utilities
    "nettable",
    "patchcurvature",
    "memsize",
    "round_left",
    "Mem",
    "multi_waitbar",
    # Utils: parallel
    "compute_spectrum",
    "compute_spectrum_parallel",
]
